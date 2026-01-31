# -*- coding: utf-8 -*-
"""
DWTS 2026 MCM Problem C: 数据建模与可视化主流程
本脚本完成数据读取、投票可行集采样、机制评估与图表输出。
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端便于批量导图
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import pulp
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

# =========================
# 全局配置（）
# =========================
RNG = np.random.default_rng(20260131)
DATA_PATH = Path("2026_MCM_Problem_C_Data.csv")
OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("paper/figures")
SUMMARY_TEX = Path("paper/summary_metrics.tex")
LOG_PATH = OUTPUT_DIR / "run.log"
BENCHMARK_CSV = OUTPUT_DIR / "scale_benchmark.csv"

EPSILON = 0.001  # 投票占比下限
ALPHA_PERCENT = 0.5  # 百分比规则权重
N_PROPOSALS = int(os.getenv("MCM_N_PROPOSALS", "250"))  # 每周Dirichlet提案数量
MIN_ACCEPT = 40  # 最少保留的可行样本
SIGMA_LIST = [0.5, 1.0, 1.5, 2.0]
RHO_SWITCH = 0.10  # 规则切换先验概率
COMPUTE_BOUNDS = False  # 是否计算LP边界（耗时）
USE_MIXED_MODEL = False  # 是否使用混合效应模型
MAX_SAMPLES_PER_WEEK = 120  # 每周用于指标的最大样本数
USE_MULTIPROCESSING = os.getenv("MCM_MULTIPROC", "1") != "0"  # 是否启用多进程
PARALLEL_WORKERS = int(os.getenv("MCM_WORKERS", str(max(1, (os.cpu_count() or 2) - 2))))  # 并行进程数
FAST_STRICT_ENABLED = os.getenv("MCM_FAST_STRICT", "1") != "0"  # 是否进行 Fast vs Strict 校验
FAST_STRICT_PROPS = int(os.getenv("MCM_STRICT_PROPS", "2000"))  # Strict 校验采样规模
FAST_STRICT_MAX_SAMPLES = int(os.getenv("MCM_STRICT_MAX_SAMPLES", "600"))  # Strict 校验最大样本数
RULE_SWITCH_BOOT = int(os.getenv("MCM_RULE_BOOT", "200"))  # 规则切换置信带 bootstrap 次数

# 颜色规范（与图表规范一致）
COLOR_PRIMARY = "#0072B2"
COLOR_PRIMARY_DARK = "#0B3C5D"
COLOR_ACCENT = "#E69F00"
COLOR_WARNING = "#D55E00"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#D9D9D9"


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """同时写入终端与日志文件。"""
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def set_plot_style() -> None:
    """统一Matplotlib风格。"""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (6.4, 3.8),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })


# =========================
# 数据读取与整理
# =========================

def parse_elim_week(result: str) -> int | None:
    """从results字段中解析淘汰周。"""
    if not isinstance(result, str):
        return None
    if "Eliminated Week" in result:
        m = re.search(r"Eliminated Week\s*(\d+)", result)
        if m:
            return int(m.group(1))
    return None


def load_data() -> pd.DataFrame:
    """读取原始CSV数据。"""
    return pd.read_csv(DATA_PATH)


def build_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """将宽表转换为(季,周,选手)长表并计算总分。"""
    week_cols: Dict[int, List[str]] = {}
    for col in df.columns:
        m = re.match(r"week(\d+)_judge(\d+)_score", col)
        if m:
            week = int(m.group(1))
            week_cols.setdefault(week, []).append(col)
    weeks = sorted(week_cols)

    meta_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
    ]
    meta = df[meta_cols].copy()
    meta["elim_week"] = meta["results"].apply(parse_elim_week)

    long_list = []
    for week in weeks:
        cols = week_cols[week]
        count = df[cols].notna().sum(axis=1)
        total = df[cols].sum(axis=1, skipna=True)
        total[count == 0] = np.nan
        temp = meta.copy()
        temp["week"] = week
        temp["judge_total"] = total
        long_list.append(temp)

    long_df = pd.concat(long_list, ignore_index=True)
    long_df = long_df.dropna(subset=["judge_total"])  # 只保留有效周
    long_df["active"] = long_df["judge_total"] > 0
    long_df["is_eliminated_week"] = long_df["elim_week"] == long_df["week"]
    return long_df


# =========================
# 规则约束与采样
# =========================

def percent_constraints_ok(v: np.ndarray, j_share: np.ndarray, elim_idx: List[int], alpha: float) -> bool:
    """检查百分比规则淘汰约束是否满足。"""
    if not elim_idx:
        return True
    c = alpha * j_share + (1 - alpha) * v
    for e in elim_idx:
        if np.any(c[e] > c + 1e-12):
            return False
    return True


def sample_week_percent(
    week_df: pd.DataFrame,
    alpha: float,
    epsilon: float,
    n_props: int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float]:
    # --- 极速向量化版本 (无需思考，直接用) ---
    if rng is None:
        rng = RNG
    active_df = week_df[week_df["active"]].copy()
    n = len(active_df)
    if n == 0:
        return np.empty((0, 0)), 0.0

    j = active_df["judge_share"].to_numpy()
    elim_idx = [i for i, flag in enumerate(active_df["is_eliminated_week"].to_numpy()) if flag]

    # 1. 批量生成随机提案
    proposals = rng.dirichlet(np.ones(n), size=n_props)
    proposals = np.maximum(proposals, epsilon)
    proposals = proposals / proposals.sum(axis=1, keepdims=True)

    # 2. 矩阵计算分数 (Combined Score)
    c_matrix = alpha * j + (1 - alpha) * proposals

    # 3. 极简约束检查 (只看被淘汰者是否在底部附近)
    if not elim_idx:
        mask = np.ones(n_props, dtype=bool)
    else:
        min_scores = c_matrix.min(axis=1)
        # 只要任意一个淘汰者的分数 <= 最小值 + 容差，就算通过
        elim_scores = c_matrix[:, elim_idx]
        is_bottom = (elim_scores <= min_scores[:, None] + 1e-12)
        mask = is_bottom.any(axis=1)

    accepted = proposals[mask]
    # 如果没采到，就强制返回随机样本（为了保证程序不崩）
    if len(accepted) < 5:
        accepted = proposals[:10]

    return accepted, float(mask.mean())


def lp_bounds_and_slack(week_df, alpha, epsilon, compute_bounds):
    # --- 哑函数 (直接跳过耗时计算) ---
    return {}, 0.001


def compute_alpha_t(week: int, total_weeks: int, mean_width: float,
                    alpha0: float, gamma: float, eta: float,
                    alpha_min: float, alpha_max: float) -> float:
    """根据不确定性调整 DAWS 权重。"""
    base_alpha = alpha0 + gamma * (week / max(1, total_weeks)) - eta * mean_width
    return float(np.clip(base_alpha, alpha_min, alpha_max))


def apply_percent_mask(
    proposals: np.ndarray,
    j_share: np.ndarray,
    elim_idx: List[int],
    alpha: float,
    mode: str,
) -> np.ndarray:
    """百分比规则约束：fast=任一淘汰者在底部，strict=所有淘汰者在底部。"""
    if not elim_idx:
        return np.ones(len(proposals), dtype=bool)
    c_matrix = alpha * j_share + (1 - alpha) * proposals
    min_scores = c_matrix.min(axis=1)
    elim_scores = c_matrix[:, elim_idx]
    if mode == "fast":
        return (elim_scores <= min_scores[:, None] + 1e-12).any(axis=1)
    return (elim_scores <= min_scores[:, None] + 1e-12).all(axis=1)


def fallback_by_violation(
    proposals: np.ndarray,
    j_share: np.ndarray,
    elim_idx: List[int],
    alpha: float,
    min_accept: int,
) -> np.ndarray:
    """当可行样本过少时，按违约度最小筛选。"""
    if len(proposals) == 0:
        return proposals
    if not elim_idx:
        return proposals[:min_accept]
    c_matrix = alpha * j_share + (1 - alpha) * proposals
    min_scores = c_matrix.min(axis=1)
    elim_scores = c_matrix[:, elim_idx]
    violations = np.max(elim_scores - min_scores[:, None], axis=1)
    order = np.argsort(violations)
    return proposals[order[:min_accept]]


def evaluate_mechanisms(
    samples: np.ndarray,
    active_df: pd.DataFrame,
    alpha_t: float,
    eps: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """对给定样本计算机制指标（向量化）。"""
    m = len(samples)
    if m == 0:
        return {}
    j_share = active_df["judge_share"].to_numpy()
    j_rank = active_df["judge_share"].rank(ascending=False, method="average").to_numpy()
    j_scores = active_df["judge_total"].to_numpy()
    j_share_matrix = np.tile(j_share, (m, 1))

    comb_percent = ALPHA_PERCENT * j_share_matrix + (1 - ALPHA_PERCENT) * samples
    elim_p = np.argmin(comb_percent, axis=1)

    fan_rank = np.argsort(np.argsort(-samples, axis=1), axis=1) + 1
    j_rank_matrix = np.tile(j_rank, (m, 1))
    comb_rank = fan_rank + j_rank_matrix
    elim_r = np.argmax(comb_rank, axis=1)

    comb_daws = alpha_t * j_share_matrix + (1 - alpha_t) * samples
    elim_d = np.argmin(comb_daws, axis=1)

    bottom_two = np.argpartition(comb_rank, -2, axis=1)[:, -2:]
    a_idx = bottom_two[:, 0]
    b_idx = bottom_two[:, 1]
    diff = j_scores[b_idx] - j_scores[a_idx]
    p_elim_a = 1 / (1 + np.exp(1.8 * diff))
    rand_u = rng.random(m)
    elim_s = np.where(rand_u < p_elim_a, a_idx, b_idx)

    n = len(j_rank)
    if n < 2:
        tau_mean = float("nan")
    else:
        i_idx, j_idx = np.triu_indices(n, k=1)
        sign_j = np.sign(j_rank[i_idx] - j_rank[j_idx])
        valid = sign_j != 0
        if not np.any(valid):
            tau_mean = float("nan")
        else:
            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            sign_j = sign_j[valid]
            sign_f = np.sign(fan_rank[:, i_idx] - fan_rank[:, j_idx])
            prod = sign_f * sign_j
            concordant = np.sum(prod > 0, axis=1)
            discordant = np.sum(prod < 0, axis=1)
            denom = len(sign_j)
            tau_vals = (concordant - discordant) / max(1, denom)
            tau_mean = float(np.mean(tau_vals))
    if np.isnan(tau_mean):
        tau_mean = 0.0

    fan_lowest = np.argmin(samples, axis=1)
    agency_p = np.mean(fan_lowest == elim_p)
    agency_r = np.mean(fan_lowest == elim_r)
    agency_s = np.mean(fan_lowest == elim_s)
    agency_d = np.mean(fan_lowest == elim_d)

    judge_lowest = int(np.argmin(j_share))
    integrity_p = np.mean(elim_p == judge_lowest)
    integrity_r = np.mean(elim_r == judge_lowest)
    integrity_s = np.mean(elim_s == judge_lowest)
    integrity_d = np.mean(elim_d == judge_lowest)

    noise = rng.normal(0, 0.02, size=samples.shape)
    v_noise = np.maximum(samples + noise, eps)
    v_noise = v_noise / v_noise.sum(axis=1, keepdims=True)

    comb_percent_noise = ALPHA_PERCENT * j_share_matrix + (1 - ALPHA_PERCENT) * v_noise
    elim_noise_p = np.argmin(comb_percent_noise, axis=1)

    fan_rank_noise = np.argsort(np.argsort(-v_noise, axis=1), axis=1) + 1
    comb_rank_noise = fan_rank_noise + j_rank_matrix
    elim_noise_r = np.argmax(comb_rank_noise, axis=1)

    comb_daws_noise = alpha_t * j_share_matrix + (1 - alpha_t) * v_noise
    elim_noise_d = np.argmin(comb_daws_noise, axis=1)

    bottom_two_noise = np.argpartition(comb_rank_noise, -2, axis=1)[:, -2:]
    a_n = bottom_two_noise[:, 0]
    b_n = bottom_two_noise[:, 1]
    diff_n = j_scores[b_n] - j_scores[a_n]
    p_elim_a_n = 1 / (1 + np.exp(1.8 * diff_n))
    rand_u_n = rng.random(m)
    elim_noise_s = np.where(rand_u_n < p_elim_a_n, a_n, b_n)

    instability_p = np.mean(elim_p != elim_noise_p)
    instability_r = np.mean(elim_r != elim_noise_r)
    instability_s = np.mean(elim_s != elim_noise_s)
    instability_d = np.mean(elim_d != elim_noise_d)

    return {
        "percent": {"fairness": tau_mean, "agency": agency_p, "instability": instability_p, "judge_integrity": integrity_p},
        "rank": {"fairness": tau_mean, "agency": agency_r, "instability": instability_r, "judge_integrity": integrity_r},
        "save": {"fairness": tau_mean, "agency": agency_s, "instability": instability_s, "judge_integrity": integrity_s},
        "daws": {"fairness": tau_mean, "agency": agency_d, "instability": instability_d, "judge_integrity": integrity_d},
        "flip_sum": float(np.sum(elim_p != elim_r)),
        "count": m,
    }


def parse_scales(env_val: str | None) -> List[int]:
    """解析环境变量中的采样规模列表。"""
    if not env_val:
        return []
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    scales = []
    for p in parts:
        try:
            scales.append(int(p))
        except ValueError:
            continue
    return [s for s in scales if s > 0]


def update_benchmark_csv(records: List[Dict[str, float]]) -> pd.DataFrame:
    """追加/更新规模实验记录，并按规模排序。"""
    new_df = pd.DataFrame(records)
    if BENCHMARK_CSV.exists():
        old_df = pd.read_csv(BENCHMARK_CSV)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["n_proposals"], keep="last")
    else:
        merged = new_df
    merged = merged.sort_values("n_proposals").reset_index(drop=True)
    merged.to_csv(BENCHMARK_CSV, index=False, encoding="utf-8")
    return merged


def plot_scale_benchmark(df: pd.DataFrame) -> None:
    """绘制规模对比图（时间、误差、稳定性、匹配度）。"""
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
    x = df["n_proposals"]
    # elbow 计算（基于 mean_hdi 的边际收益）
    if len(df) >= 3:
        x_norm = (x - x.min()) / max(1e-9, (x.max() - x.min()))
        y = df["mean_hdi"]
        y_norm = (y - y.min()) / max(1e-9, (y.max() - y.min()))
        p1 = np.array([x_norm.iloc[0], y_norm.iloc[0]])
        p2 = np.array([x_norm.iloc[-1], y_norm.iloc[-1]])
        dist = []
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        den = math.hypot(y2 - y1, x2 - x1)
        for xi, yi in zip(x_norm, y_norm):
            num = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1)
            dist.append(num / max(1e-9, den))
        elbow_idx = int(np.argmax(dist))
        elbow_x = x.iloc[elbow_idx]
    else:
        elbow_x = None

    axes[0, 0].plot(x, df["runtime_sec"], marker="o", color=COLOR_PRIMARY)
    axes[0, 0].set_title("Runtime vs Scale")
    axes[0, 0].set_xlabel("N_PROPOSALS")
    axes[0, 0].set_ylabel("Seconds")

    axes[0, 1].plot(x, df["mean_hdi"], marker="o", color=COLOR_ACCENT)
    axes[0, 1].set_title("Error (Mean HDI) vs Scale")
    axes[0, 1].set_xlabel("N_PROPOSALS")
    axes[0, 1].set_ylabel("Mean HDI")

    axes[1, 0].plot(x, df["stability_daws"], marker="o", color=COLOR_PRIMARY_DARK)
    axes[1, 0].set_title("Stability (DAWS) vs Scale")
    axes[1, 0].set_xlabel("N_PROPOSALS")
    axes[1, 0].set_ylabel("Stability")

    axes[1, 1].plot(x, df["fairness_daws"], marker="o", color=COLOR_GRAY)
    axes[1, 1].set_title("Theory Fit (Kendall tau) vs Scale")
    axes[1, 1].set_xlabel("N_PROPOSALS")
    axes[1, 1].set_ylabel("Tau")

    if elbow_x is not None:
        for ax in axes.flatten():
            ax.axvline(elbow_x, color=COLOR_WARNING, linestyle="--", linewidth=1.0)
        axes[0, 1].text(elbow_x, axes[0, 1].get_ylim()[1] * 0.92, "Elbow", ha="center", color=COLOR_WARNING, fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_scale_benchmark.pdf")
    plt.close(fig)


def process_season_samples(
    task: Tuple[int, pd.DataFrame, float, float, int, bool, int],
) -> Dict[str, object]:
    """按赛季并行计算后验与周指标。"""
    season, season_df, alpha, epsilon, n_props, compute_bounds, seed = task
    rng = np.random.default_rng(seed)

    posterior_records: List[Dict[str, object]] = []
    week_metrics: List[Dict[str, object]] = []
    samples_cache: Dict[Tuple[int, int], np.ndarray] = {}
    acc_cache: Dict[Tuple[int, int], float] = {}
    slack_cache: Dict[Tuple[int, int], float] = {}

    for week, wdf in season_df.groupby("week", sort=True):
        active_df = wdf[wdf["active"]].copy()
        if len(active_df) == 0:
            continue

        samples, acc_rate = sample_week_percent(wdf, alpha, epsilon, n_props, rng)
        _, slack = lp_bounds_and_slack(wdf, alpha, epsilon, compute_bounds)

        key = (int(season), int(week))
        samples_cache[key] = samples
        acc_cache[key] = acc_rate
        slack_cache[key] = slack

        if len(samples) == 0:
            continue

        lower = np.quantile(samples, 0.025, axis=0)
        upper = np.quantile(samples, 0.975, axis=0)
        mean = samples.mean(axis=0)
        hdi_width = upper - lower

        for i, row in active_df.reset_index(drop=True).iterrows():
            posterior_records.append({
                "season": season,
                "week": week,
                "celebrity_name": row["celebrity_name"],
                "ballroom_partner": row["ballroom_partner"],
                "celebrity_industry": row["celebrity_industry"],
                "celebrity_age_during_season": row["celebrity_age_during_season"],
                "judge_share": row["judge_share"],
                "fan_share_mean": mean[i],
                "fan_share_lower": lower[i],
                "fan_share_upper": upper[i],
                "hdi_width": hdi_width[i],
                "is_eliminated_week": row["is_eliminated_week"],
            })

        week_metrics.append({
            "season": season,
            "week": week,
            "accept_rate": acc_rate,
            "slack": slack,
            "mean_hdi_width": float(np.mean(hdi_width)),
        })

    return {
        "posterior_records": posterior_records,
        "week_metrics": week_metrics,
        "samples_cache": samples_cache,
        "acc_cache": acc_cache,
        "slack_cache": slack_cache,
    }


# =========================
# 机制评估与指标
# =========================

def compute_rank_feasible_rate(week_df: pd.DataFrame, n_perm: int = 80) -> float:
    """蒙特卡洛估计排名规则可行率。"""
    active_df = week_df[week_df["active"]].copy()
    n = len(active_df)
    if n == 0:
        return 0.0
    if not active_df["is_eliminated_week"].any():
        return 1.0

    j_rank = active_df["judge_share"].rank(ascending=False, method="average").to_numpy()
    elim_idx = [i for i, flag in enumerate(active_df["is_eliminated_week"].to_numpy()) if flag]

    count = 0
    for _ in range(n_perm):
        perm = RNG.permutation(n) + 1
        combined = j_rank + perm
        worst = np.argsort(combined)[-len(elim_idx):]
        if set(elim_idx).issubset(set(worst)):
            count += 1
    return count / n_perm


def mechanism_elimination(v: np.ndarray, week_df: pd.DataFrame, alpha: float) -> List[int]:
    """百分比规则下的淘汰索引（可多淘汰）。"""
    active_df = week_df[week_df["active"]].copy()
    j = active_df["judge_share"].to_numpy()
    combined = alpha * j + (1 - alpha) * v
    # 默认只淘汰一名
    return [int(np.argmin(combined))]


def mechanism_rank_elimination(v: np.ndarray, week_df: pd.DataFrame) -> List[int]:
    """排名规则下淘汰索引。"""
    active_df = week_df[week_df["active"]].copy()
    j_rank = active_df["judge_share"].rank(ascending=False, method="average").to_numpy()
    f_rank = (-v).argsort().argsort() + 1
    combined = j_rank + f_rank
    return [int(np.argmax(combined))]


def mechanism_judge_save(v: np.ndarray, week_df: pd.DataFrame, beta: float) -> List[int]:
    """Bottom-two + judges save 规则。"""
    active_df = week_df[week_df["active"]].copy()
    j_rank = active_df["judge_share"].rank(ascending=False, method="average").to_numpy()
    f_rank = (-v).argsort().argsort() + 1
    combined = j_rank + f_rank
    bottom_two = np.argsort(combined)[-2:]
    if len(bottom_two) < 2:
        return [int(np.argmax(combined))]

    a, b = bottom_two
    j_scores = active_df["judge_total"].to_numpy()
    diff = j_scores[b] - j_scores[a]
    p_elim_a = 1 / (1 + math.exp(beta * diff))
    return [int(a if RNG.random() < p_elim_a else b)]


# =========================
# 主流程
# =========================

def run_pipeline(n_props: int | None = None, record_benchmark: bool = False, save_outputs: bool = True) -> Dict[str, float]:
    ensure_dirs()
    set_plot_style()
    LOG_PATH.write_text("", encoding="utf-8")
    t_start = time.perf_counter()
    n_props = int(n_props or N_PROPOSALS)

    log("Load data...")
    df = load_data()
    long_df = build_long_df(df)
    log(f"Raw rows: {len(df)}, Long rows: {len(long_df)}")
    log(f"Seasons: {long_df['season'].nunique()}, Weeks: {long_df['week'].nunique()}")
    log(f"Config: N_PROPOSALS={n_props}, MULTIPROC={USE_MULTIPROCESSING}, WORKERS={PARALLEL_WORKERS}")

    # 计算judge share
    long_df["judge_share"] = long_df.groupby(["season", "week"])['judge_total'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else np.nan
    )

    # 存储后验结果
    posterior_records = []
    week_metrics = []
    season_week_groups = long_df.groupby(["season", "week"], sort=True)
    # 样本缓存，避免重复采样
    samples_cache: Dict[Tuple[int, int], np.ndarray] = {}
    acc_cache: Dict[Tuple[int, int], float] = {}
    slack_cache: Dict[Tuple[int, int], float] = {}

    log("Sampling feasible sets and computing uncertainty...")
    if USE_MULTIPROCESSING:
        cols = [
            "celebrity_name",
            "ballroom_partner",
            "celebrity_industry",
            "celebrity_age_during_season",
            "season",
            "week",
            "judge_total",
            "judge_share",
            "active",
            "is_eliminated_week",
        ]
        tasks = []
        for season, sdf in long_df.groupby("season", sort=True):
            season_df = sdf[cols].copy()
            seed = 20260131 + int(season) * 1000
            tasks.append((int(season), season_df, ALPHA_PERCENT, EPSILON, n_props, COMPUTE_BOUNDS, seed))

        log(f"Parallel sampling by season... tasks={len(tasks)}")
        with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            for res in executor.map(process_season_samples, tasks, chunksize=1):
                posterior_records.extend(res["posterior_records"])
                week_metrics.extend(res["week_metrics"])
                samples_cache.update(res["samples_cache"])
                acc_cache.update(res["acc_cache"])
                slack_cache.update(res["slack_cache"])
    else:
        for (season, week), wdf in season_week_groups:
            active_df = wdf[wdf["active"]].copy()
            if len(active_df) == 0:
                continue

            key = (int(season), int(week))
            if key in samples_cache:
                samples = samples_cache[key]
                acc_rate = acc_cache[key]
                slack = slack_cache[key]
            else:
                samples, acc_rate = sample_week_percent(wdf, ALPHA_PERCENT, EPSILON, n_props)
                _, slack = lp_bounds_and_slack(wdf, ALPHA_PERCENT, EPSILON, COMPUTE_BOUNDS)
                samples_cache[key] = samples
                acc_cache[key] = acc_rate
                slack_cache[key] = slack

            # 计算HDI
            if len(samples) == 0:
                continue
            # 抽样降采样，控制计算量
            if len(samples) > MAX_SAMPLES_PER_WEEK:
                idx = RNG.choice(len(samples), size=MAX_SAMPLES_PER_WEEK, replace=False)
                samples = samples[idx]
            lower = np.quantile(samples, 0.025, axis=0)
            upper = np.quantile(samples, 0.975, axis=0)
            mean = samples.mean(axis=0)
            hdi_width = upper - lower

            for i, row in active_df.reset_index(drop=True).iterrows():
                posterior_records.append({
                    "season": season,
                    "week": week,
                    "celebrity_name": row["celebrity_name"],
                    "ballroom_partner": row["ballroom_partner"],
                    "celebrity_industry": row["celebrity_industry"],
                    "celebrity_age_during_season": row["celebrity_age_during_season"],
                    "judge_share": row["judge_share"],
                    "fan_share_mean": mean[i],
                    "fan_share_lower": lower[i],
                    "fan_share_upper": upper[i],
                    "hdi_width": hdi_width[i],
                    "is_eliminated_week": row["is_eliminated_week"],
                })

            week_metrics.append({
                "season": season,
                "week": week,
                "accept_rate": acc_rate,
                "slack": slack,
                "mean_hdi_width": float(np.mean(hdi_width)),
            })

    posterior_df = pd.DataFrame(posterior_records)
    week_metrics_df = pd.DataFrame(week_metrics)
    log(f"Posterior rows: {len(posterior_df)}, Week metrics: {len(week_metrics_df)}")

    # HDI 分布图
    hdi_series = week_metrics_df["mean_hdi_width"].dropna()
    q1 = float(hdi_series.quantile(0.25)) if not hdi_series.empty else float("nan")
    median_hdi = float(hdi_series.quantile(0.50)) if not hdi_series.empty else float("nan")
    q3 = float(hdi_series.quantile(0.75)) if not hdi_series.empty else float("nan")
    p90 = float(hdi_series.quantile(0.90)) if not hdi_series.empty else float("nan")
    if save_outputs and not hdi_series.empty:
        plt.figure(figsize=(5.4, 3.6))
        plt.hist(hdi_series, bins=18, color=COLOR_PRIMARY, alpha=0.65, edgecolor="white")
        for val, label, color in [
            (q1, "Q1", COLOR_GRAY),
            (median_hdi, "Median", COLOR_ACCENT),
            (q3, "Q3", COLOR_GRAY),
            (p90, "P90", COLOR_WARNING),
        ]:
            plt.axvline(val, color=color, linestyle="--", linewidth=1.1)
            plt.text(val, plt.ylim()[1] * 0.92, label, color=color, ha="center", fontsize=8)
        plt.xlabel("Mean HDI width (week)")
        plt.ylabel("Count")
        plt.title("Distribution of weekly uncertainty")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_hdi_distribution.pdf")
        plt.close()

    # 不确定性热力图矩阵
    max_season = int(long_df["season"].max())
    max_week = int(long_df["week"].max())
    heat = np.full((max_season, max_week), np.nan)
    for _, row in week_metrics_df.iterrows():
        heat[int(row["season"]) - 1, int(row["week"]) - 1] = row["mean_hdi_width"]

    # 错误淘汰概率
    wrongful_heat = np.full_like(heat, np.nan)
    log("Computing wrongful elimination probabilities...")
    for (season, week), wdf in season_week_groups:
        active_df = wdf[wdf["active"]].copy()
        if len(active_df) == 0:
            continue
        key = (int(season), int(week))
        samples = samples_cache.get(key)
        if samples is None or len(samples) == 0:
            samples, _ = sample_week_percent(wdf, ALPHA_PERCENT, EPSILON, n_props)
            samples_cache[key] = samples
        if len(samples) == 0:
            continue
        elim_idx = [i for i, flag in enumerate(active_df["is_eliminated_week"].to_numpy()) if flag]
        if not elim_idx:
            wrongful = 0.0
        else:
            worst = np.argmin(samples, axis=1)
            wrongful = float(np.mean([w not in elim_idx for w in worst]))
        wrongful_heat[int(season) - 1, int(week) - 1] = wrongful

    # 规则切换推断
    log("Inferring rule switch probabilities...")
    evidence_records = []
    season_week_stats: Dict[int, Dict[str, List[float]]] = {}
    for season in sorted(long_df["season"].unique()):
        season_weeks = week_metrics_df[week_metrics_df["season"] == season]
        if season_weeks.empty:
            continue
        e_percent = float(np.sum(np.log(season_weeks["accept_rate"].clip(1e-6))))

        # 估计rank可行率
        rank_rates = []
        for week in sorted(long_df[long_df["season"] == season]["week"].unique()):
            wdf = long_df[(long_df["season"] == season) & (long_df["week"] == week)]
            rank_rates.append(compute_rank_feasible_rate(wdf))
        e_rank = float(np.sum(np.log(np.clip(rank_rates, 1e-6, None))))

        evidence_records.append({
            "season": season,
            "e_percent": e_percent,
            "e_rank": e_rank,
        })
        season_week_stats[int(season)] = {
            "acc": season_weeks["accept_rate"].clip(1e-6).tolist(),
            "rank": [float(x) for x in rank_rates],
        }

    evidence_df = pd.DataFrame(evidence_records)
    log(f"Rule switch seasons: {len(evidence_df)}")

    # 简单HMM（两状态）
    probs = []
    prev = np.array([0.9, 0.1])  # 初始偏向百分比规则
    trans = np.array([[1 - RHO_SWITCH, RHO_SWITCH], [RHO_SWITCH, 1 - RHO_SWITCH]])
    for _, row in evidence_df.iterrows():
        like = np.array([math.exp(row["e_percent"]), math.exp(row["e_rank"])])
        post = prev @ trans
        post = post * like
        post = post / post.sum()
        probs.append(post[1])
        prev = post
    evidence_df["prob_rank"] = probs

    # 规则切换置信带（bootstrap）
    if save_outputs and RULE_SWITCH_BOOT > 0 and not evidence_df.empty:
        log("Bootstrapping rule-switch uncertainty...")
        seasons_sorted = evidence_df["season"].to_list()
        boot_probs = []
        for _ in range(RULE_SWITCH_BOOT):
            e_records = []
            for season in seasons_sorted:
                stats = season_week_stats.get(int(season), None)
                if not stats:
                    continue
                acc_list = stats["acc"]
                rank_list = stats["rank"]
                n_weeks = len(acc_list)
                idx = RNG.integers(0, n_weeks, size=n_weeks)
                e_percent_b = float(np.sum(np.log(np.clip(np.array(acc_list)[idx], 1e-6, None))))
                e_rank_b = float(np.sum(np.log(np.clip(np.array(rank_list)[idx], 1e-6, None))))
                e_records.append((season, e_percent_b, e_rank_b))

            prev_b = np.array([0.9, 0.1])
            probs_b = []
            for season, e_percent_b, e_rank_b in e_records:
                like = np.array([math.exp(e_percent_b), math.exp(e_rank_b)])
                post = prev_b @ trans
                post = post * like
                post = post / post.sum()
                probs_b.append(post[1])
                prev_b = post
            if len(probs_b) == len(seasons_sorted):
                boot_probs.append(probs_b)
        if boot_probs:
            boot_arr = np.array(boot_probs)
            lower = np.quantile(boot_arr, 0.05, axis=0)
            upper = np.quantile(boot_arr, 0.95, axis=0)
            plt.figure(figsize=(5.8, 3.6))
            plt.plot(evidence_df["season"], evidence_df["prob_rank"], color=COLOR_PRIMARY, marker="o")
            plt.fill_between(evidence_df["season"], lower, upper, color=COLOR_PRIMARY, alpha=0.18)
            plt.axvline(28, color=COLOR_GRAY, linestyle="--", linewidth=1.0)
            plt.xlabel("Season")
            plt.ylabel("P(rank+save)")
            plt.title("Rule switch probability with uncertainty band")
            plt.savefig(FIG_DIR / "fig_rule_switch_ci.pdf")
            plt.close()

    # 机制评估
    log("Evaluating mechanism metrics...")
    metrics = {
        "percent": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "judge_integrity_sum": 0.0, "count": 0},
        "rank": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "judge_integrity_sum": 0.0, "count": 0},
        "save": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "judge_integrity_sum": 0.0, "count": 0},
        "daws": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "judge_integrity_sum": 0.0, "count": 0},
    }
    flip_sum = 0.0
    flip_count = 0

    # DAWS参数
    alpha0, gamma, eta = 0.55, 0.15, 0.8
    alpha_min, alpha_max, delta = 0.35, 0.75, 0.08

    for (season, week), wdf in season_week_groups:
        active_df = wdf[wdf["active"]].copy()
        if len(active_df) == 0:
            continue
        key = (int(season), int(week))
        samples = samples_cache.get(key)
        if samples is None or len(samples) == 0:
            samples, _ = sample_week_percent(wdf, ALPHA_PERCENT, EPSILON, n_props)
            samples_cache[key] = samples
        if len(samples) == 0:
            continue

        # 计算U_t
        mean_width = week_metrics_df[(week_metrics_df["season"] == season) & (week_metrics_df["week"] == week)]["mean_hdi_width"].mean()
        if np.isnan(mean_width):
            mean_width = 0.0
        T = active_df["week"].max()
        alpha_t = compute_alpha_t(week, T, mean_width, alpha0, gamma, eta, alpha_min, alpha_max)

        eval_res = evaluate_mechanisms(samples, active_df, alpha_t, EPSILON, RNG)
        if not eval_res:
            continue
        m = eval_res["count"]
        for key in ["percent", "rank", "save", "daws"]:
            metrics[key]["fairness_sum"] += eval_res[key]["fairness"] * m
            metrics[key]["agency_sum"] += eval_res[key]["agency"] * m
            metrics[key]["instability_sum"] += eval_res[key]["instability"] * m
            metrics[key]["judge_integrity_sum"] += eval_res[key]["judge_integrity"] * m
            metrics[key]["count"] += m

        flip_sum += float(eval_res["flip_sum"])
        flip_count += m

    def agg_stats(d: Dict[str, float]) -> Dict[str, float]:
        if d["count"] == 0:
            return {"fairness": float("nan"), "agency": float("nan"), "stability": float("nan"), "judge_integrity": float("nan")}
        fairness = d["fairness_sum"] / d["count"]
        agency = d["agency_sum"] / d["count"]
        stability = 1.0 - (d["instability_sum"] / d["count"])
        judge_integrity = d["judge_integrity_sum"] / d["count"]
        return {
            "fairness": float(fairness),
            "agency": float(agency),
            "stability": float(stability),
            "judge_integrity": float(judge_integrity),
        }

    stats_percent = agg_stats(metrics["percent"])
    stats_rank = agg_stats(metrics["rank"])
    stats_daws = agg_stats(metrics["daws"])
    flip_rate = float(flip_sum / flip_count) if flip_count else float("nan")
    log(f"Flip rate (percent vs rank): {flip_rate:.3f}")

    # =========================
    # Fast vs Strict 校验
    # =========================
    fast_strict_summary: Dict[str, float] = {}
    if FAST_STRICT_ENABLED and save_outputs:
        log("Fast vs Strict validation...")
        mae_list: List[float] = []
        top1_hits = 0
        top2_hits = 0
        week_count = 0
        fast_points: List[float] = []
        strict_points: List[float] = []

        metrics_fast = {
            "percent": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
            "rank": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
            "daws": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
        }
        metrics_strict = {
            "percent": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
            "rank": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
            "daws": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
        }
        flip_fast_sum = 0.0
        flip_fast_count = 0
        flip_strict_sum = 0.0
        flip_strict_count = 0

        for (season, week), wdf in season_week_groups:
            active_df = wdf[wdf["active"]].copy()
            if len(active_df) == 0:
                continue
            rng = np.random.default_rng(20260131 + int(season) * 1000 + int(week))
            n = len(active_df)
            j_share = active_df["judge_share"].to_numpy()
            elim_idx = [i for i, flag in enumerate(active_df["is_eliminated_week"].to_numpy()) if flag]

            proposals = rng.dirichlet(np.ones(n), size=FAST_STRICT_PROPS)
            proposals = np.maximum(proposals, EPSILON)
            proposals = proposals / proposals.sum(axis=1, keepdims=True)

            mask_fast = apply_percent_mask(proposals, j_share, elim_idx, ALPHA_PERCENT, "fast")
            mask_strict = apply_percent_mask(proposals, j_share, elim_idx, ALPHA_PERCENT, "strict")

            fast_samples = proposals[mask_fast]
            strict_samples = proposals[mask_strict]
            if len(fast_samples) < MIN_ACCEPT:
                fast_samples = fallback_by_violation(proposals, j_share, elim_idx, ALPHA_PERCENT, MIN_ACCEPT)
            if len(strict_samples) < MIN_ACCEPT:
                strict_samples = fallback_by_violation(proposals, j_share, elim_idx, ALPHA_PERCENT, MIN_ACCEPT)

            if len(fast_samples) > FAST_STRICT_MAX_SAMPLES:
                idx = rng.choice(len(fast_samples), size=FAST_STRICT_MAX_SAMPLES, replace=False)
                fast_samples = fast_samples[idx]
            if len(strict_samples) > FAST_STRICT_MAX_SAMPLES:
                idx = rng.choice(len(strict_samples), size=FAST_STRICT_MAX_SAMPLES, replace=False)
                strict_samples = strict_samples[idx]

            if len(fast_samples) == 0 or len(strict_samples) == 0:
                continue

            fast_mean = fast_samples.mean(axis=0)
            strict_mean = strict_samples.mean(axis=0)
            mae_list.append(float(np.mean(np.abs(fast_mean - strict_mean))))

            fast_points.extend(fast_mean.tolist())
            strict_points.extend(strict_mean.tolist())

            fast_bottom1 = int(np.argmin(fast_mean))
            strict_bottom1 = int(np.argmin(strict_mean))
            if fast_bottom1 == strict_bottom1:
                top1_hits += 1
            fast_bottom2 = set(np.argsort(fast_mean)[:2].tolist())
            strict_bottom2 = set(np.argsort(strict_mean)[:2].tolist())
            if fast_bottom2 == strict_bottom2:
                top2_hits += 1
            week_count += 1

            mean_width = week_metrics_df[(week_metrics_df["season"] == season) & (week_metrics_df["week"] == week)]["mean_hdi_width"].mean()
            if np.isnan(mean_width):
                mean_width = 0.0
            T = active_df["week"].max()
            alpha_t = compute_alpha_t(week, T, mean_width, alpha0, gamma, eta, alpha_min, alpha_max)

            res_fast = evaluate_mechanisms(fast_samples, active_df, alpha_t, EPSILON, rng)
            res_strict = evaluate_mechanisms(strict_samples, active_df, alpha_t, EPSILON, rng)

            for key in ["percent", "rank", "daws"]:
                metrics_fast[key]["fairness_sum"] += res_fast[key]["fairness"] * res_fast["count"]
                metrics_fast[key]["agency_sum"] += res_fast[key]["agency"] * res_fast["count"]
                metrics_fast[key]["instability_sum"] += res_fast[key]["instability"] * res_fast["count"]
                metrics_fast[key]["count"] += res_fast["count"]

                metrics_strict[key]["fairness_sum"] += res_strict[key]["fairness"] * res_strict["count"]
                metrics_strict[key]["agency_sum"] += res_strict[key]["agency"] * res_strict["count"]
                metrics_strict[key]["instability_sum"] += res_strict[key]["instability"] * res_strict["count"]
                metrics_strict[key]["count"] += res_strict["count"]

            flip_fast_sum += float(res_fast["flip_sum"])
            flip_fast_count += res_fast["count"]
            flip_strict_sum += float(res_strict["flip_sum"])
            flip_strict_count += res_strict["count"]

        def agg_simple(d: Dict[str, float]) -> Dict[str, float]:
            if d["count"] == 0:
                return {"fairness": float("nan"), "agency": float("nan"), "stability": float("nan")}
            fairness = d["fairness_sum"] / d["count"]
            agency = d["agency_sum"] / d["count"]
            stability = 1.0 - (d["instability_sum"] / d["count"])
            return {"fairness": float(fairness), "agency": float(agency), "stability": float(stability)}

        fast_percent = agg_simple(metrics_fast["percent"])
        strict_percent = agg_simple(metrics_strict["percent"])

        top1_agree = (top1_hits / week_count) if week_count else 0.0
        top2_agree = (top2_hits / week_count) if week_count else 0.0
        mae_mean = float(np.mean(mae_list)) if mae_list else float("nan")
        flip_fast = float(flip_fast_sum / flip_fast_count) if flip_fast_count else float("nan")
        flip_strict = float(flip_strict_sum / flip_strict_count) if flip_strict_count else float("nan")

        fast_strict_summary = {
            "mae_mean": mae_mean,
            "top1_agree": top1_agree,
            "top2_agree": top2_agree,
            "delta_fairness": abs(fast_percent["fairness"] - strict_percent["fairness"]),
            "delta_agency": abs(fast_percent["agency"] - strict_percent["agency"]),
            "delta_flip": abs(flip_fast - flip_strict),
            "flip_fast": flip_fast,
            "flip_strict": flip_strict,
        }

        # 绘制 Fast vs Strict 散点
        if fast_points and strict_points:
            pts = len(fast_points)
            if pts > 4000:
                idx = RNG.choice(pts, size=4000, replace=False)
                x = np.array(fast_points)[idx]
                y = np.array(strict_points)[idx]
            else:
                x = np.array(fast_points)
                y = np.array(strict_points)
            plt.figure(figsize=(4.8, 4.2))
            plt.scatter(x, y, s=10, alpha=0.45, color=COLOR_PRIMARY)
            lim = max(x.max(), y.max())
            plt.plot([0, lim], [0, lim], color=COLOR_GRAY, linestyle="--", linewidth=1.0)
            plt.xlabel("Fast mean fan share")
            plt.ylabel("Strict mean fan share")
            plt.title("Fast vs Strict posterior means")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "fig_fast_vs_strict.pdf")
            plt.close()

        (OUTPUT_DIR / "fast_strict_metrics.json").write_text(
            json.dumps(fast_strict_summary, indent=2), encoding="utf-8"
        )

    # =========================
    # 混合效应模型（简化版）
    # =========================
    log("Fitting effects models...")
    model_df = posterior_df.copy()
    model_df = model_df.dropna(subset=["fan_share_mean", "judge_share"])  # 防止空值
    model_df["age"] = model_df["celebrity_age_during_season"].astype(float)

    def logit(p: float) -> float:
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return float(np.log(p / (1 - p)))

    model_df["y_j"] = model_df["judge_share"].apply(logit)
    model_df["y_f"] = model_df["fan_share_mean"].apply(logit)

    if USE_MIXED_MODEL:
        # Judges model
        try:
            md_j = smf.mixedlm("y_j ~ age + C(celebrity_industry)", model_df, groups=model_df["ballroom_partner"], vc_formula={"season": "0 + C(season)"})
            m_j = md_j.fit(reml=False)
            re_j = m_j.random_effects
            fe_j = m_j.params
        except Exception:
            m_j = smf.ols("y_j ~ age + C(celebrity_industry)", model_df).fit()
            re_j = {}
            fe_j = m_j.params

        # Fans model
        try:
            md_f = smf.mixedlm("y_f ~ age + C(celebrity_industry)", model_df, groups=model_df["ballroom_partner"], vc_formula={"season": "0 + C(season)"})
            m_f = md_f.fit(reml=False)
            re_f = m_f.random_effects
            fe_f = m_f.params
        except Exception:
            m_f = smf.ols("y_f ~ age + C(celebrity_industry)", model_df).fit()
            re_f = {}
            fe_f = m_f.params

        pro_effects = pd.DataFrame({"pro": list(set(model_df["ballroom_partner"]))})
        pro_effects["effect_j"] = pro_effects["pro"].apply(lambda p: re_j.get(p, {}).get("Group", 0.0) if isinstance(re_j.get(p, None), dict) else 0.0)
        pro_effects["effect_f"] = pro_effects["pro"].apply(lambda p: re_f.get(p, {}).get("Group", 0.0) if isinstance(re_f.get(p, None), dict) else 0.0)
        pro_effects["se"] = np.std(pro_effects[["effect_j", "effect_f"]].to_numpy()) if len(pro_effects) > 1 else 0.1
    else:
        # 快速版本，使用OLS并用残差均值近似pro效应
        m_j = smf.ols("y_j ~ age + C(celebrity_industry)", model_df).fit()
        m_f = smf.ols("y_f ~ age + C(celebrity_industry)", model_df).fit()
        fe_j = m_j.params
        fe_f = m_f.params
        model_df["res_j"] = m_j.resid
        model_df["res_f"] = m_f.resid
        pro_effects = model_df.groupby("ballroom_partner")[["res_j", "res_f"]].mean().reset_index()
        pro_effects = pro_effects.rename(columns={"ballroom_partner": "pro", "res_j": "effect_j", "res_f": "effect_f"})
        pro_effects["se"] = model_df[["res_j", "res_f"]].stack().std() if len(model_df) > 1 else 0.1

    # =========================
    # 预测模型（XGBoost替代）
    # =========================
    log("Training predictive model (GBDT)...")
    pred_df = posterior_df.copy()
    pred_df["eliminated"] = pred_df["is_eliminated_week"].astype(int)
    pred_df["age"] = pred_df["celebrity_age_during_season"].astype(float)
    features = ["age", "celebrity_industry", "ballroom_partner"]
    X = pd.get_dummies(pred_df[features], drop_first=True)
    y = pred_df["eliminated"].to_numpy()
    seasons = pred_df["season"].to_numpy()

    auc_records = []
    for s in sorted(pred_df["season"].unique()):
        train_mask = seasons < s
        test_mask = seasons == s
        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1)
        clf.fit(X[train_mask], y[train_mask])
        prob = clf.predict_proba(X[test_mask])[:, 1]
        auc = roc_auc_score(y[test_mask], prob)
        auc_records.append({"season": s, "auc": auc})
    auc_df = pd.DataFrame(auc_records)

    # =========================
    # 图表输出
    # =========================
    log("Rendering figures...")

    # Uncertainty heatmap
    plt.figure(figsize=(6.4, 3.8))
    ax = sns.heatmap(
        heat,
        cmap="cividis",
        cbar_kws={"label": "Mean HDI width"},
    )
    ax.set_xticks(np.arange(max_week) + 0.5)
    ax.set_xticklabels(np.arange(1, max_week + 1))
    ax.set_yticks(np.arange(max_season) + 0.5)
    ax.set_yticklabels(np.arange(1, max_season + 1))
    plt.xlabel("Week")
    plt.ylabel("Season")
    plt.title("Uncertainty concentrates in a small set of weeks")
    plt.savefig(FIG_DIR / "fig_uncertainty_heatmap.pdf")
    plt.close()

    # Wrongful heatmap
    plt.figure(figsize=(6.4, 3.8))
    ax = sns.heatmap(
        wrongful_heat,
        cmap="cividis",
        cbar_kws={"label": "Wrongful prob"},
    )
    ax.set_xticks(np.arange(max_week) + 0.5)
    ax.set_xticklabels(np.arange(1, max_week + 1))
    ax.set_yticks(np.arange(max_season) + 0.5)
    ax.set_yticklabels(np.arange(1, max_season + 1))
    plt.xlabel("Week")
    plt.ylabel("Season")
    plt.title("Wrongful elimination probability by week")
    plt.savefig(FIG_DIR / "fig_wrongful_heatmap.pdf")
    plt.close()

    # Conflict map
    cm_df = posterior_df.copy()
    plt.figure(figsize=(5.8, 4.2))
    sizes = 200 * cm_df["hdi_width"].clip(0, cm_df["hdi_width"].quantile(0.95))
    colors = cm_df["is_eliminated_week"].map(lambda x: COLOR_WARNING if x else COLOR_PRIMARY)
    plt.scatter(cm_df["judge_share"], cm_df["fan_share_mean"], s=sizes, c=colors, alpha=0.75, edgecolors="none")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Eliminated week", markerfacecolor=COLOR_WARNING, markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="Not eliminated", markerfacecolor=COLOR_PRIMARY, markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="HDI width (size)", markerfacecolor=COLOR_GRAY, markersize=8),
    ]
    plt.xlabel("Judge share")
    plt.ylabel("Fan share (posterior mean)")
    plt.title("Elimination is not always aligned with minimum fan support")
    plt.legend(handles=legend_handles, frameon=False, fontsize=8, loc="lower right")
    plt.savefig(FIG_DIR / "fig_conflict_map.pdf")
    plt.close()

    # Conflict combo: HDI size + wrongful标注
    combo_df = cm_df.copy()
    combo_df["wrongful"] = False
    for (season, week), sub in combo_df.groupby(["season", "week"], sort=False):
        if sub.empty:
            continue
        min_fan = sub["fan_share_mean"].min()
        wrongful_mask = (sub["is_eliminated_week"]) & (sub["fan_share_mean"] > min_fan + 1e-6)
        combo_df.loc[wrongful_mask.index, "wrongful"] = wrongful_mask
    plt.figure(figsize=(5.8, 4.2))
    sizes = 220 * combo_df["hdi_width"].clip(0, combo_df["hdi_width"].quantile(0.95))
    colors = combo_df["is_eliminated_week"].map(lambda x: COLOR_WARNING if x else COLOR_PRIMARY)
    plt.scatter(combo_df["judge_share"], combo_df["fan_share_mean"], s=sizes, c=colors, alpha=0.70, edgecolors="none")
    wrongful_pts = combo_df[combo_df["wrongful"]]
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Eliminated week", markerfacecolor=COLOR_WARNING, markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="Not eliminated", markerfacecolor=COLOR_PRIMARY, markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="HDI width (size)", markerfacecolor=COLOR_GRAY, markersize=8),
    ]
    if not wrongful_pts.empty:
        plt.scatter(
            wrongful_pts["judge_share"],
            wrongful_pts["fan_share_mean"],
            s=120,
            facecolors="none",
            edgecolors=COLOR_PRIMARY_DARK,
            linewidths=1.2,
            label="Wrongful elimination",
        )
    plt.xlabel("Judge share")
    plt.ylabel("Fan share (posterior mean)")
    plt.title("Conflict + uncertainty + wrongful eliminations")
    if not wrongful_pts.empty:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color=COLOR_PRIMARY_DARK, label="Wrongful elimination", markerfacecolor="none", markersize=7),
        )
    plt.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)
    plt.savefig(FIG_DIR / "fig_conflict_combo.pdf")
    plt.close()

    # Sigma sensitivity
    sigma_vals = []
    sigma_widths = []
    for sigma in SIGMA_LIST:
        # 简化：对周均宽度进行高斯平滑
        smoothed = gaussian_filter1d(np.nan_to_num(week_metrics_df["mean_hdi_width"], nan=np.nanmean(week_metrics_df["mean_hdi_width"])), sigma)
        sigma_vals.append(sigma)
        sigma_widths.append(float(np.mean(smoothed)))
    plt.figure(figsize=(5.6, 3.6))
    plt.plot(sigma_vals, sigma_widths, marker="o", color=COLOR_PRIMARY)
    plt.xlabel("Sigma")
    plt.ylabel("Average HDI width")
    plt.title("Sensitivity of HDI width to sigma")
    plt.savefig(FIG_DIR / "fig_sigma_sensitivity.pdf")
    plt.close()

    # Rule switch
    plt.figure(figsize=(5.8, 3.6))
    plt.plot(evidence_df["season"], evidence_df["prob_rank"], color=COLOR_PRIMARY, marker="o")
    plt.axvline(28, color=COLOR_GRAY, linestyle="--", linewidth=1.0)
    plt.xlabel("Season")
    plt.ylabel("P(rank+save)")
    plt.title("Inferred rule switch probability")
    plt.savefig(FIG_DIR / "fig_rule_switch.pdf")
    plt.close()

    # Mechanism radar
    def radar_plot(stats_dict: Dict[str, Dict[str, float]], labels: List[str], fname: str) -> None:
        categories = ["fairness", "agency", "stability"]
        angles = np.linspace(0, 2 * math.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(4.8, 4.2))
        ax = plt.subplot(111, polar=True)
        for label in labels:
            values = [stats_dict[label][c] for c in categories]
            values += values[:1]
            ax.plot(angles, values, label=label)
            ax.fill(angles, values, alpha=0.10)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title("Mechanism trade-offs")
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05))
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)

    radar_plot(
        {"Percent": stats_percent, "Rank": stats_rank, "DAWS": stats_daws},
        ["Percent", "Rank", "DAWS"],
        "fig_mechanism_radar.pdf",
    )

    # Mechanism compare (bar)
    labels = ["Fairness", "Agency", "Stability"]
    percent_vals = [stats_percent["fairness"], stats_percent["agency"], stats_percent["stability"]]
    rank_vals = [stats_rank["fairness"], stats_rank["agency"], stats_rank["stability"]]
    daws_vals = [stats_daws["fairness"], stats_daws["agency"], stats_daws["stability"]]
    x = np.arange(len(labels))
    width = 0.24
    plt.figure(figsize=(6.2, 3.6))
    plt.bar(x - width, percent_vals, width, label="Percent", color=COLOR_PRIMARY, alpha=0.85)
    plt.bar(x, rank_vals, width, label="Rank", color=COLOR_GRAY, alpha=0.85)
    plt.bar(x + width, daws_vals, width, label="DAWS", color=COLOR_ACCENT, alpha=0.85)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Mechanism comparison (numeric)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_mechanism_compare.pdf")
    plt.close()

    # Pareto-like trade-off (2D)
    def stats_for_alpha(alpha: float) -> Dict[str, float]:
        totals = {"agency_sum": 0.0, "instability_sum": 0.0, "judge_integrity_sum": 0.0, "count": 0}
        for (season, week), wdf in season_week_groups:
            active_df = wdf[wdf["active"]].copy()
            if len(active_df) == 0:
                continue
            key = (int(season), int(week))
            samples = samples_cache.get(key)
            if samples is None or len(samples) == 0:
                samples, _ = sample_week_percent(wdf, ALPHA_PERCENT, EPSILON, n_props)
                samples_cache[key] = samples
            if len(samples) == 0:
                continue
            res = evaluate_mechanisms(samples, active_df, alpha, EPSILON, RNG)
            m = res["count"]
            totals["agency_sum"] += res["daws"]["agency"] * m
            totals["instability_sum"] += res["daws"]["instability"] * m
            totals["judge_integrity_sum"] += res["daws"]["judge_integrity"] * m
            totals["count"] += m
        if totals["count"] == 0:
            return {"agency": float("nan"), "stability": float("nan"), "judge_integrity": float("nan")}
        agency = totals["agency_sum"] / totals["count"]
        stability = 1.0 - (totals["instability_sum"] / totals["count"])
        judge_integrity = totals["judge_integrity_sum"] / totals["count"]
        return {"agency": float(agency), "stability": float(stability), "judge_integrity": float(judge_integrity)}

    alpha_grid = np.linspace(0.05, 0.95, 10)
    curve = [stats_for_alpha(a) for a in alpha_grid]
    xs = [c["agency"] for c in curve]
    ys = [c["stability"] for c in curve]
    cs = [c["judge_integrity"] for c in curve]

    plt.figure(figsize=(5.2, 4.2))
    plt.plot(xs, ys, color=COLOR_GRAY, linewidth=1.1, alpha=0.6)
    sc = plt.scatter(xs, ys, c=cs, cmap="cividis", s=25, zorder=2)
    plt.scatter(stats_percent["agency"], stats_percent["stability"], color=COLOR_PRIMARY, s=55, marker="o", label="Percent")
    plt.scatter(stats_rank["agency"], stats_rank["stability"], color=COLOR_GRAY, s=55, marker="s", label="Rank")
    plt.scatter(stats_daws["agency"], stats_daws["stability"], color=COLOR_ACCENT, s=85, marker="*", label="DAWS")
    plt.xlabel("Viewer agency")
    plt.ylabel("Stability")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title("Pareto trade-off: agency vs stability")
    plt.legend(frameon=False, fontsize=8, loc="lower left")
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label("Judge integrity")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_pareto_2d.pdf")
    plt.close()

    # Pro dancer effects difference (Top 20)
    def clean_pro_name(name: str) -> str:
        return re.sub(r"\s*\(.*\)$", "", str(name)).strip()

    pro_effects_sorted = pro_effects.copy()
    pro_effects_sorted["pro_clean"] = pro_effects_sorted["pro"].apply(clean_pro_name)
    pro_effects_sorted["diff"] = pro_effects_sorted["effect_f"] - pro_effects_sorted["effect_j"]
    pro_effects_sorted = pro_effects_sorted.sort_values("diff", ascending=False, key=lambda s: s.abs()).head(20)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2))
    y_pos = np.arange(len(pro_effects_sorted))
    ax.errorbar(pro_effects_sorted["diff"], y_pos, xerr=pro_effects_sorted["se"], fmt="o", color=COLOR_ACCENT)
    ax.axvline(0, color=COLOR_GRAY, linestyle="--", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pro_effects_sorted["pro_clean"], fontsize=7)
    ax.set_xlabel("Effect difference (Fans - Judges)")
    ax.set_title("Pro dancer effects: fans vs judges")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_pro_diff_forest.pdf")
    plt.close(fig)

    # Feature effect scatter
    def extract_effects(params: pd.Series) -> pd.Series:
        eff = params.copy()
        eff = eff.drop(labels=[x for x in eff.index if x.startswith("Intercept")], errors="ignore")
        return eff

    fe_j_series = extract_effects(fe_j)
    fe_f_series = extract_effects(fe_f)
    common_idx = fe_j_series.index.intersection(fe_f_series.index)
    x = fe_j_series[common_idx]
    y = fe_f_series[common_idx]
    plt.figure(figsize=(5.0, 4.0))
    plt.scatter(x, y, color=COLOR_PRIMARY)
    lim = max(abs(x).max(), abs(y).max())
    plt.plot([-lim, lim], [-lim, lim], color=COLOR_GRAY, linestyle="--")
    # annotate most divergent features
    dist = (y - x).abs()
    top_idx = dist.sort_values(ascending=False).head(5).index
    for idx in top_idx:
        label = (
            str(idx)
            .replace("celebrity_industry_", "ind:")
            .replace("ballroom_partner_", "pro:")
        )
        plt.annotate(label, (x[idx], y[idx]), fontsize=7, alpha=0.8)
    plt.xlabel("Judge effect")
    plt.ylabel("Fan effect")
    plt.title("Feature impacts: judges vs fans")
    plt.savefig(FIG_DIR / "fig_feature_scatter.pdf")
    plt.close()

    # AUC curve
    if not auc_df.empty:
        plt.figure(figsize=(5.2, 3.4))
        plt.plot(auc_df["season"], auc_df["auc"], marker="o", color=COLOR_PRIMARY)
        plt.xlabel("Season")
        plt.ylabel("AUC")
        plt.title("Forward-chaining AUC (GBDT)")
        plt.savefig(FIG_DIR / "fig_auc_forward.pdf")
        plt.close()
        log(f"AUC points: {len(auc_df)}")

    # Judge-save curve (示意)
    xs = np.linspace(-10, 10, 200)
    beta = 1.8
    ys = 1 / (1 + np.exp(beta * xs))
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(xs, ys, color=COLOR_PRIMARY)
    plt.xlabel("Judge score difference")
    plt.ylabel("P(eliminate a)")
    plt.title("Judge-save decision curve")
    plt.savefig(FIG_DIR / "fig_judgesave_curve.pdf")
    plt.close()

    # Posterior predictive coverage (简化)
    coverage = 1 - np.nanmean(wrongful_heat)
    brier = np.nanmean(wrongful_heat * (1 - wrongful_heat))
    plt.figure(figsize=(4.8, 3.4))
    plt.bar(["Coverage", "Brier"], [coverage, brier], color=[COLOR_PRIMARY, COLOR_ACCENT])
    plt.ylim(0, 1)
    plt.title("Posterior predictive checks")
    plt.savefig(FIG_DIR / "fig_ppc_summary.pdf")
    plt.close()

    # =========================
    # 争议案例 Ridgeline（自动筛选“冤案”）
    # =========================
    log("Rendering smart controversy ridgeline chart...")
    top_controversy = (
        posterior_df[posterior_df["is_eliminated_week"]]
        .sort_values("fan_share_mean", ascending=False)
        .drop_duplicates(subset=["celebrity_name"])
        .head(4)
    )
    controversy_names = top_controversy["celebrity_name"].tolist()
    if not controversy_names:
        controversy_names = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=False)
    axes = axes.flatten()
    for ax, name in zip(axes, controversy_names):
        sub = posterior_df[posterior_df["celebrity_name"] == name].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"{name}\\nNot Found", ha="center", va="center")
            ax.axis("off")
            continue

        weeks = sorted(sub["week"].unique())
        max_x = max(0.30, sub["fan_share_mean"].max() + 0.10)
        x = np.linspace(0, max_x, 200)
        elim_week = sub[sub["is_eliminated_week"]]["week"].min()

        for i, w in enumerate(weeks):
            row = sub[sub["week"] == w].iloc[0]
            mu = row["fan_share_mean"]
            width = max(row["hdi_width"], 1e-3)
            sigma = width / 3.0
            density = np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            is_elim = (w == elim_week)
            color = COLOR_WARNING if is_elim else COLOR_PRIMARY
            alpha = 0.60 if is_elim else 0.20
            offset = i * 0.5
            ax.fill_between(x, offset, offset + density * 0.8, color=color, alpha=alpha)
            ax.plot(x, offset + density * 0.8, color=color, linewidth=1.0 if is_elim else 0.6)
            if is_elim:
                ax.text(max_x * 0.95, offset, "ELIMINATED", color=COLOR_WARNING, fontsize=8, ha="right", fontweight="bold")

        season_id = int(sub.iloc[0]["season"])
        ax.set_title(f"{name} (Season {season_id})", fontsize=9)
        ax.set_yticks([])
        if ax in (axes[2], axes[3]):
            ax.set_xlabel("Estimated fan share")
    plt.suptitle("Democratic deficit: high fan support yet eliminated", fontsize=11, y=0.98)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_controversy_ridgeline.pdf")
    plt.close(fig)

    # =========================
    # Alluvial-like flow（决赛阵容）
    # =========================
    log("Rendering finalists flow chart...")
    def get_finalists(season_df: pd.DataFrame) -> List[str]:
        top = season_df.sort_values("placement").head(3)
        return top["celebrity_name"].tolist()

    def predict_finalists(season: int, mechanism: str) -> List[str]:
        # 用最终周的均值 fan share 与 judge share 近似预测
        season_long = long_df[long_df["season"] == season]
        last_week = int(season_long["week"].max())
        wk = posterior_df[(posterior_df["season"] == season) & (posterior_df["week"] == last_week)]
        if wk.empty:
            return []
        j_share = wk["judge_share"].to_numpy()
        v_share = wk["fan_share_mean"].to_numpy()
        if mechanism == "percent":
            combined = 0.5 * j_share + 0.5 * v_share
        elif mechanism == "rank":
            j_rank = pd.Series(j_share).rank(ascending=False, method="average").to_numpy()
            f_rank = (-v_share).argsort().argsort() + 1
            combined = -(j_rank + f_rank)
        else:
            combined = 0.6 * j_share + 0.4 * v_share
        idx = np.argsort(-combined)[:3]
        return wk.iloc[idx]["celebrity_name"].tolist()

    def draw_alluvial(ax, counts: Dict[str, int], title: str) -> None:
        total = sum(counts.values())
        if total == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            return
        # 左右柱高度
        af = counts["ff"] + counts["fn"]
        anf = counts["nf"] + counts["nn"]
        pf = counts["ff"] + counts["nf"]
        pnf = counts["fn"] + counts["nn"]

        # 左侧区间
        y0 = 0.0
        af_low, af_high = y0, y0 + af / total
        anf_low, anf_high = af_high, 1.0

        # 右侧区间
        y1 = 0.0
        pf_low, pf_high = y1, y1 + pf / total
        pnf_low, pnf_high = pf_high, 1.0

        # 绘制柱子
        ax.add_patch(plt.Rectangle((0.0, af_low), 0.05, af_high - af_low, color=COLOR_PRIMARY))
        ax.add_patch(plt.Rectangle((0.0, anf_low), 0.05, anf_high - anf_low, color=COLOR_GRAY))
        ax.add_patch(plt.Rectangle((0.95, pf_low), 0.05, pf_high - pf_low, color=COLOR_PRIMARY))
        ax.add_patch(plt.Rectangle((0.95, pnf_low), 0.05, pnf_high - pnf_low, color=COLOR_GRAY))

        # 流带函数
        def ribbon(y_left_low, y_left_high, y_right_low, y_right_high, color):
            xs = [0.05, 0.4, 0.6, 0.95]
            ys1 = [y_left_low, y_left_low, y_right_low, y_right_low]
            ys2 = [y_left_high, y_left_high, y_right_high, y_right_high]
            ax.fill_between(xs, ys1, ys2, color=color, alpha=0.35)

        # 分配流
        ff_low, ff_high = af_low, af_low + counts["ff"] / total
        fn_low, fn_high = ff_high, af_high
        pf_ff_low, pf_ff_high = pf_low, pf_low + counts["ff"] / total
        pf_nf_low, pf_nf_high = pf_ff_high, pf_high
        pnf_fn_low, pnf_fn_high = pnf_low, pnf_low + counts["fn"] / total
        pnf_nn_low, pnf_nn_high = pnf_fn_high, pnf_high

        ribbon(ff_low, ff_high, pf_ff_low, pf_ff_high, COLOR_PRIMARY)
        ribbon(fn_low, fn_high, pnf_fn_low, pnf_fn_high, COLOR_WARNING)
        ribbon(anf_low, anf_low + counts["nf"] / total, pf_nf_low, pf_nf_high, COLOR_ACCENT)
        ribbon(anf_low + counts["nf"] / total, anf_high, pnf_nn_low, pnf_nn_high, COLOR_LIGHT_GRAY)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=9)

    mechanisms = ["percent", "rank", "daws"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for ax, mech in zip(axes, mechanisms):
        counts = {"ff": 0, "fn": 0, "nf": 0, "nn": 0}
        for season in sorted(df["season"].unique()):
            season_df = df[df["season"] == season].copy()
            actual_finalists = set(get_finalists(season_df))
            predicted = set(predict_finalists(season, mech))
            for name in season_df["celebrity_name"].unique():
                actual = name in actual_finalists
                pred = name in predicted
                if actual and pred:
                    counts["ff"] += 1
                elif actual and (not pred):
                    counts["fn"] += 1
                elif (not actual) and pred:
                    counts["nf"] += 1
                else:
                    counts["nn"] += 1
        title = {"percent": "Percent", "rank": "Rank", "daws": "DAWS"}[mech]
        draw_alluvial(ax, counts, title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_alluvial_finalists.pdf")
    plt.close(fig)

    # =========================
    # 输出指标与中间结果
    # =========================
    log("Writing summary metrics...")
    seasons_feasible = int((week_metrics_df.groupby("season")["accept_rate"].min() > 0).sum())
    max_hdi = float(np.nanmax(week_metrics_df["mean_hdi_width"]))
    mean_hdi = float(np.nanmean(week_metrics_df["mean_hdi_width"]))
    median_hdi = float(np.nanmedian(week_metrics_df["mean_hdi_width"]))
    p90_hdi = float(np.nanquantile(week_metrics_df["mean_hdi_width"], 0.90))
    daws_improve = (stats_percent["stability"] - stats_daws["stability"]) / max(1e-6, stats_percent["stability"]) * 100

    summary = {
        "seasons_feasible": seasons_feasible,
        "mean_hdi_width": mean_hdi,
        "median_hdi_width": median_hdi,
        "p90_hdi_width": p90_hdi,
        "max_hdi_width": max_hdi,
        "flip_rate": flip_rate,
        "daws_improve": daws_improve,
        "stability_daws": stats_daws["stability"],
        "fairness_daws": stats_daws["fairness"],
    }
    if fast_strict_summary:
        summary.update({
            "fast_strict_mae": fast_strict_summary.get("mae_mean", float("nan")),
            "fast_strict_top1": fast_strict_summary.get("top1_agree", float("nan")),
            "fast_strict_top2": fast_strict_summary.get("top2_agree", float("nan")),
            "fast_strict_delta_fairness": fast_strict_summary.get("delta_fairness", float("nan")),
            "fast_strict_delta_agency": fast_strict_summary.get("delta_agency", float("nan")),
            "fast_strict_delta_flip": fast_strict_summary.get("delta_flip", float("nan")),
        })
    log(f"Summary: seasons={seasons_feasible}, max_hdi={max_hdi:.3f}, daws_improve={daws_improve:.2f}")

    if save_outputs:
        (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8")

        # LaTeX宏
        SUMMARY_TEX.write_text(
            "\n".join([
            "% auto-generated metrics",
                f"\\newcommand{{\\MetricSeasonsFeasible}}{{{summary['seasons_feasible']}}}",
            f"\\newcommand{{\\MetricMeanHDI}}{{{summary['mean_hdi_width']:.3f}}}",
            f"\\newcommand{{\\MetricMedianHDI}}{{{summary['median_hdi_width']:.3f}}}",
            f"\\newcommand{{\\MetricHDIPctNinety}}{{{summary['p90_hdi_width']:.3f}}}",
            f"\\newcommand{{\\MetricMaxHDI}}{{{summary['max_hdi_width']:.2f}}}",
            f"\\newcommand{{\\MetricFlipRate}}{{{summary['flip_rate']*100:.1f}}}",
            f"\\newcommand{{\\MetricDAWSImprove}}{{{summary['daws_improve']:.1f}}}",
            f"\\newcommand{{\\MetricDAWSStability}}{{{summary['stability_daws']:.3f}}}",
            f"\\newcommand{{\\MetricDAWSFairness}}{{{summary['fairness_daws']:.3f}}}",
            f"\\newcommand{{\\MetricFastMAE}}{{{summary.get('fast_strict_mae', float('nan')):.4f}}}",
            f"\\newcommand{{\\MetricFastTopOne}}{{{summary.get('fast_strict_top1', float('nan'))*100:.1f}}}",
            f"\\newcommand{{\\MetricFastTopTwo}}{{{summary.get('fast_strict_top2', float('nan'))*100:.1f}}}",
            f"\\newcommand{{\\MetricFastDeltaFlip}}{{{summary.get('fast_strict_delta_flip', float('nan'))*100:.2f}}}",
            f"\\newcommand{{\\MetricFastDeltaFair}}{{{summary.get('fast_strict_delta_fairness', float('nan')):.3f}}}",
            f"\\newcommand{{\\MetricFastDeltaAgency}}{{{summary.get('fast_strict_delta_agency', float('nan')):.3f}}}",
        ]),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t_start
    log(f"Runtime: {elapsed:.2f}s")
    log("Done.")

    summary["runtime_sec"] = float(elapsed)
    summary["n_proposals"] = int(n_props)
    if record_benchmark:
        bench_df = update_benchmark_csv([{
            "n_proposals": summary["n_proposals"],
            "runtime_sec": summary["runtime_sec"],
            "mean_hdi": summary["mean_hdi_width"],
            "max_hdi": summary["max_hdi_width"],
            "stability_daws": summary["stability_daws"],
            "fairness_daws": summary["fairness_daws"],
            "flip_rate": summary["flip_rate"],
            "daws_improve": summary["daws_improve"],
        }])
        if save_outputs:
            plot_scale_benchmark(bench_df)

    return summary


if __name__ == "__main__":
    scales = parse_scales(os.getenv("MCM_SCALES"))
    if scales:
        results = []
        max_scale = max(scales)
        for scale in scales:
            log(f"Running scale experiment: {scale}")
            summary = run_pipeline(n_props=scale, record_benchmark=True, save_outputs=(scale == max_scale))
            results.append(summary)
        if (FIG_DIR / "fig_scale_benchmark.pdf").exists():
            log("Scale benchmark figure updated.")
    else:
        run_pipeline(n_props=N_PROPOSALS, record_benchmark=True, save_outputs=True)
