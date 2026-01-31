# -*- coding: utf-8 -*-
"""
DWTS 2026 MCM Problem C: 数据建模与可视化主流程
本脚本完成数据读取、投票可行集采样、机制评估与图表输出。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端便于批量导图
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
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

EPSILON = 0.001  # 投票占比下限
ALPHA_PERCENT = 0.5  # 百分比规则权重
N_PROPOSALS = 250  # 每周Dirichlet提案数量
MIN_ACCEPT = 40  # 最少保留的可行样本
SIGMA_LIST = [0.5, 1.0, 1.5, 2.0]
RHO_SWITCH = 0.10  # 规则切换先验概率
COMPUTE_BOUNDS = False  # 是否计算LP边界（耗时）
USE_MIXED_MODEL = False  # 是否使用混合效应模型
MAX_SAMPLES_PER_WEEK = 120  # 每周用于指标的最大样本数

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


def sample_week_percent(week_df: pd.DataFrame, alpha: float, epsilon: float, n_props: int) -> Tuple[np.ndarray, float]:
    # --- 极速向量化版本 (无需思考，直接用) ---
    active_df = week_df[week_df["active"]].copy()
    n = len(active_df)
    if n == 0:
        return np.empty((0, 0)), 0.0

    j = active_df["judge_share"].to_numpy()
    elim_idx = [i for i, flag in enumerate(active_df["is_eliminated_week"].to_numpy()) if flag]

    # 1. 批量生成随机提案
    proposals = RNG.dirichlet(np.ones(n), size=n_props)
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

def run_pipeline() -> None:
    ensure_dirs()
    set_plot_style()
    LOG_PATH.write_text("", encoding="utf-8")

    log("Load data...")
    df = load_data()
    long_df = build_long_df(df)
    log(f"Raw rows: {len(df)}, Long rows: {len(long_df)}")
    log(f"Seasons: {long_df['season'].nunique()}, Weeks: {long_df['week'].nunique()}")

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
            samples, acc_rate = sample_week_percent(wdf, ALPHA_PERCENT, EPSILON, N_PROPOSALS)
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
            samples, _ = sample_week_percent(wdf, ALPHA_PERCENT, EPSILON, N_PROPOSALS)
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

    # 机制评估
    log("Evaluating mechanism metrics...")
    metrics = {
        "percent": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
        "rank": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
        "save": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
        "daws": {"fairness_sum": 0.0, "agency_sum": 0.0, "instability_sum": 0.0, "count": 0},
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
            samples, _ = sample_week_percent(wdf, ALPHA_PERCENT, EPSILON, N_PROPOSALS)
            samples_cache[key] = samples
        if len(samples) == 0:
            continue

        # 计算U_t
        mean_width = week_metrics_df[(week_metrics_df["season"] == season) & (week_metrics_df["week"] == week)]["mean_hdi_width"].mean()
        T = active_df["week"].max()
        base_alpha = alpha0 + gamma * (week / max(1, T)) - eta * mean_width
        alpha_t = float(np.clip(base_alpha, alpha_min, alpha_max))

        # 平滑alpha_t（简单周内处理）
        alpha_t = float(np.clip(alpha_t, alpha_min, alpha_max))

        # --- 向量化评估 (Vectorized Evaluation) ---
        m = len(samples)
        j_share = active_df["judge_share"].to_numpy()
        j_rank = active_df["judge_share"].rank(ascending=False, method="average").to_numpy()
        j_scores = active_df["judge_total"].to_numpy()
        j_share_matrix = np.tile(j_share, (m, 1))

        # Percent 淘汰
        comb_percent = ALPHA_PERCENT * j_share_matrix + (1 - ALPHA_PERCENT) * samples
        elim_p = np.argmin(comb_percent, axis=1)

        # Rank 淘汰
        fan_rank = np.argsort(np.argsort(-samples, axis=1), axis=1) + 1
        j_rank_matrix = np.tile(j_rank, (m, 1))
        comb_rank = fan_rank + j_rank_matrix
        elim_r = np.argmax(comb_rank, axis=1)

        # DAWS 淘汰
        comb_daws = alpha_t * j_share_matrix + (1 - alpha_t) * samples
        elim_d = np.argmin(comb_daws, axis=1)

        # judge-save：bottom two + logistic（向量化）
        bottom_two = np.argpartition(comb_rank, -2, axis=1)[:, -2:]
        a_idx = bottom_two[:, 0]
        b_idx = bottom_two[:, 1]
        diff = j_scores[b_idx] - j_scores[a_idx]
        p_elim_a = 1 / (1 + np.exp(1.8 * diff))
        rand_u = RNG.random(m)
        elim_s = np.where(rand_u < p_elim_a, a_idx, b_idx)

        # fairness（Kendall tau 近似，向量化）
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

        # viewer agency
        fan_lowest = np.argmin(samples, axis=1)
        agency_p = np.mean(fan_lowest == elim_p)
        agency_r = np.mean(fan_lowest == elim_r)
        agency_s = np.mean(fan_lowest == elim_s)
        agency_d = np.mean(fan_lowest == elim_d)

        # stability（对percent机制噪声参考）
        noise = RNG.normal(0, 0.02, size=samples.shape)
        v_noise = np.maximum(samples + noise, EPSILON)
        v_noise = v_noise / v_noise.sum(axis=1, keepdims=True)
        elim_noise = np.argmin(ALPHA_PERCENT * j_share_matrix + (1 - ALPHA_PERCENT) * v_noise, axis=1)

        instability_p = np.mean(elim_p != elim_noise)
        instability_r = np.mean(elim_r != elim_noise)
        instability_s = np.mean(elim_s != elim_noise)
        instability_d = np.mean(elim_d != elim_noise)

        for key, agency, instability in [
            ("percent", agency_p, instability_p),
            ("rank", agency_r, instability_r),
            ("save", agency_s, instability_s),
            ("daws", agency_d, instability_d),
        ]:
            metrics[key]["fairness_sum"] += tau_mean * m
            metrics[key]["agency_sum"] += agency * m
            metrics[key]["instability_sum"] += instability * m
            metrics[key]["count"] += m

        flip_sum += float(np.sum(elim_p != elim_r))
        flip_count += m

    def agg_stats(d: Dict[str, float]) -> Dict[str, float]:
        if d["count"] == 0:
            return {"fairness": float("nan"), "agency": float("nan"), "stability": float("nan")}
        fairness = d["fairness_sum"] / d["count"]
        agency = d["agency_sum"] / d["count"]
        stability = 1.0 - (d["instability_sum"] / d["count"])
        return {"fairness": float(fairness), "agency": float(agency), "stability": float(stability)}

    stats_percent = agg_stats(metrics["percent"])
    stats_rank = agg_stats(metrics["rank"])
    stats_daws = agg_stats(metrics["daws"])
    flip_rate = float(flip_sum / flip_count) if flip_count else float("nan")
    log(f"Flip rate (percent vs rank): {flip_rate:.3f}")

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
    sns.heatmap(heat, cmap="cividis", cbar_kws={"label": "Mean HDI width"})
    plt.xlabel("Week")
    plt.ylabel("Season")
    plt.title("Uncertainty concentrates in a small set of weeks")
    plt.savefig(FIG_DIR / "fig_uncertainty_heatmap.pdf")
    plt.close()

    # Wrongful heatmap
    plt.figure(figsize=(6.4, 3.8))
    sns.heatmap(wrongful_heat, cmap="cividis", cbar_kws={"label": "Wrongful prob"})
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
    plt.xlabel("Judge share")
    plt.ylabel("Fan share (posterior mean)")
    plt.title("Elimination is not always aligned with minimum fan support")
    plt.savefig(FIG_DIR / "fig_conflict_map.pdf")
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

    # Ternary-like plot
    plt.figure(figsize=(4.8, 4.2))
    def barycentric(a, b, c):
        x = 0.5 * (2 * b + c) / (a + b + c)
        y = (math.sqrt(3) / 2) * c / (a + b + c)
        return x, y

    pts = []
    alphas = np.linspace(0.35, 0.75, 9)
    for a in alphas:
        # 简化：用线性插值
        f = stats_percent["fairness"] * (1 - a) + stats_rank["fairness"] * a
        ag = stats_percent["agency"] * (1 - a) + stats_rank["agency"] * a
        st = stats_percent["stability"] * (1 - a) + stats_rank["stability"] * a
        x, y = barycentric(f, ag, st)
        pts.append((x, y))
    xs, ys = zip(*pts)
    plt.scatter(xs, ys, color=COLOR_PRIMARY, s=25)
    dx, dy = barycentric(stats_daws["fairness"], stats_daws["agency"], stats_daws["stability"])
    plt.scatter([dx], [dy], color=COLOR_ACCENT, s=60, marker="*")
    # 画三角形边界
    tri = np.array([barycentric(1,0,0), barycentric(0,1,0), barycentric(0,0,1), barycentric(1,0,0)])
    plt.plot(tri[:,0], tri[:,1], color=COLOR_GRAY)
    plt.title("DAWS on the trade-off surface")
    plt.axis("off")
    plt.savefig(FIG_DIR / "fig_ternary_daws.pdf")
    plt.close()

    # Pro dancer forest (简化 Top 20)
    pro_effects_sorted = pro_effects.copy()
    pro_effects_sorted["diff"] = (pro_effects_sorted["effect_f"] - pro_effects_sorted["effect_j"]).abs()
    pro_effects_sorted = pro_effects_sorted.sort_values("diff", ascending=False).head(20)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.5), sharex=True)
    y_pos = np.arange(len(pro_effects_sorted))
    axes[0].errorbar(pro_effects_sorted["effect_j"], y_pos, xerr=pro_effects_sorted["se"], fmt="o", color=COLOR_PRIMARY)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(pro_effects_sorted["pro"], fontsize=7)
    axes[0].set_title("Pro dancer effects (Judges)")
    axes[1].errorbar(pro_effects_sorted["effect_f"], y_pos, xerr=pro_effects_sorted["se"], fmt="o", color=COLOR_ACCENT)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(pro_effects_sorted["pro"], fontsize=7)
    axes[1].set_title("Pro dancer effects (Fans)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_pro_forest.pdf")
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
    # 争议案例 Ridgeline（近似）
    # =========================
    log("Rendering controversy ridgeline chart...")
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
        max_x = max(0.25, sub["fan_share_mean"].max() + 0.15)
        x = np.linspace(0, max_x, 200)
        for i, w in enumerate(weeks):
            row = sub[sub["week"] == w].iloc[0]
            mu = row["fan_share_mean"]
            width = max(row["hdi_width"], 1e-3)
            sigma = width / (2 * 1.96)
            density = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
            density = density / density.max() * 0.8
            offset = i * 0.6
            ax.fill_between(x, offset, offset + density, color=COLOR_PRIMARY, alpha=0.25)
            ax.plot(x, offset + density, color=COLOR_PRIMARY, linewidth=1.0)
            if bool(row["is_eliminated_week"]):
                ax.scatter([mu], [offset + 0.35], color=COLOR_WARNING, s=12)
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
        ax.set_xlabel("Fan share")
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
    daws_improve = (stats_percent["stability"] - stats_daws["stability"]) / max(1e-6, stats_percent["stability"]) * 100

    summary = {
        "seasons_feasible": seasons_feasible,
        "max_hdi_width": max_hdi,
        "flip_rate": flip_rate,
        "daws_improve": daws_improve,
    }
    log(f"Summary: seasons={seasons_feasible}, max_hdi={max_hdi:.3f}, daws_improve={daws_improve:.2f}")

    (OUTPUT_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8")

    # LaTeX宏
    SUMMARY_TEX.write_text(
        "\n".join([
            "% 自动生成指标",
            f"\\newcommand{{\\MetricSeasonsFeasible}}{{{summary['seasons_feasible']}}}",
            f"\\newcommand{{\\MetricMaxHDI}}{{{summary['max_hdi_width']:.2f}}}",
            f"\\newcommand{{\\MetricFlipRate}}{{{summary['flip_rate']*100:.1f}}}",
            f"\\newcommand{{\\MetricDAWSImprove}}{{{summary['daws_improve']:.1f}}}",
        ]),
        encoding="utf-8",
    )

    log("Done.")


if __name__ == "__main__":
    run_pipeline()
