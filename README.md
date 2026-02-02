# MCM/ICM 2026 — Problem C (DWTS)

[![CI](https://github.com/YOUR_USERNAME/ProblemC-2/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ProblemC-2/actions/workflows/ci.yml)

本仓库包含针对 COMAP 2026 Problem C（Dancing with the Stars）的数据处理、建模、可视化与论文产出流水线。

## 快速概览
- 数据：`data/2026_MCM_Problem_C_Data.csv`（项目自给自足，避免跨目录依赖）
- 代码：`src/pipeline.py`（主流水线，自动解析项目根目录）
- 论文：`paper/main.pdf`（英文），`paper/main_zh.pdf`（中文）
- 文档：`docs/`（已按编号整理为 01_–07_ 开头；历史材料见 `docs/archive/`）
- 输出：`outputs/`（CSV/JSON/日志/基准等）

## 项目结构（简要）
```
.
├─ .github/workflows/        # CI 配置（GitHub Actions）
├─ data/                     # 输入数据（2026_MCM_Problem_C_Data.csv）
├─ docs/                     # 项目说明与开发文档（编号归档）
├─ outputs/                  # 运行输出（summary_metrics 等）
├─ paper/                    # 论文源文件与编译输出（main.pdf / main_zh.pdf）
│  └─ figures/               # 自动生成图表
├─ src/                      # 源代码（pipeline.py）
├─ tests/                    # 测试文件（pytest）
└─ README.md
```

## 持续集成（CI）

项目配置了 **GitHub Actions** 进行自动化测试，每次推送到 `main` 或提 PR 时自动触发：

| 检查项 | 工具 | 说明 |
|--------|------|------|
| 代码规范 | **Ruff** | 快速 Linter，检测语法错误、未定义变量等 |
| 单元测试 | **pytest** | 运行 `tests/` 下所有测试用例 |

**本地运行测试**（推送前建议先跑一遍）：
```bash
pip install pytest ruff
ruff check .              # 代码检查
python -m pytest -v       # 运行测试
```

## 环境与运行
- 推荐：Miniforge + Conda 环境 `mcm2026`（已在 `docs/07_开发文档.md` 说明依赖）

从项目根目录运行（pipeline 已按脚本位置解析路径）：
```bat
conda activate mcm2026
python -u src\pipeline.py
```

可用环境变量（选项）：
- `MCM_N_PROPOSALS`：采样规模（默认 250）
- `MCM_MULTIPROC`：多进程（1 启用）
- `MCM_WORKERS`：并行进程数

## 生成论文 PDF
在 `paper/` 目录下编译（英文使用 `pdflatex`，中文使用 `xelatex`）：
```bat
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
```

编译结果：`paper/main.pdf`，`paper/main_zh.pdf`。

## 文档与规范
- 图表风格：`docs/04_图表规范.md`
- 算法与论文框架：`docs/03_算法框架.md`
- 开发规范（Git 工作流、注释要求）：`docs/06_开发规范.md`
- 任务追踪与产物：`docs/05_任务追踪清单.md`

## 变更记录（要点）
- 已将数据移动到 `data/` 并修复 `src/pipeline.py` 中的数据路径解析；路径现在基于脚本位置，运行更稳健。  
- 文档已重命名并归档旧稿到 `docs/archive/`。  
- 完成 Block 0–3 的图注/附录/一致性工作；中英文论文已编译通过并保存到 `paper/`。

如需我替你：运行完整 pipeline、生成特定图表或把输出推到远程，请告诉我下一步操作。

## 项目完成进度

> 更新日期：2026-02-01

### 核心任务（Block 0–3）

| Block | 内容 | 状态 |
|-------|------|------|
| **Block 0** | 方法论闭环（稳定性定义、三指标重构、DAWS 落地化、β 口径） | ✅ 完成 |
| **Block 1** | 主证据图（反事实风险时间线、DAWS 触发图、Pareto 2D） | ✅ 完成 |
| **Block 2** | 图注/可读性/引用一致性（热力图轴、冲突散点 legend、中文图号修复） | ✅ 完成 |
| **Block 3** | 附录增强（Sigma 敏感性、AUC 校准、图像截取目录） | ✅ 完成 |

### 可选增强包

| 包 | 内容 | 状态 |
|----|------|------|
| **包 S** | 机制指标分布图 + 中文图注权衡说明 | ✅ 完成 |
| **包 M** | DAWS 小范围调参 + 参数表 | 🔲 可选 |

### 回归检查

| 检查项 | 状态 |
|--------|------|
| Block 0 回归检查（指标/DAWS/β 口径一致） | ✅ 通过 |
| Block 1 回归检查（图与数据/叙事一致） | ✅ 通过 |
| Block 2 回归检查（中英文图号/引用一致） | ✅ 通过 |
| Block 3 回归检查（附录不抢主线） | ✅ 通过 |
| 依赖/输出检查（PDF 编译、summary_metrics 一致） | ✅ 通过 |

### 产出物

| 类型 | 文件 | 说明 |
|------|------|------|
| 英文论文 | `paper/main.pdf` | 28 页，pdfLaTeX 编译 |
| 中文论文 | `paper/main_zh.pdf` | 29 页，XeLaTeX 编译 |
| 图表 | `paper/figures/*.pdf` | 23+ 张自动生成图表 |
| 指标汇总 | `outputs/summary_metrics.json` | seasons_feasible=34, max_hdi=0.947 |
| CI 配置 | `.github/workflows/ci.yml` | pytest + Ruff 自动化测试 |

### DAWS 算法参数（已锁定）

| 参数 | 值 | 说明 |
|------|-----|------|
| `DAWS_U_P90` | 0.90 | 争议周阈值（触发 α=0.60） |
| `DAWS_U_P97` | 0.97 | 极端周阈值（触发 α=0.70） |
| `DAWS_ALPHA_BASE` | 0.50 | 常规周评委权重 |
| `DAWS_ALPHA_DISPUTE` | 0.60 | 争议周评委权重 |
| `DAWS_ALPHA_EXTREME` | 0.70 | 极端周评委权重 |

## 目录与文件说明（逐项）

以下清单覆盖**当前目录中的每一个文件**，并标注用途、来源/生成方式与内容要点（尤其是 CSV/JSON 输出）。

### 根目录
- `.gitignore` — Git 忽略规则；忽略 `data/*.csv`、`outputs/`、LaTeX 编译产物、`__pycache__` 等。
- `README.md` — 项目总说明（本文件）。
- `requirements.txt` — Python 依赖清单（numpy/pandas/scipy/matplotlib/seaborn/statsmodels/scikit-learn/pulp/pytest）。

### .github/workflows
- `.github/workflows/ci.yml` — CI 配置（Ruff + pytest）。

### data
- `data/2026_MCM_Problem_C_Data.csv` — 题目给定的原始 DWTS 数据（**来源：赛题附带数据**，默认不入库）；内容为选手与每周评委打分的宽表。主要字段：`celebrity_name`、`ballroom_partner`、`celebrity_industry`、`celebrity_homestate`、`celebrity_homecountry/region`、`celebrity_age_during_season`、`season`、`results`、`placement`，以及 `week1_judge1_score`…`week11_judge4_score`（缺失为 `N/A`）。

### outputs（由 `src/pipeline.py` 生成；默认忽略入库）
- `outputs/audit_beta_sensitivity.json` — Hard-7 β 敏感性审计结果；包含 `beta_values_tested`、`integrity_range`、`agency_deviation_range`、`all_integrity_pass` 等。
- `outputs/audit_block5_gate.json` — Hard-5 Gate 诊断统计；包含 `accept_rate_strict_*`、`recommended_n_proposals_by_q_gate`、约束项与诊断来源说明。
- `outputs/audit_double_elim_check.json` — 双淘汰周对比审计；包含 `n_double_elim_weeks`、`median_accept_rate_strict_*`、`R_double_vs_single` 与理论对照。
- `outputs/audit_double_elim_structural_check.json` — Hard-3 结构性复核；记录 `structural_check_scope`、`eps_ord_used`、样本检查率与通过结论。
- `outputs/audit_lp_milp_scope.json` — LP/MILP 作用域核验；明确 LP/MILP 仅作诊断、不参与采样/后验/指标计算。
- `outputs/audit_week_meta.csv` — **逐季逐周**采样与审计元数据；列：`season`、`week`、`seed`、`n_proposals`、`accept_rate`、`n_accept`、`n_accept_fast`、`n_accept_strict`、`accept_rate_strict`、`min_violation`、`max_violation`、`percent_rule_residual`、`percent_rule_diagnostic`、`rank_rule_residual`、`rank_rule_diagnostic`、`eliminated_k`、`is_double_elim_week`、`n_contestants`、`strict_bottomk_check_rate`、`strict_bottomk_check_n`、`feasible_flag`、`fallback_flag`、`strict_feasible_flag`、`used_fallback`、`excluded_from_metrics`、`valid_week`、`violation_proxy`、`audit_weak`、`confidence_tag`。
- `outputs/beta_sensitivity.csv` — β 敏感性统计表；列：`beta`、`integrity`、`agency_deviation`、`conflict_samples`、`integrity_pass`、`note`。
- `outputs/daws_tiers.csv` — DAWS 风险分层输出；列：`Season`、`Week`、`Risk_Score`、`Tier_Label`、`Action`、`Audit_Weak`、`Confidence_Tag`。
- `outputs/fast_strict_metrics.json` — Fast vs Strict 核验摘要；含 `mae_mean`、`top1_agree`、`top2_agree`、`delta_fairness`、`delta_agency`、`delta_flip` 等。
- `outputs/run.log` — 一次 pipeline 运行日志（加载数据/采样/建模/作图进度）。
- `outputs/scale_benchmark.csv` — 规模基准测试；列：`n_proposals`、`seed`、`runtime_sec`、`mean_hdi`、`max_hdi`、`stability_daws`、`fairness_daws`、`flip_rate`、`daws_improve`。
- `outputs/summary_metrics.csv` — 关键指标汇总（CSV 版）；列：`seasons_feasible`、`mean_hdi_width`、`median_hdi_width`、`p90_hdi_width`、`max_hdi_width`、`flip_rate`、`daws_improve`、`percent_agency`、`percent_stability`、`percent_integrity`、`rank_agency`、`rank_stability`、`rank_integrity`、`daws_agency`、`stability_daws`、`integrity_daws`、`fairness_daws`、`audit_weak_rate`、`fast_strict_mae`、`fast_strict_top1`、`fast_strict_top2`、`fast_strict_delta_fairness`、`fast_strict_delta_agency`、`fast_strict_delta_flip`。
- `outputs/summary_metrics.json` — 与 `summary_metrics.csv` 同口径的 JSON 版本，便于程序读取。

### src
- `src/pipeline.py` — 主流水线：读取 `data/2026_MCM_Problem_C_Data.csv`，完成采样、审计、指标计算、图表渲染，并写入 `outputs/` 与 `paper/figures/`。
- `src/__pycache__/pipeline.cpython-312.pyc` — Python 字节码缓存（可删）。

### tests
- `tests/__init__.py` — 测试包标记文件。
- `tests/test_smoke.py` — 冒烟测试：以小规模采样验证 pipeline 可运行与关键输出存在。
- `tests/__pycache__/__init__.cpython-312.pyc` — Python 字节码缓存（可删）。
- `tests/__pycache__/test_smoke.cpython-312-pytest-9.0.2.pyc` — pytest 运行缓存（可删）。

### docs
- `docs/01_题目原文.md` — 题目原文（Markdown 版）。
- `docs/01_题目原文.pdf` — 题目原文（PDF 版）。
- `docs/02_功能说明.md` — 功能/模块说明。
- `docs/03_算法框架.md` — 算法框架与主流程说明。
- `docs/04_图表规范.md` — 图表风格与可视化规范。
- `docs/05_任务追踪清单.md` — 任务追踪与状态记录。
- `docs/06_开发规范.md` — 开发与 Git 规范。
- `docs/07_开发文档.md` — 环境/依赖/运行说明。
- `docs/08_终极重构冲刺清单.md` — 冲刺阶段清单。
- `docs/main_zh.docx` — 中文论文草稿（Docx 版）。
- `docs/main_zh.md` — 中文论文草稿（Markdown/LaTeX 片段混排版）。

### docs/archive（历史材料归档）
- `docs/archive/Block2_图注一致性检查表.md` — Block2 图注一致性检查记录。
- `docs/archive/Block3_附录索引.md` — Block3 附录索引。
- `docs/archive/“与星共舞”投票机制审计与DAWS算法正确性深度评估报告.md` — 专题审计报告（历史归档）。
- `docs/archive/任务清单.md` — 旧版任务清单。
- `docs/archive/图表逐一评审.md` — 图表逐条评审记录。
- `docs/archive/待改进任务.md` — 待改进项记录。
- `docs/archive/论文冲奖任务清单.md` — 冲奖版任务清单（历史）。
- `docs/archive/评审分析报告.md` — 评审分析报告（历史）。
- `docs/archive/题目详情.md` — 题目详情（历史）。

### docs/figures_png（文档预览用 PNG 图）
- `docs/figures_png/fig_auc_forward.png` — AUC 前瞻校准图（PNG）。
- `docs/figures_png/fig_beta_sensitivity.png` — β 敏感性图（PNG）。
- `docs/figures_png/fig_conflict_combo.png` — 冲突组合图（PNG）。
- `docs/figures_png/fig_conflict_map.png` — 冲突地图图（PNG）。
- `docs/figures_png/fig_controversy_ridgeline.png` — 争议脊线图（PNG）。
- `docs/figures_png/fig_counterfactual_risk_timeline.png` — 反事实风险时间线（PNG）。
- `docs/figures_png/fig_dashboard_concept.png` — 仪表盘概念图（PNG）。
- `docs/figures_png/fig_daws_trigger.png` — DAWS 触发图（PNG）。
- `docs/figures_png/fig_fast_vs_strict.png` — Fast vs Strict 对比图（PNG）。
- `docs/figures_png/fig_feature_scatter.png` — 特征散点图（PNG）。
- `docs/figures_png/fig_hdi_distribution.png` — HDI 分布图（PNG）。
- `docs/figures_png/fig_judgesave_curve.png` — Judge-save 曲线（PNG）。
- `docs/figures_png/fig_mechanism_compare.png` — 机制对比图（PNG）。
- `docs/figures_png/fig_mechanism_radar.png` — 机制雷达图（PNG）。
- `docs/figures_png/fig_ppc_summary.png` — PPC 摘要图（PNG）。
- `docs/figures_png/fig_pro_diff_forest.png` — 专业差异森林图（PNG）。
- `docs/figures_png/fig_rule_switch_ci.png` — 规则切换置信区间（PNG）。
- `docs/figures_png/fig_scale_benchmark.png` — 规模基准图（PNG）。
- `docs/figures_png/fig_sigma_sensitivity.png` — Sigma 敏感性图（PNG）。
- `docs/figures_png/fig_synthetic_validation.png` — 合成验证图（PNG）。
- `docs/figures_png/fig_uncertainty_heatmap.png` — 不确定性热力图（PNG）。
- `docs/figures_png/fig_wrongful_heatmap.png` — 错误淘汰热力图（PNG）。

### paper（论文源文件与编译产物）
- `paper/MCM-ICM_Summary.tex` — MCM/ICM 摘要页 LaTeX 源文件。
- `paper/summary_metrics.tex` — 指标汇总 LaTeX 片段（由 pipeline 写入，用于论文中引用）。
- `paper/main.tex` — 英文论文 LaTeX 主文件。
- `paper/main.pdf` — 英文论文编译产物。
- `paper/main.aux` — LaTeX 辅助文件（可删）。
- `paper/main.fdb_latexmk` — LaTeX 构建缓存（可删）。
- `paper/main.fls` — LaTeX 构建依赖清单（可删）。
- `paper/main.log` — LaTeX 编译日志（可删）。
- `paper/main.out` — LaTeX 输出辅助文件（可删）。
- `paper/main.toc` — 目录缓存（可删）。
- `paper/main_page1-01.png` — 英文论文首页渲染图（PNG）。
- `paper/main_zh.tex` — 中文论文 LaTeX 主文件。
- `paper/main_zh.pdf` — 中文论文编译产物。
- `paper/compile_zh.log` — 中文编译日志（可删）。
- `paper/main_zh.aux` — LaTeX 辅助文件（可删）。
- `paper/main_zh.fdb_latexmk` — LaTeX 构建缓存（可删）。
- `paper/main_zh.fls` — LaTeX 构建依赖清单（可删）。
- `paper/main_zh.log` — LaTeX 编译日志（可删）。
- `paper/main_zh.out` — LaTeX 输出辅助文件（可删）。
- `paper/main_zh.synctex.gz` — SyncTeX 文件（可删）。
- `paper/main_zh.toc` — 目录缓存（可删）。

### paper/figures（论文用 PDF 图）
- `paper/figures/fig_alluvial_finalists.pdf` — 决赛选手流图（PDF）。
- `paper/figures/fig_auc_forward.pdf` — AUC 前瞻校准图（PDF）。
- `paper/figures/fig_beta_sensitivity.pdf` — β 敏感性图（PDF）。
- `paper/figures/fig_conflict_combo.pdf` — 冲突组合图（PDF）。
- `paper/figures/fig_conflict_map.pdf` — 冲突地图图（PDF）。
- `paper/figures/fig_controversy_ridgeline.pdf` — 争议脊线图（PDF）。
- `paper/figures/fig_counterfactual_risk_timeline.pdf` — 反事实风险时间线（PDF）。
- `paper/figures/fig_dashboard_concept.pdf` — 仪表盘概念图（PDF）。
- `paper/figures/fig_daws_trigger.pdf` — DAWS 触发图（PDF）。
- `paper/figures/fig_fast_vs_strict.pdf` — Fast vs Strict 对比图（PDF）。
- `paper/figures/fig_feature_scatter.pdf` — 特征散点图（PDF）。
- `paper/figures/fig_hdi_distribution.pdf` — HDI 分布图（PDF）。
- `paper/figures/fig_judgesave_curve.pdf` — Judge-save 曲线（PDF）。
- `paper/figures/fig_mechanism_compare.pdf` — 机制对比图（PDF）。
- `paper/figures/fig_mechanism_compare_conflict.pdf` — 机制对比（冲突周）图（PDF）。
- `paper/figures/fig_mechanism_distribution.pdf` — 机制指标分布图（PDF）。
- `paper/figures/fig_mechanism_radar.pdf` — 机制雷达图（PDF）。
- `paper/figures/fig_mechanism_radar_conflict.pdf` — 机制雷达（冲突周）图（PDF）。
- `paper/figures/fig_param_optimization.pdf` — 参数优化示意图（PDF）。
- `paper/figures/fig_pareto_2d.pdf` — Pareto 2D 图（PDF）。
- `paper/figures/fig_ppc_summary.pdf` — PPC 摘要图（PDF）。
- `paper/figures/fig_pro_diff_forest.pdf` — 专业差异森林图（PDF）。
- `paper/figures/fig_rule_switch.pdf` — 规则切换概率图（PDF）。
- `paper/figures/fig_rule_switch_ci.pdf` — 规则切换置信区间图（PDF）。
- `paper/figures/fig_scale_benchmark.pdf` — 规模基准图（PDF）。
- `paper/figures/fig_sigma_sensitivity.pdf` — Sigma 敏感性图（PDF）。
- `paper/figures/fig_synthetic_validation.pdf` — 合成验证图（PDF）。
- `paper/figures/fig_uncertainty_heatmap.pdf` — 不确定性热力图（PDF）。
- `paper/figures/fig_wrongful_heatmap.pdf` — 错误淘汰热力图（PDF）。

### paper/figures/snips
- `paper/figures/snips/README.md` — 截图目录说明与命名规范。

### paper/figure_renders（PDF 图转 PNG 预览）
- `paper/figure_renders/fig_alluvial_finalists-1.png` — 决赛选手流图渲染（PNG）。
- `paper/figure_renders/fig_auc_forward-1.png` — AUC 前瞻校准图渲染（PNG）。
- `paper/figure_renders/fig_conflict_combo-1.png` — 冲突组合图渲染（PNG）。
- `paper/figure_renders/fig_conflict_map-1.png` — 冲突地图图渲染（PNG）。
- `paper/figure_renders/fig_controversy_ridgeline-1.png` — 争议脊线图渲染（PNG）。
- `paper/figure_renders/fig_fast_vs_strict-1.png` — Fast vs Strict 对比图渲染（PNG）。
- `paper/figure_renders/fig_feature_scatter-1.png` — 特征散点图渲染（PNG）。
- `paper/figure_renders/fig_hdi_distribution-1.png` — HDI 分布图渲染（PNG）。
- `paper/figure_renders/fig_judgesave_curve-1.png` — Judge-save 曲线渲染（PNG）。
- `paper/figure_renders/fig_mechanism_compare-1.png` — 机制对比图渲染（PNG）。
- `paper/figure_renders/fig_mechanism_radar-1.png` — 机制雷达图渲染（PNG）。
- `paper/figure_renders/fig_pareto_2d-1.png` — Pareto 2D 图渲染（PNG）。
- `paper/figure_renders/fig_ppc_summary-1.png` — PPC 摘要图渲染（PNG）。
- `paper/figure_renders/fig_pro_diff_forest-1.png` — 专业差异森林图渲染（PNG）。
- `paper/figure_renders/fig_rule_switch-1.png` — 规则切换概率图渲染（PNG）。
- `paper/figure_renders/fig_rule_switch_ci-1.png` — 规则切换置信区间渲染（PNG）。
- `paper/figure_renders/fig_scale_benchmark-1.png` — 规模基准图渲染（PNG）。
- `paper/figure_renders/fig_sigma_sensitivity-1.png` — Sigma 敏感性图渲染（PNG）。
- `paper/figure_renders/fig_uncertainty_heatmap-1.png` — 不确定性热力图渲染（PNG）。
- `paper/figure_renders/fig_wrongful_heatmap-1.png` — 错误淘汰热力图渲染（PNG）。

### paper/page_renders（论文每页 PNG 预览）
- `paper/page_renders/main_page-01.png` — 英文论文第 1 页渲染。
- `paper/page_renders/main_page-02.png` — 英文论文第 2 页渲染。
- `paper/page_renders/main_page-03.png` — 英文论文第 3 页渲染。
- `paper/page_renders/main_page-04.png` — 英文论文第 4 页渲染。
- `paper/page_renders/main_page-05.png` — 英文论文第 5 页渲染。
- `paper/page_renders/main_page-06.png` — 英文论文第 6 页渲染。
- `paper/page_renders/main_page-07.png` — 英文论文第 7 页渲染。
- `paper/page_renders/main_page-08.png` — 英文论文第 8 页渲染。
- `paper/page_renders/main_page-09.png` — 英文论文第 9 页渲染。
- `paper/page_renders/main_page-10.png` — 英文论文第 10 页渲染。
- `paper/page_renders/main_page-11.png` — 英文论文第 11 页渲染。
- `paper/page_renders/main_page-12.png` — 英文论文第 12 页渲染。
- `paper/page_renders/main_page-13.png` — 英文论文第 13 页渲染。
- `paper/page_renders/main_page-14.png` — 英文论文第 14 页渲染。
- `paper/page_renders/main_page-15.png` — 英文论文第 15 页渲染。
- `paper/page_renders/main_page-16.png` — 英文论文第 16 页渲染。
- `paper/page_renders/main_page-17.png` — 英文论文第 17 页渲染。
- `paper/page_renders/main_page-18.png` — 英文论文第 18 页渲染。
- `paper/page_renders/main_page-19.png` — 英文论文第 19 页渲染。
- `paper/page_renders/main_page-20.png` — 英文论文第 20 页渲染。
- `paper/page_renders/main_page-21.png` — 英文论文第 21 页渲染。
- `paper/page_renders/main_page-22.png` — 英文论文第 22 页渲染。
- `paper/page_renders/main_page-23.png` — 英文论文第 23 页渲染。
- `paper/page_renders/main_page-24.png` — 英文论文第 24 页渲染。
- `paper/page_renders/main_page-25.png` — 英文论文第 25 页渲染。
- `paper/page_renders/main_page-26.png` — 英文论文第 26 页渲染。

### 缓存/临时文件（可删）
- `.pytest_cache/.gitignore` — pytest 缓存配置（可删）。
- `.pytest_cache/CACHEDIR.TAG` — pytest 缓存标记（可删）。
- `.pytest_cache/README.md` — pytest 缓存说明（可删）。
- `.pytest_cache/v/cache/nodeids` — pytest 用例缓存（可删）。
- `.ruff_cache/.gitignore` — Ruff 缓存配置（可删）。
- `.ruff_cache/CACHEDIR.TAG` — Ruff 缓存标记（可删）。
- `.ruff_cache/0.14.14/12561587934736260300` — Ruff 缓存数据（可删）。
- `.ruff_cache/0.14.14/6790024607104991487` — Ruff 缓存数据（可删）。

## 开发者提示（重要）

- 每次完成一个可验收的任务或 Block，请立即执行 `git add`、`git commit`、`git push`，保持远程仓库与本地工作树同步，避免丢失或冲突。
- 推荐的最小提交流程（中文提交信息）：

```bash
git add <changed-files>
git commit -m "块/图表/修复：简短中文说明（例如：完成 Block2 图注修正）"
git push origin $(git rev-parse --abbrev-ref HEAD)
```

- 提交信息建议遵守：`<模块或区域>：<中文短语，动词开头，说明做了什么>`，例如：
	- `图表：修正热力图轴并补充图注`
	- `文档：归档历史文档并规范命名`

- 注意：数据集文件应保持私有，不要将 `data/2026_MCM_Problem_C_Data.csv` 提交到仓库（项目已将 `data/` 目录写入 `.gitignore`）。文档、代码与论文源文件应保留在版本控制中。

如需我为你自动执行提交和推送，请确认并授权我进行 Git 操作。 
