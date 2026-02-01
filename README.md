# MCM/ICM 2026 Problem C: Data With The Stars

<p align="center">
  <img src="https://img.shields.io/badge/MCM%2FICM-2026%20Problem%20C-blue" alt="MCM/ICM 2026"/>
  <img src="https://img.shields.io/badge/Python-3.9+-green" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/LaTeX-XeLaTeX-orange" alt="LaTeX"/>
  <img src="https://img.shields.io/badge/Version-v0.3.3-brightgreen" alt="Version"/>
</p>

本仓库包含 **DWTS（Dancing with the Stars）** 投票机制审计与优化设计的完整数学建模解决方案，涵盖模型实现、可视化及论文产出。

## 📋 项目概述

**Dancing with the Stars** 是一档国际知名的舞蹈真人秀节目，本项目针对其投票机制进行深入分析：

- 🔍 **核心问题**：评委评分与观众投票如何组合？不同规则对比赛结果有何影响？
- 🎯 **研究目标**：估计未公开的观众投票、评估现有机制、提出改进方案
- 💡 **创新点**：Polytope Inversion Audit + DAWS (Dynamic Adaptive Weighting System)

### 主要任务

| 任务 | 描述 |
|------|------|
| **Task 1** | 建模估计未知的观众投票（fan votes） |
| **Task 2** | 对比 rank 与 percentage 两种投票组合方法 |
| **Task 3** | 分析职业舞伴、明星特征对成绩的影响 |
| **Task 4** | 设计更公平/更有吸引力的投票机制（DAWS） |

## 📁 目录结构

```
.
├── 📂 docs/                          # 文档与题目说明
│   ├── 题目原文.md                    # 题目原文（英文）
│   ├── 功能说明.md                    # 功能模块说明
│   ├── 开发文档.md                    # 开发指南与 Git 工作流规范
│   ├── 算法框架.md                    # 冲奖算法与论文框架
│   ├── 图表规范.md                    # 图表设计规范
│   ├── 论文冲奖任务清单.md             # 任务追踪
│   └── 历史存档/                     # 历史文档归档
│       ├── 任务清单.md
│       ├── 待改进任务.md
│       ├── 评审分析报告.md
│       ├── 图表逐一评审.md
│       └── ...
│
├── 📂 outputs/                       # 运行输出
│   ├── summary_metrics.json          # 汇总指标（JSON）
│   ├── summary_metrics.csv           # 汇总指标（CSV）
│   ├── scale_benchmark.csv           # 规模实验结果
│   └── fast_strict_metrics.json      # Fast vs Strict 校验指标
│
├── 📂 paper/                         # 论文与图表
│   ├── main.tex                      # 英文论文（主版本）
│   ├── main_zh.tex                   # 中文论文
│   ├── summary_metrics.tex           # 指标宏（自动生成）
│   └── figures/                      # 自动生成的图表（20+ PDF）
│
├── 📂 src/
│   └── pipeline.py                   # 主流水线代码（~1900 行）
│
├── 2026_MCM_Problem_C_Data.csv       # 官方数据集（S1-S34）
├── 2026_MCM_Problem_C.pdf            # 题目 PDF
├── MCM-ICM_Summary.tex               # Summary Sheet 模板
└── README.md                         # 本文件
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.9+
- **包管理**: Miniforge / Conda / pip
- **LaTeX**: TeX Live / MiKTeX（含 XeLaTeX）

### 安装依赖

```bash
# 使用 Conda（推荐）
conda create -n mcm2026 python=3.10
conda activate mcm2026
pip install -r requirements.txt

# 或直接安装
pip install numpy pandas matplotlib seaborn scipy pulp statsmodels scikit-learn
```

### 运行主流程

```bash
# Windows
conda activate mcm2026
python -u src\pipeline.py

# Linux / macOS
conda activate mcm2026
python -u src/pipeline.py
```

### 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MCM_N_PROPOSALS` | 250 | 每周 Dirichlet 提案数量 |
| `MCM_MULTIPROC` | 1 | 是否启用多进程（0=禁用） |
| `MCM_WORKERS` | CPU-2 | 并行进程数 |
| `MCM_FAST_STRICT` | 1 | 是否进行 Fast vs Strict 校验 |
| `MCM_RULE_BOOT` | 200 | 规则切换置信带 bootstrap 次数 |

```bash
# 高精度运行示例
MCM_N_PROPOSALS=500 MCM_RULE_BOOT=500 python -u src/pipeline.py
```

## 📊 生成的图表

运行 `pipeline.py` 后，将在 `paper/figures/` 生成以下图表：

| 图表文件 | 描述 |
|----------|------|
| `fig_conflict_map.pdf` | Judge vs Fan 冲突热力图 |
| `fig_conflict_combo.pdf` | 冲突指数组合图 |
| `fig_uncertainty_heatmap.pdf` | 投票不确定性热力图 |
| `fig_wrongful_heatmap.pdf` | 错误淘汰概率热图 |
| `fig_rule_switch.pdf` | 规则切换后验概率曲线 |
| `fig_rule_switch_ci.pdf` | 规则切换置信区间 |
| `fig_mechanism_radar.pdf` | 机制评估雷达图 |
| `fig_mechanism_compare.pdf` | 机制对比条形图 |
| `fig_pareto_2d.pdf` | Pareto 权衡图（agency vs stability） |
| `fig_alluvial_finalists.pdf` | 决赛阵容流向图 |
| `fig_pro_diff_forest.pdf` | 职业舞伴效应森林图 |
| `fig_auc_forward.pdf` | 分季 AUC 预测曲线 |
| ... | 更多图表详见 `docs/功能说明.md` |

## 📝 论文编译

```bash
cd paper

# 英文版（pdfLaTeX）
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex  # 运行两次生成目录

# 中文版（XeLaTeX）
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
```

> 💡 **提示**：运行 `pipeline.py` 后会自动更新 `paper/summary_metrics.tex`，确保论文中的数值始终与最新计算结果一致。

## 🧮 核心算法

### 1. Polytope Inversion Audit（投票可行集逆推）
- 使用 LP/MILP 从淘汰结果恢复"粉丝票可行多面体"
- 量化"规则一致但信息不足"的不确定性

### 2. Uncertainty-aware Reconstruction（不确定性量化）
- Dirichlet 采样 + 可行性约束筛选
- 输出后验均值与 95% HDI 区间

### 3. Mechanism Counterfactual（反事实机制评估）
- 对比 rank / percent / judge-save / DAWS 四种机制
- 计算冲突指数（Kendall τ）、观众能动性、稳定性

### 4. DAWS（Dynamic Adaptive Weighting System）
- 动态调整评委/观众权重
- 在公平性、观众主权、稳定性之间寻求最优平衡

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| [02_功能模块说明.md](docs/02_功能模块说明.md) | 功能模块详细说明与输出清单 |
| [03_算法与论文框架.md](docs/03_算法与论文框架.md) | 冲奖版算法与论文框架 |
| [04_图表设计规范.md](docs/04_图表设计规范.md) | 图表设计规范与配色方案 |
| [06_开发规范.md](docs/06_开发规范.md) | 开发与调试指南、Git 工作流规范 |
| [01_题目原文.md](docs/01_题目原文.md) | 题目原文（英文） |
| [05_任务追踪清单.md](docs/05_任务追踪清单.md) | 论文优化任务追踪 |

## 📈 输出指标

运行后生成的 `outputs/summary_metrics.json` 包含：

```json
{
  "feasible_seasons": 34,
  "avg_hdi_width": 0.082,
  "conflict_kendall_tau": -0.23,
  "daws_integrity": 0.91,
  "daws_agency": 0.78,
  "daws_stability": 0.85,
  ...
}
```

## 🔧 技术栈

- **数值计算**: NumPy, SciPy
- **数据处理**: Pandas
- **可视化**: Matplotlib, Seaborn
- **优化求解**: PuLP（LP/MILP）
- **统计建模**: Statsmodels
- **机器学习**: Scikit-learn（GBDT）
- **排版**: LaTeX (pdfLaTeX + XeLaTeX)

## 📄 许可证

本项目仅供 MCM/ICM 2026 竞赛学习交流使用。

---

<p align="center">
  <b>MCM/ICM 2026 Problem C - Data With The Stars</b><br>
  Built with ❤️ for Mathematical Modeling
</p>
