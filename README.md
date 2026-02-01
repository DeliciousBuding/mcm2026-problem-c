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
