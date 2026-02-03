# MCM/ICM 2026 — Problem C (DWTS)

本仓库包含 COMAP 2026 Problem C（Dancing with the Stars）的数据处理、建模、可视化与论文产出流水线。

## 快速入口
- **论文**：`paper/main.pdf`（英文），`paper/main_zh.pdf`（中文）
- **主代码**：`src/pipeline.py`
- **算法框架**：`docs/03_算法框架.md`
- **图表**：`paper/figures/`
- **指标汇总**：`outputs/summary_metrics.json` / `paper/summary_metrics.tex`
- **文档索引**：`docs/README.md`

## 目录结构（整理后）
```
.
├─ data/                     # 输入数据（2026_MCM_Problem_C_Data.csv）
├─ docs/                     # 正式文档
│  ├─ 01_题目原文.md/.pdf
│  ├─ README.md              # docs 索引
│  ├─ 03_算法框架.md
│  ├─ 04_图表规范.md
│  ├─ 06_开发规范.md
│  ├─ 07_开发文档.md
│  ├─ 09_AI policy.md
│  └─ archive/               # 历史材料、草稿与任务清单
├─ outputs/                  # 运行输出（CSV/JSON/日志/基准）
├─ paper/                    # 论文源文件与 PDF
│  └─ figures/               # 论文图表（PDF）
├─ scripts/                  # 辅助脚本（如有）
├─ src/                      # 源代码（pipeline）
├─ tests/                    # 测试
└─ README.md
```

## 运行流水线
```bat
conda activate mcm2026
python -u src\pipeline.py
```
可选环境变量：
- `MCM_N_PROPOSALS`：采样规模
- `MCM_MULTIPROC`：多进程（1 启用）
- `MCM_WORKERS`：并行进程数

## 编译论文
```bat
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
```

## 文档索引
- 文档目录：`docs/README.md`（完整索引）
- 算法框架：`docs/03_算法框架.md`
- 图表规范：`docs/04_图表规范.md`
- 开发规范：`docs/06_开发规范.md`
- 环境说明：`docs/07_开发文档.md`
- AI 使用政策：`docs/09_AI policy.md`

## 更新记录（节选，中文翻译）
- `a8fd9ad`：论文与图表最终定稿  
- `f37aafa`：数据基准——6K–12K/500 规模结果加密（elbow）  
- `36d2811`：流水线新增 `skip_render`，批量实验约提速 43%  
- `4bd4d1a`：Block 6 相关产物全量更新（figures/outputs/paper）  
- `3650077`：Block 6 基准多 seed 误差带增强  
- `88c1c83`：英文版补充 Appendix B/C（Hard-3/Hard-2）与结论局限  
- `94ea1da`：文档版本号更新至 v0.4.0-final  
- `11230e7`：v0.4.0 全 Hard 项闭环 + Δ 口径修复 + 论文叙事对齐  

## 备注
- `outputs/` 与 `paper/figures/` 均由流水线生成。
- `docs/archive/` 为历史材料与过程记录，方便回溯但不参与最终产出。
