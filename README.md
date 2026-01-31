# MCM/ICM 2026 Problem C

本仓库包含 DWTS（Dancing with the Stars）题目的完整建模、可视化与论文产出。

## 目录结构
```
.
├─ docs/                    # 文档与题目说明
│  ├─ 2026_MCM_Problem_C.md
│  ├─ 开发文档.md
│  ├─ 功能说明.md
│  ├─ 算法框架.md
│  ├─ 图表规范.md
│  └─ 题目详情.md
├─ outputs/                 # 运行输出指标
│  ├─ summary_metrics.json
│  └─ summary_metrics.csv
├─ paper/                   # 论文与图表
│  ├─ figures/              # 自动生成图表
│  ├─ main.tex              # 英文论文
│  ├─ main.pdf              # 英文论文编译输出
│  ├─ main_zh.tex           # 中文论文
│  ├─ main_zh.pdf           # 中文论文编译输出
│  └─ summary_metrics.tex   # 指标宏（自动生成）
├─ src/
│  └─ pipeline.py           # 主流水线代码
├─ 2026_MCM_Problem_C_Data.csv
├─ 2026_MCM_Problem_C.pdf
├─ MCM-ICM_Summary.tex
└─ .gitignore
```

## 环境与运行
- 推荐环境：Miniforge + `mcm2026`
- 运行：
```bat
conda activate mcm2026
python -u src\pipeline.py
```

## 论文编译
```bat
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
```

## 说明
- 图表遵循 `docs/图表规范.md`
- 算法与论文框架遵循 `docs/算法框架.md`
- 运行结束后会自动更新 `paper/summary_metrics.tex`
