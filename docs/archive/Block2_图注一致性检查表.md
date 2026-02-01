# Block 2 图注一致性检查表

**生成日期**：2026-02-01  
**验证状态**：✅ 全部通过

---

## 图表编号与引用一致性

| 图表名称 | 英文 Label | 英文引用 | 中文 Label | 中文引用 | 状态 |
|---------|------------|----------|------------|----------|------|
| 反事实风险时间线 | `fig:counterfactual-risk` | ✅ | `fig:counterfactual-risk` | ✅ | 通过 |
| DAWS 触发图 | `fig:daws-trigger` | ✅ | `fig:daws-trigger` | ✅ | 通过 |
| 机制雷达图 | `fig:mechanism-radar` | ✅ | `fig:mechanism-radar` | ✅ | 通过 |
| 机制比较图 | `fig:mechanism-compare` | ✅ | `fig:mechanism-compare` | ✅ | 通过 |
| Fast vs Strict | `fig:fast-vs-strict` | ✅ | `fig:fast-vs-strict` | ✅ | 通过 |
| 不确定性热力图 | `fig:uncertainty-heatmap` | ✅ | `fig:uncertainty-heatmap` | ✅ | 通过 |
| HDI 分布图 | `fig:hdi-distribution` | ✅ | `fig:hdi-distribution` | ✅ | 通过 |
| Pareto 2D | `fig:pareto-2d` | ✅ | `fig:pareto-2d` | ✅ | 通过 |
| 冲突散点图 | `fig:conflict-map` | ✅ | `fig:conflict-map` | ✅ | 通过 |
| PPC 汇总图 | `fig:ppc-summary` | ✅ | `fig:ppc-summary` | ✅ | 通过 |
| Judge-save 曲线 | `fig:judgesave-curve` | ✅ | `fig:judgesave-curve` | ✅ | 通过 |
| 特征散点图 | `fig:feature-scatter` | ✅ | `fig:feature-scatter` | ✅ | 通过 |

---

## 图注编码完整性检查

### 热力图 (`fig_uncertainty_heatmap.pdf`, `fig_wrongful_heatmap.pdf`)

| 检查项 | 英文版 | 中文版 | 状态 |
|--------|--------|--------|------|
| 轴从 Week 1 开始 | ✅ | ✅ | 通过 |
| 缺失数据说明 | ✅ 副标题 "(blank cells = no elimination or missing data)" | ✅ | 通过 |
| 颜色图例说明 | ✅ | ✅ | 通过 |

### 冲突散点图 (`fig_conflict_combo.pdf`)

| 检查项 | 英文版 | 中文版 | 状态 |
|--------|--------|--------|------|
| 点大小 = HDI 宽度 | ✅ 图注说明 | ✅ | 通过 |
| 外圈 = wrongful 标记 | ✅ 图注说明 | ✅ | 通过 |
| 颜色含义说明 | ✅ | ✅ | 通过 |

### PPC 汇总图 (`fig_ppc_summary.pdf`)

| 检查项 | 英文版 | 中文版 | 状态 |
|--------|--------|--------|------|
| Coverage 方向 (↑ = 好) | ✅ 标签 "Coverage ↑" | ✅ | 通过 |
| Brier 方向 (↓ = 好) | ✅ 标签 "Brier ↓" | ✅ | 通过 |
| 数值标注 | ✅ 每条柱添加数值 | ✅ | 通过 |

### Fast vs Strict 图 (`fig_fast_vs_strict.pdf`)

| 检查项 | 英文版 | 中文版 | 状态 |
|--------|--------|--------|------|
| MAE 数值标注 | ✅ MAE = 0.0045 | ✅ | 通过 |
| Top-1 一致率标注 | ✅ Top-1 = 76.7% | ✅ | 通过 |
| Top-2 一致率标注 | ✅ Top-2 = 80.0% | ✅ | 通过 |

### Judge-save 曲线 (`fig_judgesave_curve.pdf`)

| 检查项 | 英文版 | 中文版 | 状态 |
|--------|--------|--------|------|
| β 数值或说明 | ✅ 标注 "β = 1.8 (illustrative)" | ✅ | 通过 |

### 特征散点图 (`fig_feature_scatter.pdf`)

| 检查项 | 英文版 | 中文版 | 状态 |
|--------|--------|--------|------|
| Top 5 偏离特征标注 | ✅ 图中标注 | ✅ | 通过 |

---

## 验证方法

1. **LaTeX 编译验证**：
   - 英文版 `main.tex`：pdfLaTeX 二次编译无 undefined reference
   - 中文版 `main_zh.tex`：XeLaTeX 二次编译无 undefined reference

2. **图注内容检查**：
   - 热力图：副标题说明缺失数据含义
   - 冲突散点：图注包含 HDI、wrongful、颜色编码说明
   - PPC：方向性指标 (↑/↓) 和数值标注
   - Fast vs Strict：MAE/Top-1/Top-2 注释框
   - Judge-save：β 值和 illustrative 说明
   - 特征散点：Top 5 偏离特征自动标注

---

## 备注

- 机制指标分布图（可选任务）未实现，已标记为可选
- 所有图表通过 `src/pipeline.py` 统一生成，保证中英文版本一致
