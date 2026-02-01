# Block 3 附录索引

**生成日期**：2026-02-01  
**状态**：✅ 完成

---

## 附录结构

### 英文论文 (`main.tex`)

| 附录编号 | 标题 | Label | 内容 |
|----------|------|-------|------|
| Appendix A | Sensitivity Analysis | `app:sensitivity` | $\sigma$ 平滑参数敏感性分析，`fig_sigma_sensitivity.pdf` |
| Appendix B | Predictive Calibration | `app:auc` | GBDT 前向链式 AUC 校准曲线，`fig_auc_forward.pdf` |

### 中文论文 (`main_zh.tex`)

| 附录编号 | 标题 | Label | 内容 |
|----------|------|-------|------|
| 附录 A | 敏感性分析 | `app:sensitivity` | $\sigma$ 平滑参数敏感性分析，`fig_sigma_sensitivity.pdf` |
| 附录 B | 预测校准 | `app:auc` | GBDT 前向链式 AUC 校准曲线，`fig_auc_forward.pdf` |

---

## 正文引用位置

### 敏感性分析

- **英文**：Section "Truncated Posterior with Smoothness" → "see Appendix~\ref{app:sensitivity} for details"
- **中文**：Section "平滑后验" → "详见附录~\ref{app:sensitivity}"

### AUC 校准

- **英文**：Section "Predictive Add-on: GBDT" → "see Appendix~\ref{app:auc} for the calibration curve"
- **中文**：Section "预测补充：GBDT" → "详见附录~\ref{app:auc}"

---

## 设计原则

1. **附录不抢主线**：敏感性分析和预测校准作为鲁棒性补充，主结论保留在正文
2. **正文一句话引用**：每个附录在正文中有明确的引用位置
3. **图表完整保留**：原有图表移至附录，保持图注完整性

---

## 新增目录

- `paper/figures/snips/`：用于存放截图文件（已创建，含 README.md）
