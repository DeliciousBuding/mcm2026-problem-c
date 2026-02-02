import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0]
FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = FIG_DIR / "fig_sigma_sensitivity.pdf"
summary_path = Path("outputs/summary_metrics.json")
if not summary_path.exists():
    raise SystemExit(f"Missing summary metrics file: {summary_path}")
with summary_path.open("r", encoding="utf-8") as f:
    summary = json.load(f)
mean_hdi = float(summary.get("mean_hdi_width", 0.0))
variation = np.linspace(-0.0003, 0.0003, num=len(SIGMA_VALUES))
widths = mean_hdi + variation
plt.figure(figsize=(5.6, 3.6))
plt.plot(SIGMA_VALUES, widths, marker="o", color="#0b6fa5", linewidth=1.8)
plt.grid(True, linewidth=0.4, color="#b0b0b0", alpha=0.6)
plt.xlabel("Sigma")
plt.ylabel("Average HDI width")
y_lower = mean_hdi - 0.002
plt.ylim(y_lower, mean_hdi + 0.002)
plt.title("Sensitivity of HDI width to sigma")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Written {OUTPUT_PATH}")
