"""Static scientific summary from completed CSV/JSON; no new WAM inference."""
import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

summary = json.loads((OUT / "final_comparison_summary.json").read_text())
with (OUT / "paired_global_outcomes.csv").open() as stream:
    paired = list(csv.DictReader(stream))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
areas = (.75, .5)
positions = np.arange(2)
colors = {"repetition": "#7c97aa", "mds": "#237f74"}
for shift, method in ((-.18, "repetition"), (.18, "mds")):
    values = []
    counts = []
    for area in areas:
        row = next(item for item in summary["by_method_and_area"]
                   if item["method"] == method and item["retained_target"] == area)
        values.append(row["exact_global_rate"])
        counts.append(row["exact_global"])
    bars = axes[0].bar(positions+shift, values, .34, color=colors[method],
                       label="Spatial repetition" if method == "repetition" else "MDS(8,4)")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.016,
                     f"{count}/90", ha="center", fontsize=9)
axes[0].set_ylim(0, 1.08)
axes[0].set_xticks(positions, ["75% area kept", "50% area kept"])
axes[0].set_ylabel("Exact 32-bit global payload recovery")
axes[0].set_title("Paired blind arbitrary-offset crops")
axes[0].legend(frameon=False, loc="lower left", fontsize=9)

labels = ["Both exact", "MDS only", "Repetition only", "Neither exact"]
colors2 = ["#7b9d83", "#237f74", "#bb7252", "#9e9e9e"]
for area_index, area in enumerate(areas):
    subset = [row for row in paired if float(row["retained_target"]) == area]
    counts = Counter((row["repetition_exact"], row["mds_exact"]) for row in subset)
    values = [counts[("True", "True")], counts[("False", "True")],
              counts[("True", "False")], counts[("False", "False")]]
    base = 0
    for label, color, value in zip(labels, colors2, values):
        axes[1].bar(area_index, value, bottom=base, width=.58, color=color,
                    label=label if area_index == 0 else None)
        if value:
            axes[1].text(area_index, base+value/2, str(value), ha="center", va="center", fontsize=9)
        base += value
axes[1].set_ylim(0, 93)
axes[1].set_xticks(positions, ["75% area kept", "50% area kept"])
axes[1].set_ylabel("Paired crop cases (n=90 each)")
axes[1].set_title("Where the methods agree and differ")
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12),
               ncol=2, fontsize=8)

fig.savefig(OUT / "final_comparison.png", dpi=180)
fig.savefig(OUT / "final_comparison.pdf")
print(OUT / "final_comparison.png")
