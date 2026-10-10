"""Publication-style summary figures from saved measurement CSVs only."""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "results"


def read(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def yes(row, field):
    return row[field] == "True"


clean = read("clean_packets.csv")
crop = read("crop_packets.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

clean_raw = sum(yes(row, "raw_exact") for row in clean)
clean_protected = sum(yes(row, "useful_exact") for row in clean)
axes[0].bar([0, 1], [clean_raw / len(clean), clean_protected / len(clean)],
            color=["#9aafb9", "#247f74"], width=.6)
axes[0].set_xticks([0, 1], ["Raw WAM 32-bit", "BCH protected 16-bit"])
axes[0].set_ylim(0, 1.08)
axes[0].set_ylabel("Exact local payload recovery")
axes[0].set_title("Clean four-region test (120 packets)")
for i, count in enumerate((clean_raw, clean_protected)):
    axes[0].text(i, count/len(clean)+.02, f"{count}/120", ha="center", fontsize=10)
axes[0].text(.5, .79, "16 useful bits per 32 WAM bits", ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#bbbbbb"))

labels = ["0%", "(0, 25%)", "[25, 75%)", "[75, 100%]"]
for area, color, offset in (("0.75", "#247f74", -.18), ("0.5", "#d17b3e", .18)):
    rates, counts = [], []
    selected = [row for row in crop if row["retained_target"] == area]
    for index in range(4):
        subset = []
        for row in selected:
            retained = float(row["retained_fraction"])
            bucket = 0 if retained == 0 else 1 if retained < .25 else 2 if retained < .75 else 3
            if bucket == index:
                subset.append(row)
        counts.append(len(subset))
        rates.append(sum(yes(row, "useful_exact") for row in subset)/len(subset) if subset else 0)
    bars = axes[1].bar(np.arange(4)+offset, rates, .34, label=f"{int(float(area)*100)}% image kept", color=color)
    for bar, n in zip(bars, counts):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02, f"n={n}",
                     ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(np.arange(4), labels)
axes[1].set_ylim(0, 1.08)
axes[1].set_xlabel("Original local region physically retained")
axes[1].set_ylabel("Exact 16-bit useful payload recovery")
axes[1].set_title("Blind arbitrary-offset crop (720 region cases)")
axes[1].legend(frameon=False, loc="upper left", fontsize=9)

fig.savefig(ROOT / "summary_figure.png", dpi=180)
fig.savefig(ROOT / "summary_figure.pdf")
print(ROOT / "summary_figure.png")
