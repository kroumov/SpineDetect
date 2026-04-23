import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_PATH   = r"c:\Users\bkrou\SpineDetect\david\microns\ml_ready.csv"
OUTPUT_PATH = r"c:\Users\bkrou\SpineDetect\david\microns\hks_signatures.svg"

CLASSES    = ["soma", "shaft", "spine"]
COLORS     = {"soma": "#4fc3f7", "shaft": "#d4a017", "spine": "#e91e8c"}
LABELS     = {"soma": "Soma", "shaft": "Shaft", "spine": "Spine"}
TIMESCALES = np.logspace(0, 2, 32)   # 1 to 100 AU, log-spaced

ALPHA_BG = 0.30
ALPHA_FG = 0.65
LW_BG    = 0.4
LW_FG    = 0.8

df = pd.read_csv(DATA_PATH)
df = df[df["tag"].isin(CLASSES)].copy()

hks_cols = [f"hks_{i}" for i in range(32)]
values   = df[hks_cols].values
tags     = df["tag"].values

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

for ax, cls in zip(axes, CLASSES):
    mask    = tags == cls
    bg_vals = values[~mask]
    fg_vals = values[mask]

    for row in bg_vals:
        ax.plot(TIMESCALES, row, color="#cccccc", lw=LW_BG, alpha=ALPHA_BG)

    for row in fg_vals:
        ax.plot(TIMESCALES, row, color=COLORS[cls], lw=LW_FG, alpha=ALPHA_FG)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(LABELS[cls], color=COLORS[cls], fontsize=13, fontweight="bold", pad=6)
    ax.set_xlabel("Timescale (AU)", fontsize=9)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=8)

axes[0].set_ylabel("Scaled HKS (AU)", fontsize=9)

plt.tight_layout(w_pad=2.0)
plt.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
