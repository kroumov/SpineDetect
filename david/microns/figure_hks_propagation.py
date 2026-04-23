"""
Figure: Heat Kernel Signature propagation over time.

Shows a synthetic dendrite + spines with a heat source placed at one spine
head, visualized at 4 timescales (t = 1, 10, 20, 30 AU).

Run with any Python environment that has numpy, scipy, matplotlib.
Output: figure_hks_propagation.png
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mesh_utils import (
    build_dendrite_mask, SOURCE_SPINE_IDX,
    IMG_H, IMG_W
)

OUT_PATH = os.path.join(os.path.dirname(__file__), "figure_hks_propagation.png")

# ── timescale panels ──────────────────────────────────────────────────────────
TIME_LABELS = ["t = 1 (AU)", "t = 10 (AU)", "t = 20 (AU)", "t = 30 (AU)"]
# Gaussian sigma (pixels) matching each AU; sigma = sqrt(2*D*t) heuristic
# scaled so t=30 fills most of the dendrite length.
SIGMAS      = [6, 32, 58, 95]

# ── colormaps ─────────────────────────────────────────────────────────────────
# Heat: white → deep red  (matches reference paper style)
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "heat_cmap",
    [(1.0, 1.0, 1.0),   # white  – no heat
     (1.0, 0.65, 0.0),  # orange – medium heat
     (0.85, 0.0, 0.0)], # deep red – max heat
    N=512,
)

# ── build geometry ────────────────────────────────────────────────────────────
mask, spine_centers = build_dendrite_mask()
src_cx, src_cy = spine_centers[SOURCE_SPINE_IDX]

# ── render ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 4,
    figsize=(16, 4.2),
    facecolor="white",
    gridspec_kw={"wspace": 0.04},
)

for ax, label, sigma in zip(axes, TIME_LABELS, SIGMAS):
    # Point source at the chosen spine head
    source = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    source[int(src_cy), int(src_cx)] = 1.0

    # Free-space Gaussian diffusion
    diffused = gaussian_filter(source, sigma=sigma)

    # Mask to dendrite interior; renormalize so contrast is full per panel
    heat = diffused * mask
    heat_max = heat.max()
    if heat_max > 0:
        heat = heat / heat_max

    # ── background: dendrite outline in very pale gray ────────────────────
    bg = np.ones((IMG_H, IMG_W, 3), dtype=np.float32)   # white canvas
    bg[mask > 0] = [0.92, 0.92, 0.92]                   # dendrite = light gray

    ax.imshow(bg, interpolation="bilinear")

    # ── heat overlay ──────────────────────────────────────────────────────
    heat_rgb  = HEAT_CMAP(heat)[:, :, :3]   # (H, W, 3) float in [0,1]
    heat_alpha = (heat * mask)[..., np.newaxis]    # alpha = heat × inside dendrite
    # alpha-composite heat onto background
    composite = bg * (1 - heat_alpha) + heat_rgb * heat_alpha
    composite = np.clip(composite, 0, 1)
    ax.imshow(composite, interpolation="bilinear")

    # ── panel label ───────────────────────────────────────────────────────
    ax.set_title(label, fontsize=13, pad=6, color="#222222")
    ax.axis("off")

    # ── mark source point on t=1 panel ───────────────────────────────────
    if label == TIME_LABELS[0]:
        ax.annotate(
            "Source",
            xy=(src_cx, src_cy),
            xytext=(src_cx + 60, src_cy - 55),
            fontsize=9,
            color="#111111",
            arrowprops=dict(arrowstyle="->", color="#111111", lw=1.0),
        )

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PATH}")
