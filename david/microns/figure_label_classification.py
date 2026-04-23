"""
Figure: Anatomical label classification on a dendritic segment.

Uses real annotated positions from ml_ready.csv for a single neuron,
then overlays predicted class labels from an RF classifier trained on
master_training_data.csv.

Color scheme matches the 2D segmentation mask image:
  shaft  → red    #cc2200
  spine  → green  #00cc44
  soma   → blue   #0044cc

Run with: numpy, scipy, pandas, scikit-learn, matplotlib.
Output: figure_label_classification.png
"""

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(__file__)
ML_CSV   = os.path.join(BASE, "ml_ready.csv")
TRAIN_CSV = os.path.join(BASE, "master_training_data.csv")
OUT_PATH  = os.path.join(BASE, "figure_label_classification.png")

# Best neuron for visualisation: 33 spine + 33 shaft + 7 soma
TARGET_NEURON = 864691135778700477

# ── colour scheme ─────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Shaft": "#cc2200",
    "Spine": "#00cc44",
    "Soma":  "#0044cc",
    "Neck":  "#ffdd00",
}
FALLBACK_COLOR = "#888888"

# ── load and train classifier ────────────────────────────────────────────────
print("Loading training data …")
train = pd.read_csv(TRAIN_CSV)
train = train[train["target_label"].isin(["Spine", "Shaft", "Soma"])]

hks_t_cols = [f"hks_t{i}" for i in range(32)]
X_train = train[hks_t_cols].values
le = LabelEncoder()
y_train = le.fit_transform(train["target_label"])

print(f"Training RF on {len(X_train)} samples, classes: {list(le.classes_)} …")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ── load ml_ready.csv for the target neuron ───────────────────────────────────
print("Loading annotated points …")
df = pd.read_csv(ML_CSV)
df = df[df["post_pt_root_id"] == TARGET_NEURON].copy()
df = df[df["tag"].isin(["spine", "shaft", "soma"])].reset_index(drop=True)

print(f"  {len(df)} annotated targets: {df['tag'].value_counts().to_dict()}")

# Parse 3D voxel positions (stored as "[x y z]" strings)
positions = np.vstack(
    df["post_pt_position"].apply(
        lambda s: np.fromstring(s.strip("[]"), sep=" ")
    ).values
)  # shape (N, 3); units: 4×4×40 nm voxels

# ── apply classifier: transform ml_ready HKS to log scale ────────────────────
hks_raw_cols = [f"hks_{i}" for i in range(32)]
X_raw = df[hks_raw_cols].values
X_log = np.log(np.clip(X_raw, 1e-30, None))   # natural-log matches training data scale

y_pred_idx = clf.predict(X_log)
y_pred_labels = le.inverse_transform(y_pred_idx)   # "Spine" / "Shaft" / "Soma"

# ── 2-D projection: use (x, z) since z spans ~80 µm (depth), x ~190 µm ──────
# Multiply by [4, 4, 40] to convert to nm
nm_per_vox = np.array([4.0, 4.0, 40.0])
pos_nm = positions * nm_per_vox          # (N, 3) in nm

# Center the coordinates
pos_nm -= pos_nm.mean(axis=0)

plot_x = pos_nm[:, 0]   # lateral axis
plot_y = pos_nm[:, 1]   # second lateral axis

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6),
    facecolor="#000000",
    gridspec_kw={"wspace": 0.06},
)

def scatter_panel(ax, labels_iter, title):
    ax.set_facecolor("#000000")

    # draw each class separately so colour order is predictable
    for cls_name, color in CLASS_COLORS.items():
        cls_mask = np.array([l == cls_name for l in labels_iter])
        if cls_mask.sum() == 0:
            continue
        ax.scatter(
            plot_x[cls_mask], plot_y[cls_mask],
            c=color,
            s=1200,           # large blobs to fill in the sparse point cloud
            alpha=0.70,
            edgecolors="none",
            linewidths=0,
            zorder=2,
        )

    ax.set_title(title, color="white", fontsize=13, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


# Panel 1: ground-truth labels (capitalise first letter to match CLASS_COLORS)
gt_labels = [t.capitalize() for t in df["tag"]]
scatter_panel(axes[0], gt_labels, "Ground Truth Labels")

# Panel 2: classifier predictions
scatter_panel(axes[1], y_pred_labels, "RF Classifier Predictions")

# ── legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()
    if k in ["Shaft", "Spine", "Soma"]
]
axes[1].legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=10,
    framealpha=0.3,
    facecolor="#222222",
    edgecolor="none",
    labelcolor="white",
)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="#000000")
print(f"Saved: {OUT_PATH}")
