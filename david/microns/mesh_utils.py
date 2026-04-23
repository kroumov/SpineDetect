"""
Synthetic dendrite mesh utilities.

Creates a 2D raster image of a dendrite + spine geometry used as the
structural canvas for the HKS propagation figure. Also exports the
spine head center coordinates so the heat source can be placed precisely.

No external dependencies beyond numpy.
"""

import numpy as np


# ── image dimensions ─────────────────────────────────────────────────────────
IMG_H = 320
IMG_W = 900

# ── shaft geometry (pixel units) ──────────────────────────────────────────────
MID_Y        = IMG_H // 2
SHAFT_THICK  = 52          # total height of shaft in pixels
SHAFT_X0     = 60
SHAFT_X1     = IMG_W - 60

# ── spine geometry ────────────────────────────────────────────────────────────
N_SPINES         = 6
SPINE_NX         = [140, 255, 385, 515, 640, 760]   # x centers of spines
SPINE_DIRS       = [+1, -1, +1, -1, +1, -1]         # +1 = top, -1 = bottom
NECK_LEN         = 44     # pixels from shaft edge to head center
NECK_HALF_W      = 9      # half-width of neck in pixels
HEAD_RX          = 34     # spine head x radius
HEAD_RY          = 42     # spine head y radius

# ── source spine (0-indexed) for heat diffusion figure ────────────────────────
SOURCE_SPINE_IDX = 2       # middle spine, pointing up


def _fill_ellipse(img, cx, cy, rx, ry):
    """Fill an ellipse into img (in-place)."""
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
    img[mask] = 1


def build_dendrite_mask():
    """
    Return a (IMG_H, IMG_W) float32 array: 1 inside dendrite, 0 outside.
    Also returns a list of (cx, cy) spine head center coordinates.
    """
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    # ── shaft ─────────────────────────────────────────────────────────────────
    y0 = MID_Y - SHAFT_THICK // 2
    y1 = MID_Y + SHAFT_THICK // 2
    mask[y0:y1, SHAFT_X0:SHAFT_X1] = 1.0

    # ── round the shaft ends with semicircles ─────────────────────────────────
    _fill_ellipse(mask, SHAFT_X0, MID_Y, SHAFT_THICK // 2, SHAFT_THICK // 2)
    _fill_ellipse(mask, SHAFT_X1, MID_Y, SHAFT_THICK // 2, SHAFT_THICK // 2)

    # ── spines ────────────────────────────────────────────────────────────────
    spine_centers = []
    for sx, sd in zip(SPINE_NX, SPINE_DIRS):
        if sd == +1:
            shaft_edge = MID_Y - SHAFT_THICK // 2
            cy = shaft_edge - NECK_LEN
        else:
            shaft_edge = MID_Y + SHAFT_THICK // 2
            cy = shaft_edge + NECK_LEN

        # neck (thin rectangle)
        nx0 = sx - NECK_HALF_W
        nx1 = sx + NECK_HALF_W
        ny0, ny1 = min(shaft_edge, cy), max(shaft_edge, cy)
        mask[ny0:ny1, nx0:nx1] = 1.0

        # head (ellipse)
        _fill_ellipse(mask, sx, cy, HEAD_RX, HEAD_RY)
        spine_centers.append((sx, cy))

    return mask, spine_centers
