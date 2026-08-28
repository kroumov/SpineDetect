"""Resume mask generation for a volume whose pipeline died mid-HKS.

Unlike mask_generation_pipeline (which holds all HKS results in memory and
writes the mask TIFF only at the very end), this driver runs each neuron in
its own subprocess and checkpoints its per-neuron mask to
<vol_dir>/partial_masks/mask_<root_id>.npz. A crash — including a native
crash inside VTK/meshmash — costs only the neuron being processed; rerunning
skips completed checkpoints.

Usage:
  python resume_mask_generation.py <vol_dir>                 # driver
  python resume_mask_generation.py <vol_dir> --child <id>    # internal
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# Workaround for cloudfiles IntervalTree null-interval bug (start_us==end_us on
# fast requests) — same patch as sample_random_volume.py.
import cloudfiles.monitoring as _cfmon
def _patched_end_io(self, flight_id, num_bytes):
    import time as _t
    end_us = int(_t.monotonic() * 1e6)
    with self._lock:
        start_us = int(self._in_flight.pop(flight_id) * 1e6)
        self._in_flight_bytes -= num_bytes
        end_us = max(end_us, start_us + 1)
        self._intervaltree.addi(start_us, end_us, [flight_id, num_bytes])
        self._total_bytes_landed += num_bytes
_cfmon.TransmissionMonitor.end_io = _patched_end_io
del _cfmon, _patched_end_io

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SCRIPT_DIR))

import config
from hks.mask_generation import (
    DOWNLOAD_BASE,
    LOCAL_DIR as HKS_DIR,
    generate_submesh,
    generate_tags,
    get_hks,
    get_mesh,
    mesh2mask,
    save_masks_as_tiff,
)

# rf_ensemble.pkl predicts string labels; fixed channel order (matches sorted
# classes_, which is what the original generate_masks produced when all three
# classes were present in a neuron).
CLASSES = ["shaft", "soma", "spine"]
N_CLASSES = len(CLASSES)

# Sampled neuron root_ids for microns_864691135382932459, recovered from
# pipelines/local/logs/download/download_20260827_153820.log
# ("Batch fetching meshes for: [...]"). download_info.txt does not store them.
ROOT_IDS = [
    864691135594253611,
    864691135584763000,
    864691136951623903,
    864691135772482379,
    864691135614982859,
    864691135988766979,
    864691136310480730,
    864691135939322677,
    864691135837788563,
    864691136194890472,
    864691136011905187,
    864691135463541310,
    864691135584185720,
    864691135726196543,
    864691135475636288,
    864691135273176337,
    864691136144458804,
    864691135837176979,
]


def parse_info(vol_dir):
    txt = (vol_dir / "download_info.txt").read_text(encoding="utf-8")
    origin = [float(x) for x in re.search(r"Origin \(nm\):\s*\[([^\]]+)\]", txt).group(1).split(",")]
    size_um = [float(x) for x in re.search(r"Physical Size:\s*\[([^\]]+)\]", txt).group(1).split(",")]
    res = [float(x) for x in re.search(r"Voxel Resolution:\s*\[([^\]]+)\]", txt).group(1).split(",")]
    bbox_min = np.array(origin)
    bbox_max = bbox_min + np.array(size_um) * 1000.0
    return bbox_min, bbox_max, np.array(res)


def child(vol_dir, root_id):
    from caveclient import CAVEclient

    bbox_min, bbox_max, res_seg = parse_info(vol_dir)
    pitch = float(np.min(res_seg))
    extent = bbox_max - bbox_min
    grid_size = np.ceil(extent / pitch).astype(int)

    out_dir = vol_dir / "partial_masks"
    out_dir.mkdir(exist_ok=True)

    # Feature-level checkpoint: mesh fetch + HKS + tags are the expensive part.
    # Persist them before voxelization so a mask-stage bug never costs HKS again.
    feat_path = out_dir / f"features_{root_id}.npz"
    if feat_path.exists():
        z = np.load(feat_path)
        verts, faces, tag_pred = z["verts"], z["faces"], z["tags"]
        print(f"Loaded feature checkpoint for {root_id}", flush=True)
    else:
        client = CAVEclient(config.DATASET_NAME)
        mesh_dict = get_mesh([root_id], client)
        if root_id not in mesh_dict:
            print(f"ERROR: mesh fetch failed for {root_id}", flush=True)
            sys.exit(3)
        hks_results = get_hks([root_id], mesh_dict, client)
        if root_id not in hks_results:
            print(f"ERROR: HKS failed for {root_id}", flush=True)
            sys.exit(3)
        hks_features = hks_results[root_id]["hks_result"].simple_features
        tag_pred = generate_tags(hks_features, wd=str(HKS_DIR))
        verts, faces = hks_results[root_id]["hks_result"].simple_mesh
        np.savez_compressed(
            feat_path,
            verts=np.asarray(verts, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int64),
            tags=np.asarray(tag_pred, dtype="U8"),
            hks=np.asarray(hks_features, dtype=np.float32),
        )
        print(f"FEATURES saved for {root_id}", flush=True)

    # Fixed tag->channel mapping (generate_masks packs only the tags present,
    # which would misalign channels across neurons when a class is absent).
    masks = np.zeros((*(grid_size + 1), N_CLASSES), dtype=np.uint8)
    in_bbox = ((verts >= bbox_min) & (verts <= bbox_max)).all(axis=1)
    for t, cls in enumerate(CLASSES):
        vertex_mask = (np.asarray(tag_pred) == cls) & in_bbox
        print(f"  class '{cls}': {int(vertex_mask.sum())} vertices in bbox", flush=True)
        if vertex_mask.sum() < 3:
            continue
        submesh = generate_submesh(vertex_mask, verts, faces)
        if len(submesh.faces) == 0:
            continue
        filled = mesh2mask(submesh, bbox_min, pitch, grid_size)
        if filled is not None:
            masks[:, :, :, t] = filled

    if masks.sum() == 0:
        print(f"ERROR: all-empty mask for {root_id} — refusing to checkpoint", flush=True)
        sys.exit(4)
    np.savez_compressed(
        out_dir / f"mask_{root_id}.npz",
        packed=np.packbits(masks, axis=0),
        shape=np.array(masks.shape),
    )
    print(f"CHECKPOINT saved for {root_id}", flush=True)


def driver(vol_dir):
    identifier = vol_dir.name.replace("microns_", "")
    out_dir = vol_dir / "partial_masks"
    out_dir.mkdir(exist_ok=True)

    failed = []
    for rid in ROOT_IDS:
        if (out_dir / f"mask_{rid}.npz").exists():
            print(f"[skip] {rid} (checkpoint exists)", flush=True)
            continue
        print(f"[run ] {rid}", flush=True)
        ret = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(vol_dir), "--child", str(rid)]
        ).returncode
        if ret != 0:
            print(f"[FAIL] {rid} exited {ret}", flush=True)
            failed.append(rid)

    done = [rid for rid in ROOT_IDS if (out_dir / f"mask_{rid}.npz").exists()]
    print(f"Per-neuron masks: {len(done)}/{len(ROOT_IDS)} complete; failed: {failed}", flush=True)
    if not done:
        sys.exit(1)

    bbox_min, bbox_max, res_seg = parse_info(vol_dir)
    acc = None
    for rid in done:
        d = np.load(out_dir / f"mask_{rid}.npz")
        m = np.unpackbits(d["packed"], axis=0, count=int(d["shape"][0])).astype(np.uint8)
        acc = m if acc is None else (acc | m)

    from skimage.transform import resize

    target_shape = ((bbox_max - bbox_min) / res_seg).astype(int)
    resized = np.zeros((*target_shape, acc.shape[3]), dtype=np.uint8)
    for ch in range(acc.shape[3]):
        resized[:, :, :, ch] = resize(
            acc[:, :, :, ch], target_shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

    save_masks_as_tiff(resized, identifier, output_dir=DOWNLOAD_BASE)
    print(f"MASKS TIFF written: {vol_dir / f'masks_{identifier}.tiff'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    vol = Path(sys.argv[1])
    if not vol.is_dir():
        print(f"ERROR: {vol} is not a directory")
        sys.exit(2)
    if "--child" in sys.argv:
        child(vol, int(sys.argv[sys.argv.index("--child") + 1]))
    else:
        driver(vol)
