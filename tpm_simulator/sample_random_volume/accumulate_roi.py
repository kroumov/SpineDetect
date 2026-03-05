import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile as tf
from skimage.transform import resize
from tqdm import tqdm

try:
    import config as _config
except ImportError:
    _config = None

# Block files: channel_nucleusId_i_j_k.npy
FILENAME_PATTERN = r"^(em|svx|vas|seg)_(\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.npy$"
METADATA_FILENAME = "download_info.txt"

CHAN_NAMING = {
    "em": "em",
    "svx": "neurons",
    "vas": "vessels",
    "seg": "seg",
}

def _parse_metadata(input_path):
    """Read resolution from metadata file if present."""
    info_path = input_path / METADATA_FILENAME
    res_nm = [128.0, 128.0, 128.0]
    
    if info_path.exists():
        content = info_path.read_text()
        match = re.search(r"Voxel Resolution:\s*\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\s*\]", content)
        if match:
            res_nm = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return res_nm

def accumulate_roi(input_path):
    """Stitch block .npy files into volume TIFFs; writes one TIFF per channel, then removes blocks."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory.")
        return

    files = list(input_path.glob("*.npy"))
    if not files:
        print("No .npy files found.")
        return

    db = {}
    nucleus_id = "unknown"
    for f in files:
        match = re.match(FILENAME_PATTERN, f.name)
        if match:
            chan, n_id, i, j, k = match.groups()
            nucleus_id = n_id
            i, j, k = int(i), int(j), int(k)
            if chan not in db: db[chan] = []
            db[chan].append((i, j, k, f))

    if not db:
        print("No matching blocks found.")
        return

    # Resolution and grid from metadata or config
    res_nm = _parse_metadata(input_path)
    block_px = getattr(_config, "BLOCK_PX", 128)
    stitch_workers = getattr(_config, "STITCH_WORKERS", 4)

    print(f"Nucleus: {nucleus_id} | Resolution: {res_nm} nm")

    # Stitch each channel with its own grid (em/svx may be 4x4x8, vas may be 12x12x12)
    for chan, blocks in db.items():
        coords = [item[:3] for item in blocks]
        gx = max(c[0] for c in coords) + 1
        gy = max(c[1] for c in coords) + 1
        gz = max(c[2] for c in coords) + 1
        vol_shape = (gx * block_px, gy * block_px, gz * block_px)
        print(f"  {chan}: grid {gx}x{gy}x{gz} -> {vol_shape[0]}x{vol_shape[1]}x{vol_shape[2]} px")

        acc = np.zeros(vol_shape, dtype=np.float32)

        def load_one(item):
            bi, bj, bk, path = item
            data = np.load(path)
            if data.shape != (block_px, block_px, block_px):
                data = resize(data, (block_px, block_px, block_px), order=0, preserve_range=True)
            return bi, bj, bk, data

        with ThreadPoolExecutor(max_workers=stitch_workers) as pool:
            results = list(tqdm(pool.map(load_one, blocks), total=len(blocks), desc=f"Stitching {chan}"))

        for bi, bj, bk, data in results:
            s = (bi * block_px, bj * block_px, bk * block_px)
            e = (s[0] + block_px, s[1] + block_px, s[2] + block_px)
            acc[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = data

        output_prefix = CHAN_NAMING.get(chan, chan)
        out_path = input_path / f"{output_prefix}_{nucleus_id}.tiff"
        out_data = acc.transpose(2, 1, 0)
        del acc

        if chan in ["svx", "vas", "seg"]:
            out_data = (out_data > 0.5).astype(np.uint8) * 255
        else:
            out_data = out_data.astype(np.uint8)

        res_um = [r / 1000.0 for r in res_nm]
        resolution = (1.0 / res_um[1], 1.0 / res_um[0]) 
        
        tf.imwrite(
            out_path, out_data, imagej=True, resolution=resolution,
            metadata={'spacing': res_um[2], 'unit': 'um'}
        )
        print(f"Saved: {out_path}")

    # Remove block .npy after all channels written
    print("Cleaning up block files...")
    for _, blocks in db.items():
        for item in blocks:
            path = item[3]
            if path.exists():
                path.unlink()
    print("Cleanup complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python accumulate_roi.py <data_directory>")
    else:
        accumulate_roi(sys.argv[1])