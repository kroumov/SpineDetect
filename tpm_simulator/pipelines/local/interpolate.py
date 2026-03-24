"""
Bicubic interpolation along Z-axis for all folders under noise.
Input Z: 65 px -> Output Z: 192 px. XY unchanged.
Output: tpm_simulator/data/train/tpm/
interpolate_info.txt = noise_info content + interpolation section.
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile as tf
from skimage.transform import resize
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_DIR = Path(__file__).resolve().parent
# Cloud pipeline may set CLOUD_NOISE_DIR / CLOUD_OUTPUT_DIR (e.g. DATA_ROOT on scratch)
NOISE_DIR = Path(os.environ["CLOUD_NOISE_DIR"]) if os.environ.get("CLOUD_NOISE_DIR") else PROJECT_ROOT / "data" / "noise"
OUTPUT_DIR = Path(os.environ["CLOUD_OUTPUT_DIR"]) if os.environ.get("CLOUD_OUTPUT_DIR") else PROJECT_ROOT / "data" / "train" / "tpm"
LOGS_DIR = LOCAL_DIR / "logs" / "interpolate"

Z_IN = 65
Z_OUT = 192
MICRONS_PATTERN = re.compile(r"^microns_\d+_\d+$")
TIFF_PATTERN = re.compile(r"^neurons_.+\.tiff?$", re.I)


def main():
    parser = argparse.ArgumentParser(description="Bicubic Z-interpolation: 65 -> 192 px")
    parser.add_argument("--count", "-n", type=int, default=None, help="Limit number of folders")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"interpolate_{ts}.log"

    def log(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
        with open(log_file, "a", encoding="utf-8", errors="replace") as lf:
            lf.write(line)
        try:
            print(line.rstrip(), flush=True)
        except UnicodeEncodeError:
            print(line.rstrip().encode("ascii", errors="replace").decode("ascii"), flush=True)

    if not NOISE_DIR.is_dir():
        log(f"ERROR: noise dir not found: {NOISE_DIR}")
        sys.exit(1)

    folders = sorted(
        d for d in NOISE_DIR.iterdir()
        if d.is_dir() and MICRONS_PATTERN.match(d.name)
    )
    cloud_volume = os.environ.get("CLOUD_VOLUME")
    cloud_single = os.environ.get("CLOUD_SINGLE_FOLDER")  # for parallel: process one chunk only
    if cloud_single:
        folders = [d for d in folders if d.name == cloud_single]
    elif cloud_volume:
        prefix = cloud_volume.rstrip("_") + "_"
        folders = [d for d in folders if d.name == cloud_volume or d.name.startswith(prefix)]
    if not folders:
        log("No microns_*_* folders in noise dir.")
        sys.exit(0)

    if args.count is not None:
        folders = folders[: args.count]

    log(f"Interpolate pipeline started, log: {log_file}")
    log(f"Noise dir: {NOISE_DIR}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Folders to process: {len(folders)}" + (f" (count={args.count})" if args.count else ""))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok_count = 0

    for folder in tqdm(folders, desc="Interpolate"):
        tiffs = [f for f in folder.iterdir() if f.is_file() and TIFF_PATTERN.match(f.name)]
        if not tiffs:
            continue

        tiff_path = tiffs[0]
        noise_info_path = folder / "noise_info.txt"

        out_folder = OUTPUT_DIR / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        out_tiff_path = out_folder / tiff_path.name

        data = tf.imread(tiff_path)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        nz, ny, nx = data.shape
        if nz != Z_IN:
            log(f"SKIP {folder.name}: Z={nz} (expected {Z_IN})")
            continue

        data_f = data.astype(np.float32)
        if data_f.max() > 1:
            data_f = data_f / 255.0

        out_shape = (Z_OUT, ny, nx)
        resized = resize(data_f, out_shape, order=3, preserve_range=True, anti_aliasing=False)
        out_vol = (resized * 255).clip(0, 255).astype(np.uint8)

        res_nm = [64.0, 64.0, 320.0]
        if noise_info_path.exists():
            content = noise_info_path.read_text()
            match = re.search(r"Output Resolution:\s*\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\s*\]", content)
            if not match:
                match = re.search(r"Voxel Resolution:\s*\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\s*\]", content)
            if match:
                res_nm = [float(match.group(1)), float(match.group(2)), float(match.group(3))]

        res_um = [r / 1000.0 for r in res_nm]
        res_um[2] = res_um[2] * Z_IN / Z_OUT
        resolution = (1.0 / res_um[1], 1.0 / res_um[0])
        tf.imwrite(
            out_tiff_path,
            out_vol,
            imagej=True,
            resolution=resolution,
            metadata={"spacing": res_um[2], "unit": "um"},
        )

        noise_content = ""
        if noise_info_path.exists():
            noise_content = noise_info_path.read_text(encoding="utf-8").rstrip()

        interp_section = (
            "\n--- Interpolation ---\n"
            f"Method:           bicubic\n"
            f"Z Input:          {Z_IN} px\n"
            f"Z Output:         {Z_OUT} px\n"
            f"XY:               unchanged\n"
            "========================================\n"
        )
        (out_folder / "interpolate_info.txt").write_text(
            noise_content + interp_section, encoding="utf-8"
        )

        ok_count += 1

    log(f"Finished: {ok_count}/{len(folders)} succeeded")


if __name__ == "__main__":
    main()
