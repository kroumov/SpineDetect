"""
Z-axis downsampling for all microns volumes in tpm_simulator/data/download.

Two branches:
  1. Max pooling -> tpm_simulator/data/train/em/microns_* (512×512×256, no Z crop; chunk.py crops Z to 192)
  2. Gaussian average -> tpm_simulator/data/downsample/microns_*

Input:  512×512×1024 (XY=512, Z=1024)
Output: em 512×512×256; downsample 512×512×256

Logs: local/logs/downsample/ or cloud/logs/downsample/ (when CLOUD_LOGS_DIR set)
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile as tf
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "download"
EM_OUTPUT_DIR = PROJECT_ROOT / "data" / "train" / "em"
DOWNSAMPLE_OUTPUT_DIR = PROJECT_ROOT / "data" / "downsample"
LOGS_DIR = LOCAL_DIR / "logs" / "downsample"

Z_FACTOR = 4  # 1024 -> 256
BLOCK_PX = 128
SVX_PATTERN = re.compile(r"^svx_(\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.npy$")
VAS_PATTERN = re.compile(r"^vas_(\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.npy$")
METADATA_FILENAME = "download_info.txt"

CHANNELS = [
    ("neurons", "neurons_*.tiff", "svx_*.npy", SVX_PATTERN),
    ("vessels", "vessels_*.tiff", "vas_*.npy", VAS_PATTERN),
    ("masks", "masks_*.tiff", None, None),
]


def _parse_resolution(input_path: Path):
    """Read resolution from download_info.txt."""
    info_path = input_path / METADATA_FILENAME
    res_nm = [128.0, 128.0, 128.0]
    if info_path.exists():
        content = info_path.read_text()
        match = re.search(r"Voxel Resolution:\s*\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\s*\]", content)
        if match:
            res_nm = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return res_nm


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] float32."""
    data = data.astype(np.float32)
    if data.max() > 1:
        data = data / 255.0
    return data


def load_channel_from_tiff(vol_path: Path, pattern: str):
    """Load volume from TIFF. Returns (Z, Y, X) float32 or None."""
    tiffs = list(vol_path.glob(pattern))
    if not tiffs:
        return None
    data = tf.imread(tiffs[0])
    return _to_float32(data)


def load_channel_from_blocks(vol_path: Path, block_pattern: str, filename_pattern: re.Pattern):
    """Stitch blocks into volume. Returns (Z, Y, X) float32 or None."""
    blocks = []
    for f in vol_path.glob(block_pattern):
        m = filename_pattern.match(f.name)
        if m:
            blocks.append((int(m.group(2)), int(m.group(3)), int(m.group(4)), f))

    if not blocks:
        return None

    gx = max(b[0] for b in blocks) + 1
    gy = max(b[1] for b in blocks) + 1
    gz = max(b[2] for b in blocks) + 1
    shape = (gx * BLOCK_PX, gy * BLOCK_PX, gz * BLOCK_PX)

    acc = np.zeros(shape, dtype=np.float32)
    for bi, bj, bk, path in tqdm(blocks, desc="Stitching blocks"):
        data = np.load(path).astype(np.float32)
        if data.shape != (BLOCK_PX, BLOCK_PX, BLOCK_PX):
            from skimage.transform import resize
            data = resize(data, (BLOCK_PX, BLOCK_PX, BLOCK_PX), order=0, preserve_range=True)
        s = (bi * BLOCK_PX, bj * BLOCK_PX, bk * BLOCK_PX)
        e = (s[0] + BLOCK_PX, s[1] + BLOCK_PX, s[2] + BLOCK_PX)
        acc[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = data

    return acc.transpose(2, 1, 0)


def downsample_z_maxpool(vol: np.ndarray, factor: int) -> np.ndarray:
    """Max over factor layers along Z."""
    nz = vol.shape[0]
    nz_out = nz // factor
    out = np.zeros((nz_out, vol.shape[1], vol.shape[2]), dtype=np.float32)
    for i in range(nz_out):
        out[i] = vol[i * factor:(i + 1) * factor].max(axis=0)
    return out


def downsample_z_gaussian_avg(vol: np.ndarray, factor: int, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing along Z, then average over factor layers."""
    smoothed = gaussian_filter(vol, sigma=(0, 0, sigma), mode="reflect")
    nz = smoothed.shape[0]
    nz_out = nz // factor
    out = np.zeros((nz_out, smoothed.shape[1], smoothed.shape[2]), dtype=np.float32)
    for i in range(nz_out):
        out[i] = smoothed[i * factor:(i + 1) * factor].mean(axis=0)
    return out


def _save_volume(out_folder: Path, chan_name: str, nid: str, out_vol: np.ndarray, res_nm_z: list):
    """Save downsampled volume as TIFF."""
    
    out_name = f"{chan_name}_{nid}.tiff"
    out_path = out_folder / out_name
    res_um = [r / 1000.0 for r in res_nm_z]
    resolution = (1.0 / res_um[1], 1.0 / res_um[0])
    if chan_name == "masks":
        tf.imwrite(out_path, out_vol, resolution=resolution, photometric='minisblack', metadata={"spacing": res_um[2], "unit": "um", "axes": "ZCYX"}, imagej=True)
    else:
        out_vol = (out_vol * 255).clip(0, 255).astype(np.uint8)
        tf.imwrite(
            out_path,
            out_vol,
            imagej=True,
            resolution=resolution,
            metadata={"spacing": res_um[2], "unit": "um"},
        )


def _read_download_info(vol_path: Path) -> str:
    """Read download_info.txt from source folder."""
    info_path = vol_path / METADATA_FILENAME
    if info_path.exists():
        return info_path.read_text()
    return ""


def _write_downsample_info(out_folder: Path, download_content: str, nid: str,
                           res_nm_orig, res_nm_new, shape_in, shape_out, method: str):
    """Write downsample_info.txt: download info + downsample section."""
    lines = []
    if download_content.strip():
        lines.append(download_content.rstrip())
        lines.append("")
    lines.append("--- Downsampling Info ---")
    lines.append(f"Method:           {method}")
    lines.append(f"Z Factor:         {Z_FACTOR}")
    lines.append(f"Input Shape:      {shape_in[0]}x{shape_in[1]}x{shape_in[2]} pixels")
    lines.append(f"Output Shape:     {shape_out[0]}x{shape_out[1]}x{shape_out[2]} pixels")
    lines.append(f"Input Resolution: [{res_nm_orig[0]}, {res_nm_orig[1]}, {res_nm_orig[2]}] nm")
    lines.append(f"Output Resolution:[{res_nm_new[0]}, {res_nm_new[1]}, {res_nm_new[2]}] nm")
    lines.append("========================================")

    info_name = "downsample_info.txt"
    (out_folder / info_name).write_text("\n".join(lines), encoding="utf-8")


def main():
    # Cloud: cloud/logs/downsample/; Local: local/logs/downsample/
    cloud_logs = os.environ.get("CLOUD_LOGS_DIR")
    _logs_base = Path(cloud_logs) / "downsample" if cloud_logs else LOGS_DIR
    _logs_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _logs_base / f"downsample_{ts}.log"

    logger = logging.getLogger("downsample")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    log = logger.info

    if not DOWNLOAD_DIR.is_dir():
        log(f"ERROR: {DOWNLOAD_DIR} not found")
        sys.exit(1)

    EM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOWNSAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    volumes = sorted(DOWNLOAD_DIR.glob("microns_*"))
    cloud_volume = os.environ.get("CLOUD_VOLUME")
    if cloud_volume:
        volumes = [v for v in volumes if v.name == cloud_volume]
    if not volumes:
        log(f"ERROR: No microns_* folders in {DOWNLOAD_DIR}")
        sys.exit(1)

    log(f"Log: {log_file}")
    log(f"Processing {len(volumes)} volumes: max-pool -> em/, gaussian-avg -> downsample/")

    for vol_path in tqdm(volumes, desc="Volumes"):
        nid = vol_path.name.replace("microns_", "")
        em_folder = EM_OUTPUT_DIR / vol_path.name
        downsample_folder = DOWNSAMPLE_OUTPUT_DIR / vol_path.name
        em_folder.mkdir(parents=True, exist_ok=True)
        downsample_folder.mkdir(parents=True, exist_ok=True)

        download_content = _read_download_info(vol_path)
        res_nm = _parse_resolution(vol_path)
        res_nm_z = list(res_nm)
        if len(res_nm_z) >= 3:
            res_nm_z[2] *= Z_FACTOR

        shape_in = None
        shape_out_em = None
        shape_out_ds = None

        for chan_name, tiff_glob, block_glob, block_pattern in CHANNELS:
            log(f"  Loading {chan_name}...")
            sys.stdout.flush()

            vol = load_channel_from_tiff(vol_path, tiff_glob)
            if vol is None and block_glob and block_pattern:
                vol = load_channel_from_blocks(vol_path, block_glob, block_pattern)
            if vol is None:
                continue
            log(f"  {chan_name} loaded: {vol.shape}")

            nz = vol.shape[0]
            if nz % Z_FACTOR != 0:
                logger.warning(f"SKIP {vol_path.name}/{chan_name}: Z={nz} not divisible by {Z_FACTOR}")
                continue

            # Downsample masks differently: maxpooling then make mutually exclusive
            if chan_name == "masks":  
                shape_in = (vol.shape[3], vol.shape[2], vol.shape[0], vol.shape[1]) # X, Y, Z, C
                nz_out = nz // Z_FACTOR 
                out_vol_masks = np.zeros((nz_out, vol.shape[1], vol.shape[2], vol.shape[3]), dtype=np.uint8)
                for ch in range(vol.shape[1]):
                    out_vol_masks[:, ch, :, :] = downsample_z_maxpool(vol[:, ch, :, :], Z_FACTOR)
                # Make mutually exclusive (priority: channel 0 > 1 > 2)
                for ch in range(1, vol.shape[1]):
                    out_vol_masks[:, ch, :, :] = np.where(out_vol_masks[:, :ch, :, :].max(axis=1) > 0, 0, out_vol_masks[:, ch, :, :])
                _save_volume(downsample_folder, chan_name, nid, out_vol_masks, res_nm_z)
                shape_out_masks = (out_vol_masks.shape[3], out_vol_masks.shape[2], out_vol_masks.shape[0], out_vol_masks.shape[1])
                log(f"  {chan_name} [downsample]: {shape_in[0]}x{shape_in[1]}x{shape_in[2]}x{shape_in[3]} -> {shape_out_masks[0]}x{shape_out_masks[1]}x{shape_out_masks[2]}x{shape_out_masks[3]}")
                continue

            shape_in = (vol.shape[1], vol.shape[2], vol.shape[0])  # X, Y, Z

            # Branch 1: Max pooling -> em (no Z crop; chunk.py crops Z to 192)
            out_vol_em = downsample_z_maxpool(vol, Z_FACTOR)
            shape_out_em = (out_vol_em.shape[1], out_vol_em.shape[2], out_vol_em.shape[0])
            log(f"  {chan_name} [em]: {shape_in[0]}x{shape_in[1]}x{shape_in[2]} -> {shape_out_em[0]}x{shape_out_em[1]}x{shape_out_em[2]}")
            _save_volume(em_folder, chan_name, nid, out_vol_em, res_nm_z)

            # Branch 2: Gaussian average -> degrade
            out_vol_degrade = downsample_z_gaussian_avg(vol, Z_FACTOR)
            shape_out_ds = (out_vol_degrade.shape[1], out_vol_degrade.shape[2], out_vol_degrade.shape[0])
            log(f"  {chan_name} [downsample]: {shape_in[0]}x{shape_in[1]}x{shape_in[2]} -> {shape_out_ds[0]}x{shape_out_ds[1]}x{shape_out_ds[2]}")
            _save_volume(downsample_folder, chan_name, nid, out_vol_degrade, res_nm_z)

        if shape_in is not None and shape_out_em is not None and shape_out_ds is not None:
            _write_downsample_info(
                em_folder, download_content, nid, res_nm, res_nm_z, shape_in, shape_out_em,
                method="max pooling (Z-axis)"
            )
            _write_downsample_info(
                downsample_folder, download_content, nid, res_nm, res_nm_z, shape_in, shape_out_ds,
                method="gaussian average (Z-axis)"
            )

    log(f"Done. EM: {EM_OUTPUT_DIR}, Downsample: {DOWNSAMPLE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()