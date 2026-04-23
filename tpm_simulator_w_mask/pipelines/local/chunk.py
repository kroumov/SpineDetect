"""
Split neurons 512×512×256 into 128×128×256 chunks; extract corresponding vessel chunks from 1536×1536×384.
Each neuron chunk is at the center of its vessel chunk (vcpx size for NAOMi).

Coordinate system: 1536×1536×384 (vessels) and 512×512×256 (neurons) are CENTER-ALIGNED.
  - Neuron center (256, 256, 128) = Vessel center (768, 768, 192) = physical origin.
  - Transform: vessel_pixel = neuron_pixel + (512, 512, 64)

Input:  tpm_simulator/data/downsample/microns_*/
Output: tpm_simulator/data/chunk/microns_{nid}_{chunk_idx}/
        - neurons_{nid}_{chunk_idx}.tiff (128×128×256)
        - vessels_{nid}_{chunk_idx}.tiff (vcpx size)
        - chunk_info.txt (downsample_info + chunk section)

Also processes tpm_simulator/data/train/em/microns_*/ (512×512×256):
  - Crop Z 32 top/bottom -> 192, chunk 4×4 in XY -> 16× 128×128×192
  - Output: data/train/em/microns_{nid}_{chunk_idx}/ (neurons only, chunk_info.txt)
  - Delete original em/microns_{nid}/ after chunking

vcpx from compute_vcpx (tpm_config): [1003, 1003, 95], vcpx_z_full=287 for chunk Z=256

Logs: local/logs/chunk/ or cloud/logs/chunk/ (when CLOUD_LOGS_DIR set)
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile as tf
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_DIR = Path(__file__).resolve().parent
DOWNsample_DIR = PROJECT_ROOT / "data" / "downsample"
EM_DIR = PROJECT_ROOT / "data" / "train" / "em"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chunk"
LOGS_DIR = LOCAL_DIR / "logs" / "chunk"

CHUNK_SIZE_XY = 128
CHUNK_SIZE_Z = 256
EM_CHUNK_SIZE_Z = 192
EM_Z_CROP = 32  # crop 32 from top and bottom: 256 -> 192
# vcpx from MATLAB compute_vcpx (voxel_um=[0.064,0.064,0.32], slab_z=64, focal_spacing=3)
VCPX = (1003, 1003, 95)
VCPX_Z_FULL = 287  # (num_slabs-1)*focal_spacing + vcpx(3) for chunk Z=256

# Center alignment: neuron(512,512,256) center = vessel(1536,1536,384) center = physical origin
NEURON_CENTER = (256, 256, 128)
VESSEL_CENTER = (768, 768, 192)
# Offset: vessel_pixel = neuron_pixel + OFFSET
OFFSET_X = VESSEL_CENTER[0] - NEURON_CENTER[0]  # 512
OFFSET_Y = VESSEL_CENTER[1] - NEURON_CENTER[1]  # 512
OFFSET_Z = VESSEL_CENTER[2] - NEURON_CENTER[2]  # 64


def load_volume_tiff(path: Path) -> np.ndarray:
    """Load TIFF, return (X, Y, Z) uint8."""
    data = tf.imread(path)
    if data.ndim == 3:
        # tifffile returns (Z, Y, X) for stack
        data = np.transpose(data, (2, 1, 0))  # -> (X, Y, Z)
    elif data.ndim == 4:
        # load masks with shape (Z, C, Y, X)
        data = np.transpose(data, (3, 2, 0, 1))  # -> (X, Y, Z, C)
    return data


def save_volume_tiff(path: Path, vol: np.ndarray, res_nm: list):
    """Save to TIFF with resolution metadata."""
    res_um = [r / 1000.0 for r in res_nm]
    resolution = (1.0 / res_um[1], 1.0 / res_um[0])
    
    if vol.ndim == 3:
        # (X, Y, Z) -> (Z, Y, X)
        vol_tiff = np.transpose(vol, (2, 1, 0)).astype(np.uint8)
        
        tf.imwrite(
            path,
            vol_tiff,
            imagej=True,
            resolution=resolution,
            metadata={"spacing": res_um[2], "unit": "um"},
        )
    elif vol.ndim == 4:
        # (X, Y, Z, C) -> (Z, C, Y, X)
        vol_tiff = np.transpose(vol, (2, 3, 1, 0)).astype(np.uint8)
        tf.imwrite(
            path,
            vol_tiff,
            imagej=True,
            resolution=resolution,
            metadata={"spacing": res_um[2], "unit": "um", "axes": "ZCYX"},
        )


def neuron_center_to_vessel_center(cx: int, cy: int, cz: int) -> Tuple[int, int, int]:
    """Convert neuron pixel coords to vessel pixel coords (center-aligned)."""
    return (cx + OFFSET_X, cy + OFFSET_Y, cz + OFFSET_Z)


def extract_vessel_chunk(vessels: np.ndarray, vcx: int, vcy: int, vcz: int,
                        vcpx_x: int, vcpx_y: int, vcpx_z: int) -> np.ndarray:
    """
    Extract vcpx-sized vessel region centered at (vcx, vcy, vcz).
    (vcx, vcy, vcz) are in vessel pixel coordinates.
    """
    half_x = vcpx_x // 2
    half_y = vcpx_y // 2
    half_z = vcpx_z // 2

    x_lo = max(0, vcx - half_x)
    x_hi = min(vessels.shape[0], vcx + vcpx_x - half_x)
    y_lo = max(0, vcy - half_y)
    y_hi = min(vessels.shape[1], vcy + vcpx_y - half_y)
    z_lo = max(0, vcz - half_z)
    z_hi = min(vessels.shape[2], vcz + vcpx_z - half_z)

    out = np.zeros((vcpx_x, vcpx_y, vcpx_z), dtype=np.uint8)
    src = vessels[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi]
    dx = half_x - (vcx - x_lo)
    dy = half_y - (vcy - y_lo)
    dz = half_z - (vcz - z_lo)
    out[dx:dx + src.shape[0], dy:dy + src.shape[1], dz:dz + src.shape[2]] = src
    return out


def _parse_resolution(info_path: Path) -> list:
    """Read resolution from downsample_info.txt or chunk_info.txt."""
    res_nm = [64.0, 64.0, 320.0]
    if info_path.exists():
        content = info_path.read_text()
        match = re.search(r"Output Resolution:\s*\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\s*\]", content)
        if match:
            res_nm = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return res_nm


def process_downsample_folder(input_path: Path, log, log_debug, logger) -> bool:
    """Process one downsample folder. Returns True on success."""
    folder_name = input_path.name
    if not folder_name.startswith("microns_"):
        log(f"Skip (not microns_*): {folder_name}")
        return False

    nid = folder_name.replace("microns_", "")

    neurons_tiff = list(input_path.glob("neurons_*.tiff")) + list(input_path.glob("neurons_*.tif"))
    masks_tiff = list(input_path.glob("masks_*.tiff")) + list(input_path.glob("masks_*.tif"))
    vessels_tiff = list(input_path.glob("vessels_*.tiff")) + list(input_path.glob("vessels_*.tif"))
    if not neurons_tiff:
        log(f"Skip (no neurons TIFF): {input_path}")
        return False
    if not masks_tiff:
        log(f"Skip (no masks TIFF): {input_path}")
        return False
    if not vessels_tiff:
        log(f"Skip (no vessels TIFF): {input_path}")
        return False

    log(f"Processing: {folder_name}")
    log(f"Loading neurons from {neurons_tiff[0].name}...")
    neurons = load_volume_tiff(neurons_tiff[0])
    log(f"Loading masks from {masks_tiff[0].name}...")
    masks = load_volume_tiff(masks_tiff[0])
    log(f"Loading vessels from {vessels_tiff[0].name}...")
    vessels = load_volume_tiff(vessels_tiff[0])

    nx_neur, ny_neur, nz_neur = neurons.shape
    nv_x, nv_y, nv_z = vessels.shape
    if (nx_neur, ny_neur, nz_neur) != (512, 512, 256):
        logger.warning(f"Expected neurons 512×512×256, got {nx_neur}×{ny_neur}×{nz_neur}")

    if nv_x < nx_neur or nv_y < ny_neur or nv_z < nz_neur:
        log(f"ERROR: Vessels {vessels.shape} must cover neurons {neurons.shape}")
        return False

    log(f"Neurons: {nx_neur}×{ny_neur}×{nz_neur} | Vessels: {nv_x}×{nv_y}×{nv_z}")
    log_debug(f"Center alignment: neuron_center={NEURON_CENTER}, vessel_center={VESSEL_CENTER}, offset=({OFFSET_X},{OFFSET_Y},{OFFSET_Z})")

    downsample_info_path = input_path / "downsample_info.txt"
    downsample_content = downsample_info_path.read_text(encoding="utf-8") if downsample_info_path.exists() else ""
    res_nm = _parse_resolution(downsample_info_path)

    ncx = (nx_neur + CHUNK_SIZE_XY - 1) // CHUNK_SIZE_XY
    ncy = (ny_neur + CHUNK_SIZE_XY - 1) // CHUNK_SIZE_XY
    ncz = (nz_neur + CHUNK_SIZE_Z - 1) // CHUNK_SIZE_Z
    total_chunks = ncx * ncy * ncz

    log(f"Chunk grid: {ncx}×{ncy}×{ncz} = {total_chunks} chunks | vcpx: {VCPX[0]}×{VCPX[1]}×{VCPX_Z_FULL}")

    half_x, half_y, half_z = VCPX[0] // 2, VCPX[1] // 2, VCPX_Z_FULL // 2

    pbar = tqdm(total=total_chunks, desc=folder_name)
    for ck in range(ncz):
        for cj in range(ncy):
            for ci in range(ncx):
                chunk_idx = (ck * ncy + cj) * ncx + ci + 1
                out_folder = OUTPUT_DIR / f"microns_{nid}_{chunk_idx:04d}"
                out_folder.mkdir(parents=True, exist_ok=True)

                cx = ci * CHUNK_SIZE_XY + CHUNK_SIZE_XY // 2
                cy = cj * CHUNK_SIZE_XY + CHUNK_SIZE_XY // 2
                cz = ck * CHUNK_SIZE_Z + CHUNK_SIZE_Z // 2

                vcx, vcy, vcz = neuron_center_to_vessel_center(cx, cy, cz)

                x0, x1 = ci * CHUNK_SIZE_XY, min((ci + 1) * CHUNK_SIZE_XY, nx_neur)
                y0, y1 = cj * CHUNK_SIZE_XY, min((cj + 1) * CHUNK_SIZE_XY, ny_neur)
                z0, z1 = ck * CHUNK_SIZE_Z, min((ck + 1) * CHUNK_SIZE_Z, nz_neur)
                neur_chunk = neurons[x0:x1, y0:y1, z0:z1].copy()
                masks_chunk = masks[x0:x1, y0:y1, z0:z1, :].copy()
                chunk_shape = (CHUNK_SIZE_XY, CHUNK_SIZE_XY, CHUNK_SIZE_Z)
                if neur_chunk.shape != chunk_shape:
                    pad = np.zeros(chunk_shape, dtype=neur_chunk.dtype)
                    pad[: neur_chunk.shape[0], : neur_chunk.shape[1], : neur_chunk.shape[2]] = neur_chunk
                    neur_chunk = pad

                    pad = np.zeros((*chunk_shape, masks_chunk.shape[3]), dtype=masks_chunk.dtype)
                    pad[: masks_chunk.shape[0], : masks_chunk.shape[1], : masks_chunk.shape[2], :] = masks_chunk
                    masks_chunk = pad

                ves_chunk = extract_vessel_chunk(
                    vessels, vcx, vcy, vcz,
                    VCPX[0], VCPX[1], VCPX_Z_FULL,
                )

                x_lo = max(0, vcx - half_x)
                x_hi = min(nv_x, vcx + VCPX[0] - half_x)
                y_lo = max(0, vcy - half_y)
                y_hi = min(nv_y, vcy + VCPX[1] - half_y)
                z_lo = max(0, vcz - half_z)
                z_hi = min(nv_z, vcz + VCPX_Z_FULL - half_z)
                pad_x = (x_lo == 0 and vcx - half_x < 0) or (x_hi == nv_x and vcx + VCPX[0] - half_x > nv_x)
                pad_y = (y_lo == 0 and vcy - half_y < 0) or (y_hi == nv_y and vcy + VCPX[1] - half_y > nv_y)
                pad_z = (z_lo == 0 and vcz - half_z < 0) or (z_hi == nv_z and vcz + VCPX_Z_FULL - half_z > nv_z)
                log_debug(
                    f"Chunk {chunk_idx} (ci={ci},cj={cj},ck={ck}): "
                    f"neuron_center=({cx},{cy},{cz}) -> vessel_center=({vcx},{vcy},{vcz}) | "
                    f"vessel_extract x[{x_lo}:{x_hi}] y[{y_lo}:{y_hi}] z[{z_lo}:{z_hi}] | "
                    f"padding=({pad_x},{pad_y},{pad_z})"
                )

                save_volume_tiff(out_folder / f"neurons_{nid}_{chunk_idx:04d}.tiff", neur_chunk, res_nm)
                save_volume_tiff(out_folder / f"masks_{nid}_{chunk_idx:04d}.tiff", masks_chunk, res_nm)
                save_volume_tiff(out_folder / f"vessels_{nid}_{chunk_idx:04d}.tiff", ves_chunk, res_nm)

                chunk_section = [
                    "",
                    "--- Chunk Info ---",
                    f"Chunk Index:      {chunk_idx}/{total_chunks} (ci={ci}, cj={cj}, ck={ck})",
                    f"Chunk Size:       {CHUNK_SIZE_XY}×{CHUNK_SIZE_XY}×{CHUNK_SIZE_Z} px",
                    f"Chunk Grid:       {ncx}x{ncy}x{ncz} = {total_chunks} chunks",
                    f"Vessel Chunk:     {VCPX[0]}x{VCPX[1]}x{VCPX_Z_FULL} px (vcpx for NAOMi)",
                    f"Neuron Input:     {nx_neur}x{ny_neur}x{nz_neur} px",
                    f"Vessel Input:     {nv_x}x{nv_y}x{nv_z} px",
                    f"Center Align:     neuron_center=({cx},{cy},{cz}) -> vessel_center=({vcx},{vcy},{vcz})",
                    f"Vessel Extract:   x[{x_lo}:{x_hi}] y[{y_lo}:{y_hi}] z[{z_lo}:{z_hi}]",
                    "========================================",
                ]
                chunk_info_path = out_folder / "chunk_info.txt"
                content = downsample_content.rstrip() + "\n" + "\n".join(chunk_section)
                chunk_info_path.write_text(content, encoding="utf-8")

                pbar.update(1)

    pbar.close()
    log(f"Done: {folder_name} -> {total_chunks} chunks")
    return True


def process_em_folder(input_path: Path, log, log_debug, logger) -> bool:
    """Process one em folder: crop Z to 192, chunk 4×4, save to em/microns_{nid}_{chunk_idx}/, delete original."""
    folder_name = input_path.name
    if not folder_name.startswith("microns_"):
        log(f"Skip (not microns_*): {folder_name}")
        return False

    nid = folder_name.replace("microns_", "")

    neurons_tiff = list(input_path.glob("neurons_*.tiff")) + list(input_path.glob("neurons_*.tif"))
    if not neurons_tiff:
        log(f"Skip (no neurons TIFF): {input_path}")
        return False

    log(f"Processing EM: {folder_name}")
    neurons = load_volume_tiff(neurons_tiff[0])

    nx, ny, nz = neurons.shape
    if (nx, ny, nz) != (512, 512, 256):
        logger.warning(f"Expected em 512×512×256, got {nx}×{ny}×{nz}")

    # Crop Z: 32 from top and bottom -> 192
    z0, z1 = EM_Z_CROP, nz - EM_Z_CROP
    neurons = neurons[:, :, z0:z1]
    nz_crop = neurons.shape[2]
    log(f"EM cropped Z: 256 -> {nz_crop}")

    downsample_info_path = input_path / "downsample_info.txt"
    downsample_content = downsample_info_path.read_text(encoding="utf-8") if downsample_info_path.exists() else ""
    res_nm = _parse_resolution(downsample_info_path)

    ncx = (nx + CHUNK_SIZE_XY - 1) // CHUNK_SIZE_XY
    ncy = (ny + CHUNK_SIZE_XY - 1) // CHUNK_SIZE_XY
    ncz = (nz_crop + EM_CHUNK_SIZE_Z - 1) // EM_CHUNK_SIZE_Z
    total_chunks = ncx * ncy * ncz

    log(f"EM chunk grid: {ncx}×{ncy}×{ncz} = {total_chunks} chunks (128×128×{EM_CHUNK_SIZE_Z})")

    pbar = tqdm(total=total_chunks, desc=f"EM {folder_name}")
    for ck in range(ncz):
        for cj in range(ncy):
            for ci in range(ncx):
                chunk_idx = (ck * ncy + cj) * ncx + ci + 1
                out_folder = EM_DIR / f"microns_{nid}_{chunk_idx:04d}"
                out_folder.mkdir(parents=True, exist_ok=True)

                x0, x1 = ci * CHUNK_SIZE_XY, min((ci + 1) * CHUNK_SIZE_XY, nx)
                y0, y1 = cj * CHUNK_SIZE_XY, min((cj + 1) * CHUNK_SIZE_XY, ny)
                z0_ch, z1_ch = ck * EM_CHUNK_SIZE_Z, min((ck + 1) * EM_CHUNK_SIZE_Z, nz_crop)

                neur_chunk = neurons[x0:x1, y0:y1, z0_ch:z1_ch].copy()

                chunk_shape = (CHUNK_SIZE_XY, CHUNK_SIZE_XY, EM_CHUNK_SIZE_Z)
                if neur_chunk.shape != chunk_shape:
                    pad_n = np.zeros(chunk_shape, dtype=neur_chunk.dtype)
                    pad_n[: neur_chunk.shape[0], : neur_chunk.shape[1], : neur_chunk.shape[2]] = neur_chunk
                    neur_chunk = pad_n

                save_volume_tiff(out_folder / f"neurons_{nid}_{chunk_idx:04d}.tiff", neur_chunk, res_nm)

                cx = ci * CHUNK_SIZE_XY + CHUNK_SIZE_XY // 2
                cy = cj * CHUNK_SIZE_XY + CHUNK_SIZE_XY // 2
                cz = ck * EM_CHUNK_SIZE_Z + EM_CHUNK_SIZE_Z // 2
                chunk_section = [
                    "",
                    "--- Chunk Info ---",
                    f"Chunk Index:      {chunk_idx}/{total_chunks} (ci={ci}, cj={cj}, ck={ck})",
                    f"Chunk Size:       {CHUNK_SIZE_XY}×{CHUNK_SIZE_XY}×{EM_CHUNK_SIZE_Z} px (EM)",
                    f"Chunk Grid:       {ncx}x{ncy}x{ncz} = {total_chunks} chunks",
                    f"EM Input:         512x512x256 (cropped Z to {nz_crop})",
                    f"Neuron Extract:   x[{x0}:{x1}] y[{y0}:{y1}] z[{z0_ch}:{z1_ch}]",
                    "========================================",
                ]
                chunk_info_path = out_folder / "chunk_info.txt"
                content = downsample_content.rstrip() + "\n" + "\n".join(chunk_section)
                chunk_info_path.write_text(content, encoding="utf-8")

                pbar.update(1)

    pbar.close()

    # Delete original em/microns_{nid}/ folder and its contents
    for f in input_path.iterdir():
        f.unlink()
    input_path.rmdir()
    log(f"Done EM: {folder_name} -> {total_chunks} chunks, removed original folder")
    return True


def main():
    # Cloud: cloud/logs/chunk/; Local: local/logs/chunk/
    cloud_logs = os.environ.get("CLOUD_LOGS_DIR")
    _logs_base = Path(cloud_logs) / "chunk" if cloud_logs else LOGS_DIR
    _logs_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _logs_base / f"chunk_{ts}.log"

    logger = logging.getLogger("chunk")
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
    log_debug = logger.debug

    if not DOWNsample_DIR.is_dir():
        print(f"ERROR: downsample dir not found: {DOWNsample_DIR}")
        sys.exit(1)

    ds_folders = sorted(d for d in DOWNsample_DIR.iterdir() if d.is_dir() and d.name.startswith("microns_"))
    em_folders = []
    if EM_DIR.is_dir():
        em_folders = sorted(
            d for d in EM_DIR.iterdir()
            if d.is_dir() and d.name.startswith("microns_") and not re.match(r"^microns_\d+_\d+_\d{4}$", d.name)
        )
    cloud_volume = os.environ.get("CLOUD_VOLUME")
    if cloud_volume:
        ds_folders = [d for d in ds_folders if d.name == cloud_volume]
        em_folders = [d for d in em_folders if d.name == cloud_volume]

    if not ds_folders and not em_folders:
        log("No microns_* folders in downsample or em dir.")
        sys.exit(0)

    log(f"Log: {log_file}")
    log(f"Downsample: {DOWNsample_DIR} ({len(ds_folders)} folders)")
    log(f"EM: {EM_DIR} ({len(em_folders)} folders)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EM_DIR.mkdir(parents=True, exist_ok=True)

    ok_ds = 0
    for i, input_path in enumerate(ds_folders, 1):
        log(f"--- [Downsample {i}/{len(ds_folders)}] {input_path.name} ---")
        if process_downsample_folder(input_path, log, log_debug, logger):
            ok_ds += 1

    ok_em = 0
    for i, input_path in enumerate(em_folders, 1):
        log(f"--- [EM {i}/{len(em_folders)}] {input_path.name} ---")
        if process_em_folder(input_path, log, log_debug, logger):
            ok_em += 1

    log(f"Finished: downsample {ok_ds}/{len(ds_folders)}, em {ok_em}/{len(em_folders)}")


if __name__ == "__main__":
    main()