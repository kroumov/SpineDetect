"""
Sample random 3D volumes from MICrONS: download EM image and segmentation blocks
in parallel, then produce binary masks for selected neurons.
"""

import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox
from skimage.measure import block_reduce
from skimage.transform import resize
from tqdm import tqdm

try:
    import config
except ImportError:
    raise ImportError("config.py not found in the current directory.")


# -----------------------------------------------------------------------------
# Block download and processing
# -----------------------------------------------------------------------------

def process_block(image_cv, seg_cv, block_origin_nm, neuron_ids, timestamp):
    """
    Download one block of EM + segmentation, downsample EM, and build a binary mask
    for the selected neuron IDs. Resolution and block shape come from config.
    """
    output_res = np.array([config.RES_NM_X, config.RES_NM_Y, config.RES_NM_Z])
    block_shape = np.array([config.BLOCK_PX_X, config.BLOCK_PX_Y, config.BLOCK_PX_Z])

    res_img = np.array(image_cv.scales[0]["resolution"])
    res_seg = np.array(seg_cv.scales[config.SEGMENTATION_MIP_LEVEL]["resolution"])

    block_size_nm = block_shape * output_res
    end_nm = block_origin_nm + block_size_nm

    # EM at MIP 0; segmentation at higher MIP for speed
    bbox_img = Bbox(block_origin_nm / res_img, end_nm / res_img)
    em_data = np.array(image_cv.download(bbox_img, mip=0)).squeeze()

    down_factor = (output_res / res_img).astype(int)
    em_downsampled = block_reduce(em_data, block_size=tuple(down_factor), func=np.mean)

    bbox_seg = Bbox(block_origin_nm / res_seg, end_nm / res_seg)
    seg_data = np.array(seg_cv.download(
        bbox_seg, mip=config.SEGMENTATION_MIP_LEVEL, timestamp=timestamp
    )).squeeze()

    # Binary mask for selected neurons, resized to match EM block shape
    neuron_mask = np.isin(seg_data, neuron_ids).astype(np.float32)
    mask_resized = resize(
        neuron_mask, tuple(block_shape), order=0, preserve_range=True, anti_aliasing=False
    )

    return em_downsampled.astype(np.float32), mask_resized.astype(np.uint8)


def download_block_worker(args):
    """Worker: skip if cached; otherwise download one grid block and save em/svx .npy."""
    save_dir, grid_idx, origin_nm, neuron_ids, timestamp = args
    gx, gy, gz = grid_idx

    em_path = Path(save_dir) / f"em_{gx}_{gy}_{gz}.npy"
    svx_path = Path(save_dir) / f"svx_{gx}_{gy}_{gz}.npy"
    if em_path.exists() and svx_path.exists():
        return 0

    # Convert grid index (may be negative) to physical nm offset from origin
    block_shape = np.array([config.BLOCK_PX_X, config.BLOCK_PX_Y, config.BLOCK_PX_Z])
    output_res = np.array([config.RES_NM_X, config.RES_NM_Y, config.RES_NM_Z])

    block_offset = np.array([
        gx + config.X_BLOCKS_NEG,
        gy + config.Y_BLOCKS_NEG,
        gz + config.Z_BLOCKS_NEG,
    ])
    block_origin_nm = origin_nm + block_offset * (block_shape * output_res)

    try:
        image_cv = CloudVolume(
            config.IMAGE_URL, use_https=True, fill_missing=True, progress=False
        )
        seg_cv = CloudVolume(
            config.SEGMENTATION_URL, use_https=True, fill_missing=True, progress=False
        )

        em, svx = process_block(image_cv, seg_cv, block_origin_nm, neuron_ids, timestamp)
        np.save(em_path, em)
        np.save(svx_path, svx)
        return 0
    except Exception as exc:
        print(f"Block {grid_idx} failed: {exc}")
        return 1


# -----------------------------------------------------------------------------
# ROI sampling
# -----------------------------------------------------------------------------

def build_neuron_whitelist(client, table_names):
    """Collect pt_root_id from multiple materialization tables into a set."""
    whitelist = set()
    for table in table_names:
        try:
            df = client.materialize.query_table(table)
            whitelist.update(df["pt_root_id"].unique())
        except Exception:
            continue
    return whitelist


def sample_roi_with_neurons(client, seg_cv, whitelist, timestamp):
    """
    Pick a random nucleus, compute an aligned ROI around it, and select a subset
    of neurons in that region for mask generation.
    """
    output_res = np.array([config.RES_NM_X, config.RES_NM_Y, config.RES_NM_Z])
    block_shape = np.array([config.BLOCK_PX_X, config.BLOCK_PX_Y, config.BLOCK_PX_Z])
    offset_nm = np.array([config.OFFSET_NM_X, config.OFFSET_NM_Y, config.OFFSET_NM_Z])

    total_blocks = np.array([
        config.X_BLOCKS_NEG + config.X_BLOCKS_POS,
        config.Y_BLOCKS_NEG + config.Y_BLOCKS_POS,
        config.Z_BLOCKS_NEG + config.Z_BLOCKS_POS,
    ])
    roi_size_nm = total_blocks * block_shape * output_res

    while True:
        nuc = client.materialize.query_table("nucleus_detection_v0").sample(1).iloc[0]
        # Minnie65 nucleus pt_position is in voxels at 4×4×40 nm
        base_res = np.array([4, 4, 40])
        center_nm = np.array(nuc["pt_position"]) * base_res + offset_nm

        # Align ROI origin to output resolution grid
        origin_nm = (
            np.floor((center_nm - roi_size_nm / 2) / output_res) * output_res
        )

        res_seg = np.array(seg_cv.scales[config.SEGMENTATION_MIP_LEVEL]["resolution"])
        bbox_preview = Bbox(origin_nm / res_seg, (origin_nm + roi_size_nm) / res_seg)

        try:
            preview = seg_cv.download(
                bbox_preview,
                mip=config.SEGMENTATION_MIP_LEVEL,
                timestamp=timestamp,
            )
            found_ids = [int(i) for i in np.unique(preview) if i in whitelist]
            if not found_ids:
                continue

            # Sample subset of neurons (sparse labeling, e.g. 2P-like)
            count = max(1, int(len(found_ids) * config.NEURON_SAMPLING_RATIO))
            selected = random.sample(found_ids, count)
            return origin_nm, selected, nuc
        except Exception:
            continue


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    client = CAVEclient(config.DATASET_NAME)
    client.version = config.MATERIALIZATION_VERSION
    timestamp = int(
        client.materialize.get_timestamp(version=config.MATERIALIZATION_VERSION).timestamp()
    )

    whitelist = build_neuron_whitelist(client, [
        "allen_column_mtypes_v2",
        "aibs_metamodel_celltypes_v661",
        "baylor_gnn_cell_type_fine_model_v2",
    ])
    print("Neuron whitelist built.")

    seg_cv = CloudVolume(
        config.SEGMENTATION_URL, use_https=True, fill_missing=True
    )
    origin_nm, neuron_ids, nuc = sample_roi_with_neurons(
        client, seg_cv, whitelist, timestamp
    )
    print(f"ROI centered at nucleus {nuc['pt_root_id']}, {len(neuron_ids)} neurons selected.")

    save_dir = Path(config.OUTPUT_ROOT_DIR) / f"neurons_{nuc['pt_root_id']}"
    save_dir.mkdir(parents=True, exist_ok=True)

    task_list = [
        (
            str(save_dir),
            (x, y, z),
            origin_nm,
            neuron_ids,
            timestamp,
        )
        for x, y, z in itertools.product(
            range(-config.X_BLOCKS_NEG, config.X_BLOCKS_POS),
            range(-config.Y_BLOCKS_NEG, config.Y_BLOCKS_POS),
            range(-config.Z_BLOCKS_NEG, config.Z_BLOCKS_POS),
        )
    ]

    print(f"Downloading {len(task_list)} blocks (parallel workers: {config.PARALLEL_JOBS}).")
    with ProcessPoolExecutor(max_workers=config.PARALLEL_JOBS) as pool:
        list(tqdm(pool.map(download_block_worker, task_list), total=len(task_list), desc="Blocks"))

    print(f"Output saved to {save_dir.absolute()}.")