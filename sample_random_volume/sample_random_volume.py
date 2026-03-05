"""
Sample random 3D volumes from MICrONS; writes blocks and metadata.
"""

import itertools
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox
from tqdm import tqdm

try:
    import config
except ImportError:
    raise ImportError("config.py not found in the current directory.")

INFO_FILENAME = "download_info.txt"

def _log(msg):
    print(msg, flush=True)

SEG_MIP_LEVEL = getattr(config, "MIP_LEVEL", 2)
BLOCK_PX = getattr(config, "BLOCK_PX", 128)
GRID_SIZE = getattr(config, "GRID_SIZE", 4)
DOWNLOAD_WORKERS = getattr(config, "DOWNLOAD_WORKERS", 8)
OUTPUT_ROOT = getattr(config, "OUTPUT_ROOT_DIR", ".")
N_RATIO = getattr(config, "NEURON_SAMPLING_RATIO", 0.05)
V_RATIO = getattr(config, "VESSEL_SAMPLING_RATIO", 0.5)

def _get_resolution_nm(cv, mip):
    res = cv.scales[mip].get("resolution")
    return np.array(res, dtype=float) if res else np.array([128, 128, 128])

def get_matching_mip(cv, target_res_nm):
    """Return MIP index whose resolution matches target (nm)."""
    for mip, scale in enumerate(cv.scales):
        if np.isclose(scale['resolution'][0], target_res_nm, atol=0.1):
            return int(mip)
    return 0

def write_detailed_metadata(save_dir, info_dict):
    """Write run metadata and sampling stats to save_dir."""
    lines = [
        "========================================",
        "      MICRONS DATA DOWNLOAD INFO",
        "========================================",
        f"Nucleus ID:       {info_dict['root_id']}",
        f"MIP Level (SEG):  {info_dict['seg_mip']}",
        f"MIP Level (EM):   {info_dict['em_mip']}",
        f"Voxel Resolution: {info_dict['res_nm']} nm",
        f"Volume Size:      {GRID_SIZE * BLOCK_PX}^3 pixels",
        f"Physical Size:    {info_dict['phys_size_um']} um",
        f"Voxel Origin:     {info_dict['voxel_origin']}",
        f"Origin (nm):      {info_dict['origin_nm']}",
        "",
        "--- Sampling Statistics ---",
        f"Neurons Found:    {info_dict['found_neuron_count']}",
        f"Neurons Sampled:  {info_dict['sampled_neuron_count']} (Ratio: {info_dict['n_ratio']})",
        f"Vessels Found:    {info_dict['found_vessel_count']}",
        f"Vessels Sampled:  {info_dict['sampled_vessel_count']} (Ratio: {info_dict['v_ratio']})",
        "",
        "--- Whitelists Table Info ---",
        "Neuron Tables Statistics:"
    ]
    for table, count in info_dict['neuron_tables'].items():
        lines.append(f"  - {table}: {count} IDs loaded")
    
    lines.append("Vessel Tables Statistics:")
    for table, count in info_dict['vessel_tables'].items():
        lines.append(f"  - {table}: {count} IDs loaded")
    
    lines.append("========================================")
    
    with open(Path(save_dir) / INFO_FILENAME, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --- single block download (EM + neuron/vessel masks from seg) ---
def download_one_block(args):
    save_dir, root_id, (i, j, k), origin_vox_seg, seg_mip, em_mip, neuron_ids, vessel_ids, timestamp = args
    start_vox = (origin_vox_seg[0] + i * BLOCK_PX, origin_vox_seg[1] + j * BLOCK_PX, origin_vox_seg[2] + k * BLOCK_PX)
    bbox_vox = Bbox(start_vox, np.array(start_vox) + BLOCK_PX)

    em_path = Path(save_dir) / f"em_{root_id}_{i}_{j}_{k}.npy"
    svx_path = Path(save_dir) / f"svx_{root_id}_{i}_{j}_{k}.npy"
    vas_path = Path(save_dir) / f"vas_{root_id}_{i}_{j}_{k}.npy"
    
    if em_path.exists() and svx_path.exists() and vas_path.exists(): return 0

    try:
        img_cv = CloudVolume(config.IMAGE_URL, mip=em_mip, use_https=True, fill_missing=True)
        seg_cv = CloudVolume(config.SEGMENTATION_URL, mip=seg_mip, use_https=True, fill_missing=True)
        seg_data = np.array(seg_cv.download(bbox_vox, timestamp=timestamp)).squeeze()
        em_data = np.array(img_cv.download(bbox_vox)).squeeze()

        np.save(em_path, em_data.astype(np.float32))
        np.save(svx_path, np.isin(seg_data, neuron_ids).astype(np.uint8))
        np.save(vas_path, np.isin(seg_data, list(vessel_ids)).astype(np.uint8) if vessel_ids else np.zeros_like(seg_data))
        return 0
    except Exception as e:
        print(f"Block ({i},{j},{k}) failed: {e}")
        return 1

# --- whitelist from CAVE tables ---
def build_whitelist_with_stats(client, table_names):
    available_tables = client.materialize.get_tables()
    full_set, stats = set(), {}
    for table in table_names:
        if table in available_tables:
            try:
                df = client.materialize.query_table(table)
                ids = df["pt_root_id"].unique()
                full_set.update(ids)
                stats[table] = len(ids)
            except: stats[table] = "Error"
    return full_set, stats

# --- pick random ROI and sample neuron/vessel IDs ---
def sample_roi_at_mip(client, img_cv, seg_cv, neuron_whitelist, vessel_whitelist, timestamp):
    res_seg = _get_resolution_nm(seg_cv, SEG_MIP_LEVEL)
    em_mip = get_matching_mip(img_cv, res_seg[0])
    roi_size_nm = (GRID_SIZE * BLOCK_PX) * res_seg
    seg_bounds = seg_cv.meta.bounds(SEG_MIP_LEVEL)

    while True:
        nuc = client.materialize.query_table("nucleus_detection_v0").sample(1).iloc[0]
        root_id = nuc["pt_root_id"]
        if root_id == 0 or (Path(OUTPUT_ROOT) / f"microns_{root_id}").exists(): continue

        center_nm = np.array(nuc["pt_position"]) * np.array([4, 4, 40])
        origin_nm = np.floor((center_nm - roi_size_nm / 2) / res_seg) * res_seg
        bbox_seg = Bbox(origin_nm / res_seg, (origin_nm + roi_size_nm) / res_seg)
        if not seg_bounds.contains_bbox(bbox_seg): continue

        try:
            preview = seg_cv.download(bbox_seg, mip=SEG_MIP_LEVEL, timestamp=timestamp)
            uniq = np.unique(preview)
            found_n = [int(x) for x in uniq if x in neuron_whitelist]
            found_v = [int(x) for x in uniq if x in vessel_whitelist]
            if not found_n: continue
            
            n_ids = random.sample(found_n, max(1, int(len(found_n) * N_RATIO)))
            v_ids = random.sample(found_v, max(1, int(len(found_v) * V_RATIO))) if found_v else []
            return origin_nm, em_mip, n_ids, v_ids, len(found_n), len(found_v), nuc
        except: continue

# --- main: one volume per run ---
if __name__ == "__main__":
    client = CAVEclient(config.DATASET_NAME)
    client.materialize.version = getattr(config, "MATERIALIZATION_VERSION", 1300)
    timestamp = int(client.materialize.get_timestamp(version=client.materialize.version).timestamp())

    NEURON_TABLES = ["allen_column_mtypes_v2", "aibs_metamodel_celltypes_v661", "baylor_gnn_cell_type_fine_model_v2", "bodor_pt_cells", "l5et_column"]
    VESSEL_TABLES = ["coregistration_manual_v4", "apl_functional_coreg_vess_fwd", "coregistration_auto_phase3_fwd_apl_vess_combined_v2"]
    
    neuron_whitelist, n_stats = build_whitelist_with_stats(client, NEURON_TABLES)
    vessel_whitelist, v_stats = build_whitelist_with_stats(client, VESSEL_TABLES)

    img_cv = CloudVolume(config.IMAGE_URL, use_https=True)
    seg_cv = CloudVolume(config.SEGMENTATION_URL, use_https=True)
    
    origin_nm, em_mip, neuron_ids, vessel_ids, f_n_count, f_v_count, nuc = sample_roi_at_mip(
        client, img_cv, seg_cv, neuron_whitelist, vessel_whitelist, timestamp
    )
    
    root_id = nuc["pt_root_id"]
    save_dir = Path(OUTPUT_ROOT) / f"microns_{root_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    res_seg = _get_resolution_nm(seg_cv, SEG_MIP_LEVEL)
    meta_info = {
        "root_id": root_id, "seg_mip": SEG_MIP_LEVEL, "em_mip": em_mip,
        "res_nm": res_seg.tolist(), "phys_size_um": (res_seg * GRID_SIZE * BLOCK_PX / 1000.0).tolist(),
        "voxel_origin": (origin_nm / res_seg).astype(int).tolist(), "origin_nm": origin_nm.tolist(),
        "found_neuron_count": f_n_count, "sampled_neuron_count": len(neuron_ids), "n_ratio": N_RATIO,
        "found_vessel_count": f_v_count, "sampled_vessel_count": len(vessel_ids), "v_ratio": V_RATIO,
        "neuron_tables": n_stats, "vessel_tables": v_stats
    }
    write_detailed_metadata(save_dir, meta_info)

    _log(f"Downloading {root_id} (grid {GRID_SIZE}^3)")
    tasks = [(str(save_dir), root_id, (i, j, k), np.array(meta_info["voxel_origin"]), SEG_MIP_LEVEL, em_mip, neuron_ids, vessel_ids, timestamp)
             for i, j, k in itertools.product(range(GRID_SIZE), range(GRID_SIZE), range(GRID_SIZE))]

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(tqdm(as_completed([pool.submit(download_one_block, t) for t in tasks]), total=len(tasks), desc="Blocks"))