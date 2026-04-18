"""
Sample random 3D volumes from MICrONS; writes blocks and metadata.
"""

import itertools
import logging
import os
import random
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from hks.mask_generation import mask_generation_pipeline

# Workaround for cloudfiles IntervalTree null-interval bug (start_us==end_us on fast requests)
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

import numpy as np
from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox
from tqdm import tqdm

try:
    import config
except ImportError:
    raise ImportError("config.py not found in the current directory.")

INFO_FILENAME = "download_info.txt"
# When LOG_DIR and LOG_FILE are set (e.g. by download.py pipeline), use them; else default to OUTPUT_ROOT_DIR/logs
_custom_log_dir = getattr(config, "LOG_DIR", None)
_custom_log_file = getattr(config, "LOG_FILE", None)
if _custom_log_dir and _custom_log_file:
    LOG_DIR = Path(_custom_log_dir)
    LOG_FILE = Path(_custom_log_file)
else:
    LOG_DIR = Path(getattr(config, "OUTPUT_ROOT_DIR", ".")) / "logs"
    LOG_FILE = LOG_DIR / f"sample_random_volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def _setup_logging():
    logger = logging.getLogger("sample_random_volume")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    # LOG_LEVEL=DEBUG: show all debug info in real-time (e.g. block progress, skips)
    stream_level = logging.DEBUG if os.environ.get("LOG_LEVEL") == "DEBUG" else logging.INFO
    ch.setLevel(stream_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger

def _log(msg):
    print(msg, flush=True)

_logger = None

def _get_logger():
    global _logger
    if _logger is None:
        _logger = _setup_logging()
    return _logger

SEG_MIP_LEVEL = getattr(config, "MIP_LEVEL", 2)
BLOCK_PX = getattr(config, "BLOCK_PX", 128)
_NG = getattr(config, "NEURON_GRID_SIZE", 4)
_NGX = getattr(config, "NEURON_GRID_SIZE_X", _NG)
_NGY = getattr(config, "NEURON_GRID_SIZE_Y", _NG)
_NGZ = getattr(config, "NEURON_GRID_SIZE_Z", _NG)
NEURON_GRID_SIZE = (_NGX, _NGY, _NGZ)
_VGX = getattr(config, "VESSEL_GRID_SIZE_X", _NGX)
_VGY = getattr(config, "VESSEL_GRID_SIZE_Y", _NGY)
_VGZ = getattr(config, "VESSEL_GRID_SIZE_Z", _NGZ)
VESSEL_GRID_SIZE = (_VGX, _VGY, _VGZ)
DOWNLOAD_WORKERS = getattr(config, "DOWNLOAD_WORKERS", 8)
OUTPUT_ROOT = getattr(config, "OUTPUT_ROOT_DIR", ".")
N_RATIO = getattr(config, "NEURON_SAMPLING_RATIO", 0.05)
V_RATIO = getattr(config, "VESSEL_SAMPLING_RATIO", 0.5)
BLOCK_DOWNLOAD_TIMEOUT = getattr(config, "BLOCK_DOWNLOAD_TIMEOUT", 180)  # seconds, 3 min

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
    ng = info_dict.get("neuron_grid", NEURON_GRID_SIZE)
    vg = info_dict.get("vessel_grid", VESSEL_GRID_SIZE)
    lines = [
        "========================================",
        "      MICRONS DATA DOWNLOAD INFO",
        "========================================",
        f"Nucleus ID:       {info_dict['root_id']}",
        f"MIP Level (SEG):  {info_dict['seg_mip']}",
        f"MIP Level (EM):   {info_dict['em_mip']}",
        f"Voxel Resolution: {info_dict['res_nm']} nm",
        f"Neuron Grid:      {ng[0]}x{ng[1]}x{ng[2]} blocks -> {ng[0]*BLOCK_PX}x{ng[1]*BLOCK_PX}x{ng[2]*BLOCK_PX} px",
        f"Vessel Grid:      {vg[0]}x{vg[1]}x{vg[2]} blocks -> {vg[0]*BLOCK_PX}x{vg[1]*BLOCK_PX}x{vg[2]*BLOCK_PX} px",
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
def _do_block_download(args):
    """Internal: one attempt, no timeout. Returns 0 on success, raises on error.
    args: (save_dir, root_id, (i,j,k), origin_vox_seg, seg_mip, em_mip, neuron_ids, vessel_ids, timestamp, vessel_only)
    vessel_only: if True, only download vas (no em, svx).
    """
    nargs = len(args)
    vessel_only = args[-1] if nargs >= 10 else False
    save_dir, root_id, (i, j, k), origin_vox_seg, seg_mip, em_mip, neuron_ids, vessel_ids, timestamp = args[:9]
    start_vox = (origin_vox_seg[0] + i * BLOCK_PX, origin_vox_seg[1] + j * BLOCK_PX, origin_vox_seg[2] + k * BLOCK_PX)
    bbox_vox = Bbox(start_vox, np.array(start_vox) + BLOCK_PX)

    seg_cv = CloudVolume(config.SEGMENTATION_URL, mip=seg_mip, use_https=True)
    seg_data = np.array(seg_cv.download(bbox_vox, timestamp=timestamp)).squeeze()
    vas_path = Path(save_dir) / f"vas_{root_id}_{i}_{j}_{k}.npy"
    np.save(vas_path, np.isin(seg_data, list(vessel_ids)).astype(np.uint8) if vessel_ids else np.zeros_like(seg_data))

    if vessel_only:
        return 0

    img_cv = CloudVolume(config.IMAGE_URL, mip=em_mip, use_https=True)
    em_data = np.array(img_cv.download(bbox_vox)).squeeze()
    em_path = Path(save_dir) / f"em_{root_id}_{i}_{j}_{k}.npy"
    svx_path = Path(save_dir) / f"svx_{root_id}_{i}_{j}_{k}.npy"
    np.save(em_path, em_data.astype(np.float32))
    np.save(svx_path, np.isin(seg_data, neuron_ids).astype(np.uint8))
    return 0

def download_one_block(args):
    log = _get_logger()
    nargs = len(args)
    vessel_only = args[-1] if nargs >= 10 else False
    save_dir, root_id, (i, j, k), origin_vox_seg, seg_mip, em_mip, neuron_ids, vessel_ids, timestamp = args[:9]
    t0 = time.perf_counter()
    em_path = Path(save_dir) / f"em_{root_id}_{i}_{j}_{k}.npy"
    svx_path = Path(save_dir) / f"svx_{root_id}_{i}_{j}_{k}.npy"
    vas_path = Path(save_dir) / f"vas_{root_id}_{i}_{j}_{k}.npy"

    if vessel_only:
        skip = vas_path.exists()
    else:
        skip = em_path.exists() and svx_path.exists() and vas_path.exists()
    if skip:
        log.debug(f"Block ({i},{j},{k}) vessel_only={vessel_only}: skipped (exists)")
        return 0

    log.debug(f"Block ({i},{j},{k}) root_id={root_id}")
    last_error = None
    for attempt in range(1, 4):  # 3 attempts
        result_box = [None]

        def run():
            try:
                _do_block_download(args)
                result_box[0] = 0
            except Exception as e:
                result_box[0] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=BLOCK_DOWNLOAD_TIMEOUT)

        if t.is_alive():
            duration = time.perf_counter() - t0
            log.warning(f"Block ({i},{j},{k}) timeout after {BLOCK_DOWNLOAD_TIMEOUT}s, retry {attempt}/3")
            print(f"Block ({i},{j},{k}) timeout ({BLOCK_DOWNLOAD_TIMEOUT}s), retry {attempt}/3")
            continue

        if isinstance(result_box[0], Exception):
            duration = time.perf_counter() - t0
            e = result_box[0]
            last_error = e
            err_detail = f"{type(e).__name__}: {e}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            tb_msg = "".join(tb_str).strip()
            log.warning(f"Block ({i},{j},{k}) attempt {attempt} ERROR: {err_detail}, duration={duration:.2f}s\n{tb_msg}")
            print(f"Block ({i},{j},{k}) failed (attempt {attempt}/3): {err_detail}")
            print(f"[TRACEBACK] {tb_msg}", flush=True)
            continue

        duration = time.perf_counter() - t0
        log.debug(f"Block ({i},{j},{k}) ok, attempt={attempt}, duration={duration:.2f}s")
        return 0

    log.exception(f"Block ({i},{j},{k}) FAILED after 3 attempts")
    print(f"Block ({i},{j},{k}) failed after 3 attempts")
    if last_error is not None:
        tb_msg = "".join(traceback.format_exception(type(last_error), last_error, last_error.__traceback__)).strip()
        print(f"[LAST_ERROR] {type(last_error).__name__}: {last_error}\n{tb_msg}", flush=True)
    return 1

# --- whitelist from CAVE tables ---
QUERY_CHUNK_SIZE = 50000

def build_whitelist_with_stats(client, table_names, table_filters=None):
    """Build whitelist from tables. Matches original microns logic: NO valid filter.
    valid=True was excluding 'good' neurons (valid=False in annotations), causing selection bias:
    we only succeeded when sampling nuclei in 'valid' regions, so we never hit the old good neurons."""
    log = _get_logger()
    t0 = time.perf_counter()
    log.debug(f"Whitelist IN: table_names={table_names}")

    available_tables = client.materialize.get_tables()
    full_set, stats = set(), {}
    table_filters = table_filters or {}
    # NO base_filters - original had none; valid=True caused sampling bias
    base_filters = {}

    for table in table_names:
        if table not in available_tables:
            log.debug(f"Whitelist skip {table}: not in available_tables")
            continue
        try:
            _log(f"Loading whitelist: {table}...")
            extra = table_filters.get(table, {})
            kwargs = dict(base_filters)
            for k, v in extra.items():
                if k == "filter_equal_dict":
                    kwargs[k] = {**kwargs.get(k, {}), **v}
                else:
                    kwargs[k] = v

            table_ids = set()
            offset = 0
            while True:
                df = client.materialize.query_table(
                    table, limit=QUERY_CHUNK_SIZE, offset=offset, **kwargs
                )
                if df.empty:
                    break

                if "pt_root_id" in df.columns:
                    valid_ids = df[df["pt_root_id"] > 0]["pt_root_id"].unique()
                    table_ids.update(valid_ids)

                if len(df) < QUERY_CHUNK_SIZE:
                    break
                offset += QUERY_CHUNK_SIZE
                _log(f"  {table}: +{len(df)} rows (offset {offset})")

            full_set.update(table_ids)
            stats[table] = len(table_ids)
            _log(f"  {table}: {len(table_ids)} IDs")
            log.debug(f"Whitelist table={table} ids={len(table_ids)}")
        except Exception as e:
            stats[table] = "Error"
            log.exception(f"Whitelist table={table} ERROR: {e}")

    duration = time.perf_counter() - t0
    log.debug(f"Whitelist OUT: total_ids={len(full_set)}, stats={stats}, duration={duration:.2f}s")
    return full_set, stats

# --- pick random ROI and sample neuron/vessel IDs ---
def sample_roi_at_mip(client, img_cv, seg_cv, neuron_whitelist, vessel_whitelist, timestamp):
    log = _get_logger()
    t0 = time.perf_counter()
    res_seg = _get_resolution_nm(seg_cv, SEG_MIP_LEVEL)
    em_mip = get_matching_mip(img_cv, res_seg[0])
    roi_size_vessel_nm = np.array([VESSEL_GRID_SIZE[0] * BLOCK_PX, VESSEL_GRID_SIZE[1] * BLOCK_PX, VESSEL_GRID_SIZE[2] * BLOCK_PX]) * res_seg
    roi_size_neuron_nm = np.array([NEURON_GRID_SIZE[0] * BLOCK_PX, NEURON_GRID_SIZE[1] * BLOCK_PX, NEURON_GRID_SIZE[2] * BLOCK_PX]) * res_seg
    seg_bounds = seg_cv.meta.bounds(SEG_MIP_LEVEL)
    log.debug(f"ROI sampling IN: whitelist_sizes=({len(neuron_whitelist)}, {len(vessel_whitelist)}), vessel_nm={roi_size_vessel_nm.tolist()} neuron_nm={roi_size_neuron_nm.tolist()}")

    attempt = 0
    while True:
        attempt += 1
        if attempt % 10 == 1:
            _log(f"Sampling ROI (attempt {attempt})...")
        nuc = client.materialize.query_table("nucleus_detection_v0").sample(1).iloc[0]
        root_id = nuc["pt_root_id"]
        if root_id == 0:
            log.debug(f"ROI sampling attempt={attempt} skip: root_id=0")
            continue
        if (Path(OUTPUT_ROOT) / f"microns_{root_id}").exists():
            log.debug(f"ROI sampling attempt={attempt} skip: microns_{root_id} exists")
            continue

        center_nm = np.array(nuc["pt_position"]) * np.array([4, 4, 40])
        origin_nm = np.floor((center_nm - roi_size_vessel_nm / 2) / res_seg) * res_seg
        bbox_bounds = Bbox(origin_nm / res_seg, (origin_nm + roi_size_vessel_nm) / res_seg)
        if not seg_bounds.contains_bbox(bbox_bounds):
            log.debug(f"ROI sampling attempt={attempt} skip: bbox out of bounds")
            continue

        try:
            bbox_preview = Bbox(origin_nm / res_seg, (origin_nm + roi_size_neuron_nm) / res_seg)
            preview = seg_cv.download(bbox_preview, mip=SEG_MIP_LEVEL, timestamp=timestamp)
            uniq = np.unique(preview)
            found_n = [int(x) for x in uniq if x in neuron_whitelist]
            found_v = [int(x) for x in uniq if x in vessel_whitelist]
            if not found_n:
                log.debug(f"ROI sampling attempt={attempt} root_id={root_id} skip: no neurons in ROI (uniq_count={len(uniq)})")
                continue

            n_ids = random.sample(found_n, max(1, int(len(found_n) * N_RATIO)))
            v_ids = random.sample(found_v, max(1, int(len(found_v) * V_RATIO))) if found_v else []
            out = (origin_nm, em_mip, n_ids, v_ids, len(found_n), len(found_v), nuc)
            duration = time.perf_counter() - t0
            log.debug(f"ROI sampling OUT: root_id={root_id}, found_n={len(found_n)}, found_v={len(found_v)}, duration={duration:.2f}s")
            return out
        except Exception as e:
            log.debug(f"ROI sampling attempt={attempt} root_id={root_id} retry: {e}")
            continue

# --- main: one volume per run ---
if __name__ == "__main__":
    log = _get_logger()
    log.debug(f"Start log_file={LOG_FILE}")
    _log(f"Log file: {LOG_FILE}")

    client = CAVEclient(config.DATASET_NAME)
    client.materialize.version = getattr(config, "MATERIALIZATION_VERSION", 1300)
    timestamp = int(client.materialize.get_timestamp(version=client.materialize.version).timestamp())

    NEURON_TABLES = ["allen_column_mtypes_v2", "aibs_metamodel_celltypes_v661", "baylor_gnn_cell_type_fine_model_v2", "bodor_pt_cells", "l5et_column"]
    NEURON_TABLE_FILTERS = {"aibs_metamodel_celltypes_v661": {"filter_in_dict": {"cell_type": ["23P"]}}}
    log.debug(f"neuron table filters: {NEURON_TABLE_FILTERS}")
#    NEURON_TABLE_FILTERS = {"aibs_metamodel_celltypes_v661": {"filter_out_dict": {"classification_system": "nonneuron"}}}
    # Vessel proxy: pericyte + astrocyte (vessel-associated cells) from aibs_metamodel_celltypes_v661.
    # Coregistration tables (coregistration_manual_v4, apl_functional_coreg_vess_fwd, etc.) contain pt_root_id
    # which are NEURON IDs, not vessel segment IDs.
    VESSEL_TABLES = ["aibs_metamodel_celltypes_v661"]
    VESSEL_TABLE_FILTERS = {"aibs_metamodel_celltypes_v661": {"filter_equal_dict": {"classification_system": "nonneuron"}, "filter_in_dict": {"cell_type": ["pericyte", "astrocyte"]}}}

    log.debug("Building neuron whitelist...")
    neuron_whitelist, n_stats = build_whitelist_with_stats(client, NEURON_TABLES, NEURON_TABLE_FILTERS)
    USE_VESSEL = getattr(config, "USE_VESSEL_MASK", True)
    if USE_VESSEL:
        log.debug("Building vessel whitelist...")
        vessel_whitelist, v_stats = build_whitelist_with_stats(client, VESSEL_TABLES, VESSEL_TABLE_FILTERS)
        # Pericyte/astrocyte IDs are disjoint from neuron IDs by construction
        _log(f"Whitelist ready: {len(neuron_whitelist)} neurons, {len(vessel_whitelist)} vessels")
    else:
        vessel_whitelist, v_stats = set(), {}
        _log(f"Whitelist ready: {len(neuron_whitelist)} neurons, vessels DISABLED (USE_VESSEL_MASK=False)")
    log.debug(f"Whitelist n_stats={n_stats} v_stats={v_stats}")

    img_cv = CloudVolume(config.IMAGE_URL, use_https=True)
    seg_cv = CloudVolume(config.SEGMENTATION_URL, use_https=True)
    log.debug("Sampling ROI...")
    origin_nm, em_mip, neuron_ids, vessel_ids, f_n_count, f_v_count, nuc = sample_roi_at_mip(
        client, img_cv, seg_cv, neuron_whitelist, vessel_whitelist, timestamp
    )
    log.debug(f"neuron root IDs: {neuron_ids}")
    
    root_id = nuc["pt_root_id"]
    save_dir = Path(OUTPUT_ROOT) / f"microns_{root_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log.debug(f"ROI chosen root_id={root_id} save_dir={save_dir} found_n={f_n_count} found_v={f_v_count}")
    
    res_seg = _get_resolution_nm(seg_cv, SEG_MIP_LEVEL)
    meta_info = {
        "root_id": root_id, "seg_mip": SEG_MIP_LEVEL, "em_mip": em_mip,
        "res_nm": res_seg.tolist(), "phys_size_um": (res_seg * np.array(NEURON_GRID_SIZE) * BLOCK_PX / 1000.0).tolist(),
        "voxel_origin": (origin_nm / res_seg).astype(int).tolist(), "origin_nm": origin_nm.tolist(),
        "found_neuron_count": f_n_count, "sampled_neuron_count": len(neuron_ids), "n_ratio": N_RATIO,
        "found_vessel_count": f_v_count, "sampled_vessel_count": len(vessel_ids), "v_ratio": V_RATIO,
        "neuron_tables": n_stats, "vessel_tables": v_stats,
        "neuron_grid": NEURON_GRID_SIZE, "vessel_grid": VESSEL_GRID_SIZE,
    }
    write_detailed_metadata(save_dir, meta_info)

    neuron_block_set = set(itertools.product(range(NEURON_GRID_SIZE[0]), range(NEURON_GRID_SIZE[1]), range(NEURON_GRID_SIZE[2])))
    vessel_block_set = set(itertools.product(range(VESSEL_GRID_SIZE[0]), range(VESSEL_GRID_SIZE[1]), range(VESSEL_GRID_SIZE[2])))
    vessel_only_set = vessel_block_set - neuron_block_set

    def mk_task(ijk, vessel_only):
        return (str(save_dir), root_id, ijk, np.array(meta_info["voxel_origin"]), SEG_MIP_LEVEL, em_mip, neuron_ids, vessel_ids, timestamp, vessel_only)
    neuron_tasks = [mk_task((i, j, k), False) for (i, j, k) in neuron_block_set]
    vessel_only_tasks = [mk_task((i, j, k), True) for (i, j, k) in vessel_only_set]
    tasks = neuron_tasks + vessel_only_tasks

    _log(f"Downloading {root_id} (neuron {NEURON_GRID_SIZE[0]}x{NEURON_GRID_SIZE[1]}x{NEURON_GRID_SIZE[2]}, vessel {VESSEL_GRID_SIZE[0]}x{VESSEL_GRID_SIZE[1]}x{VESSEL_GRID_SIZE[2]})")

    def _on_sigint(signum, frame):
        print("\nCtrl+C received. Exiting immediately.", flush=True)
        os._exit(130)

    def _on_sigterm(signum, frame):
        print("\nSIGTERM received. Exiting immediately.", flush=True)
        os._exit(143)

    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _on_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = [pool.submit(download_one_block, t) for t in tasks]
            pending = set(futures)
            results = []
            with tqdm(total=len(tasks), desc="Blocks") as pbar:
                while pending:
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for f in done:
                        pbar.update(1)
                        try:
                            results.append(f.result(timeout=0))
                        except Exception:
                            results.append(1)
        failed = sum(1 for r in results if r != 0)
        log.debug(f"End root_id={root_id}, blocks_failed={failed}")
        if failed > 0:
            print(f"WARN: {failed}/{len(tasks)} blocks failed (see tracebacks above)", flush=True)
    except KeyboardInterrupt:
        _log("Interrupted. Exiting.")
        sys.exit(130)

    ## generate labels using the hks features
    bbox_min = np.array(origin_nm)
    bbox_max = bbox_min + np.array(meta_info["phys_size_um"]) * 1000  # convert um back to nm
    mask_generation_pipeline(root_id, neuron_ids, client, bbox_min, bbox_max, res_seg)
