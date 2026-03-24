"""
Degrade pipeline: run NAOMi simulation on chunk data.

Input:  tpm_simulator/data/chunk/microns_{nid}_{idx}/
Output: tpm_simulator/data/degrade/microns_{nid}_{idx}/neurons_{nid}_{idx}.tiff, degrade_info.txt

Calls MATLAB simulation_module_v4 main() via run_from_path_files to avoid
encoding issues with non-ASCII paths.

Logs: tpm_simulator/pipelines/local/logs/degrade/degrade_YYYYMMDD_HHMMSS.log
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_DIR = Path(__file__).resolve().parent
CHUNK_DIR = PROJECT_ROOT / "data" / "chunk"
OUTPUT_DIR = PROJECT_ROOT / "data" / "degrade"
LOGS_DIR = LOCAL_DIR / "logs" / "degrade"
MODULE_DIR = PROJECT_ROOT / "simulation_module_v4"
MANIFEST_FILE = MODULE_DIR / "tmp" / "degrade_args.txt"

MICRONS_PATTERN = re.compile(r"^microns_(\d+)_(\d+)$")


def main():
    parser = argparse.ArgumentParser(description="Degrade pipeline: run NAOMi on chunk data")
    parser.add_argument("--folder", "-f", type=str, default=None, help="Process only this folder (name or path); when set, count defaults to 1")
    parser.add_argument("--count", "-n", type=int, default=None, help="Limit number of chunks to process")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"degrade_{ts}.log"

    def log(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(line)
        print(line.rstrip(), flush=True)

    if not CHUNK_DIR.is_dir():
        log(f"ERROR: chunk dir not found: {CHUNK_DIR}")
        sys.exit(1)

    if args.folder:
        folder_path = Path(args.folder)
        if folder_path.is_absolute():
            if folder_path.is_dir():
                folders = [folder_path]
            else:
                log(f"ERROR: folder not found: {folder_path}")
                sys.exit(1)
        else:
            candidate = CHUNK_DIR / args.folder
            if candidate.is_dir():
                folders = [candidate]
            else:
                log(f"ERROR: folder not found: {candidate}")
                sys.exit(1)
        if args.count is None:
            args.count = 1
    else:
        folders = sorted(
            d for d in CHUNK_DIR.iterdir()
            if d.is_dir() and MICRONS_PATTERN.match(d.name)
        )
        if not folders:
            log("No microns_*_* folders in chunk dir.")
            sys.exit(0)
        if args.count is not None:
            folders = folders[: args.count]

    log(f"Degrade pipeline started, log: {log_file}")
    log(f"Chunk dir: {CHUNK_DIR}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Folders to process: {len(folders)}" + (f" (folder={args.folder})" if args.folder else (f" (count={args.count})" if args.count else "")))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok_count = 0

    for i, chunk_path in enumerate(folders, 1):
        log(f"--- [{i}/{len(folders)}] {chunk_path.name} ---")
        chunk_abs = str(chunk_path.resolve())
        output_abs = str(OUTPUT_DIR.resolve())
        log_abs = str(log_file.resolve())

        # Use forward slashes for MATLAB compatibility on Windows
        chunk_abs = chunk_abs.replace("\\", "/")
        output_abs = output_abs.replace("\\", "/")
        log_abs = log_abs.replace("\\", "/")

        manifest_content = f"{chunk_abs}\n{output_abs}\n{log_abs}\n"
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_FILE.write_text(manifest_content, encoding="utf-8")

        cmd = [
            "matlab",
            "-batch",
            "addpath('scripts'); run_from_path_files('tmp/degrade_args.txt')",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(MODULE_DIR),
                capture_output=False,
            )
            if proc.returncode == 0:
                ok_count += 1
                log(f"OK: {chunk_path.name}")
            else:
                log(f"FAIL: {chunk_path.name} (exit {proc.returncode})")
        except FileNotFoundError:
            log("ERROR: matlab not found in PATH")
            sys.exit(1)
        except Exception as e:
            log(f"ERROR: {e}")
            sys.exit(1)

    log(f"Finished: {ok_count}/{len(folders)} succeeded")


if __name__ == "__main__":
    main()
