"""
Noise pipeline: add Poisson-Gaussian + pixel bleed to degrade outputs.

Input:  tpm_simulator/data/degrade/microns_*_*/neurons_*.tiff, degrade_info.txt
Output: tpm_simulator/data/noise/microns_*_*/neurons_*.tiff, noise_info.txt

Uses tpm_simulator/noise_model via run_noise_from_paths (MATLAB).
noise_info.txt = degrade_info content + noise parameters section.

Logs: tpm_simulator/pipelines/local/logs/noise/noise_YYYYMMDD_HHMMSS.log
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
DEGRADE_DIR = PROJECT_ROOT / "data" / "degrade"
OUTPUT_DIR = PROJECT_ROOT / "data" / "noise"
LOGS_DIR = LOCAL_DIR / "logs" / "noise"
NOISE_MODULE_DIR = PROJECT_ROOT / "noise_model"
MANIFEST_FILE = NOISE_MODULE_DIR / "noise_args.txt"

MICRONS_PATTERN = re.compile(r"^microns_\d+_\d+$")
TIFF_PATTERN = re.compile(r"^neurons_.+\.tiff?$", re.I)

# Noise model params (passed to MATLAB; edit here to tune)
NOISE_MU = 500         # Mean gain per photon
NOISE_SIGMA = 50     # Variance of gain per photon
NOISE_MU0 = 0          # Readout offset
NOISE_SIGMA0 = 0.1     # Readout noise std
NOISE_DARKCOUNT = 0.01 # PMT dark count rate
NOISE_BLEEDP = 0.01    # Pixel bleed probability (0 disables)
NOISE_BLEEDW = 0.05     # Max bleed fraction when it occurs


def _noise_params_section():
    """Format noise params for noise_info.txt."""
    return (
        "\n--- Noise Model (noise_config) ---\n"
        f"mu:               {NOISE_MU}\n"
        f"sigma:            {NOISE_SIGMA}\n"
        f"mu0:              {NOISE_MU0}\n"
        f"sigma0:           {NOISE_SIGMA0}\n"
        f"darkcount:        {NOISE_DARKCOUNT}\n"
        f"bleedp:           {NOISE_BLEEDP}\n"
        f"bleedw:           {NOISE_BLEEDW}\n"
        "========================================\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Add noise to degrade outputs")
    parser.add_argument("--count", "-n", type=int, default=None, help="Limit number of folders to process")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"noise_{ts}.log"

    def log(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
        with open(log_file, "a", encoding="utf-8", errors="replace") as lf:
            lf.write(line)
        try:
            print(line.rstrip(), flush=True)
        except UnicodeEncodeError:
            print(line.rstrip().encode("ascii", errors="replace").decode("ascii"), flush=True)

    if not DEGRADE_DIR.is_dir():
        log(f"ERROR: degrade dir not found: {DEGRADE_DIR}")
        sys.exit(1)

    folders = sorted(
        d for d in DEGRADE_DIR.iterdir()
        if d.is_dir() and MICRONS_PATTERN.match(d.name)
    )
    if not folders:
        log("No microns_*_* folders in degrade dir.")
        sys.exit(0)

    if args.count is not None:
        folders = folders[: args.count]

    log(f"Noise pipeline started, log: {log_file}")
    log(f"Degrade dir: {DEGRADE_DIR}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Folders to process: {len(folders)}" + (f" (count={args.count})" if args.count else ""))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok_count = 0

    for i, degrade_folder in enumerate(folders, 1):
        log(f"--- [{i}/{len(folders)}] {degrade_folder.name} ---")

        tiffs = [f for f in degrade_folder.iterdir() if f.is_file() and TIFF_PATTERN.match(f.name)]
        if not tiffs:
            log(f"  SKIP: no neurons_*.tiff found")
            continue

        tiff_path = tiffs[0]
        degrade_info_path = degrade_folder / "degrade_info.txt"

        out_folder = OUTPUT_DIR / degrade_folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        out_tiff_path = out_folder / tiff_path.name

        in_abs = str(tiff_path.resolve()).replace("\\", "/")
        out_dir_abs = str(out_folder.resolve()).replace("\\", "/")

        log(f"  Input tiff: {tiff_path.name}")
        log(f"  Input path: {in_abs}")
        log(f"  Output dir: {out_dir_abs}")

        manifest_content = (
            f"{in_abs}\n{out_dir_abs}\n"
            f"{NOISE_MU}\n{NOISE_SIGMA}\n{NOISE_MU0}\n{NOISE_SIGMA0}\n"
            f"{NOISE_DARKCOUNT}\n{NOISE_BLEEDP}\n{NOISE_BLEEDW}\n"
        )
        MANIFEST_FILE.write_text(manifest_content, encoding="utf-8")
        log(f"  [PARAMS] mu={NOISE_MU} sigma={NOISE_SIGMA} sigma0={NOISE_SIGMA0} "
            f"darkcount={NOISE_DARKCOUNT} bleedp={NOISE_BLEEDP} bleedw={NOISE_BLEEDW}")

        cmd = [
            "matlab",
            "-batch",
            "addpath('.'); run_noise_from_paths('noise_args.txt');",
        ]
        try:
            proc = subprocess.run(cmd, cwd=str(NOISE_MODULE_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.stdout:
                for line in proc.stdout.strip().split("\n"):
                    log(f"  MATLAB: {line}")
            if proc.stderr:
                for line in proc.stderr.strip().split("\n"):
                    log(f"  MATLAB stderr: {line}")
            if proc.returncode != 0:
                log(f"  FAIL: MATLAB exit {proc.returncode}")
                continue
        except FileNotFoundError:
            log("ERROR: matlab not found in PATH")
            sys.exit(1)
        except Exception as e:
            log(f"  ERROR: {e}")
            continue

        degrade_content = ""
        if degrade_info_path.exists():
            degrade_content = degrade_info_path.read_text(encoding="utf-8").rstrip()
            log(f"  [READ] degrade_info.txt ({len(degrade_content)} chars)")
        else:
            log(f"  [WARN] degrade_info.txt not found, noise_info will have no degrade section")
        noise_section = _noise_params_section()
        noise_info_content = degrade_content + noise_section
        (out_folder / "noise_info.txt").write_text(noise_info_content, encoding="utf-8")
        log(f"  Wrote noise_info.txt")

        out_tiff_exists = out_folder / tiff_path.name
        if out_tiff_exists.exists():
            sz = out_tiff_exists.stat().st_size
            log(f"  Output: {tiff_path.name} ({sz} bytes)")
        else:
            log(f"  WARN: output tiff not found: {out_tiff_exists}")

        ok_count += 1
        log(f"  OK: {tiff_path.name}")

    log(f"Finished: {ok_count}/{len(folders)} succeeded")


if __name__ == "__main__":
    main()
