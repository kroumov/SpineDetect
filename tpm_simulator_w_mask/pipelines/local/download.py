"""
Local MICrONS batch download. Same parameters as batch_download.sh.
Output: tpm_simulator/data/download
Logs: local/logs/download/ or cloud/logs/download/ (when CLOUD_LOGS_DIR set)
"""
import os
import random
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT / "sample_random_volume"
DOWNLOAD_BASE = PROJECT_ROOT / "data" / "download"
LOGS_DIR = LOCAL_DIR / "logs" / "download"

MIPS = [3]
TOTAL_COUNT = 1
NEURON_RATIO_MIN = 0.01
NEURON_RATIO_MAX = 0.02
VESSEL_RATIO_MIN = 0.5
VESSEL_RATIO_MAX = 1.0

def _tee_output(pipe, file_handle):
    """Read from pipe, write to both file and stdout."""
    while True:
        data = pipe.read(4096)
        if not data:
            break
        file_handle.write(data)
        file_handle.flush()
        sys.stdout.write(data)
        sys.stdout.flush()
    pipe.close()


def patch_config(updates: dict) -> None:
    """Patch config.py with updates."""
    config_path = SCRIPT_DIR / "config.py"
    content = config_path.read_text(encoding="utf-8")
    for key, val in updates.items():
        if isinstance(val, str):
            val_repl = repr(val.replace("\\", "/"))
        else:
            val_repl = str(val)
        content = re.sub(rf"^({re.escape(key)})\s*=\s*.*$", rf"\1 = {val_repl}", content, flags=re.M)
    config_path.write_text(content, encoding="utf-8")


def main():
    if not SCRIPT_DIR.is_dir():
        print(f"ERROR: {SCRIPT_DIR} not found")
        sys.exit(1)
        
    print(f"!!!!!!! {SCRIPT_DIR}")

    DOWNLOAD_BASE.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCRIPT_DIR))

    config_path = SCRIPT_DIR / "config.py"
    original_config = config_path.read_text(encoding="utf-8")
    try:
        _run_downloads(config_path, original_config)
    finally:
        config_path.write_text(original_config, encoding="utf-8")
    print(f"Done: {DOWNLOAD_BASE}")


def _run_downloads(config_path, original_config):
    # Cloud: cloud/logs/download/; Local: local/logs/download/
    cloud_logs = os.environ.get("CLOUD_LOGS_DIR")
    _logs_base = Path(cloud_logs) / "download" if cloud_logs else LOGS_DIR
    _logs_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _logs_base / f"download_{ts}.log"
    patch_config({
        "LOG_DIR": str(_logs_base),
        "LOG_FILE": str(log_file),
    })

    num_mips = len(MIPS)
    per_mip = TOTAL_COUNT // num_mips

    with open(log_file, "w", encoding="utf-8") as lf:
        def log(msg):
            line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
            lf.write(line)
            lf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

        def run_with_tee(cmd, cwd):
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            t = threading.Thread(target=_tee_output, args=(proc.stdout, lf))
            t.daemon = True
            t.start()
            proc._tee_thread = t
            return proc

        log(f"Download pipeline started, log: {log_file}")

        for mip in MIPS:
            log(f"MIP {mip} ({per_mip} volumes)")

            for v in range(1, per_mip + 1):
                log(f"[MIP {mip}] {v}/{per_mip}")
                nr = random.uniform(NEURON_RATIO_MIN, NEURON_RATIO_MAX)
                vr = random.uniform(VESSEL_RATIO_MIN, VESSEL_RATIO_MAX)
                patch_config({
                    "OUTPUT_ROOT_DIR": str(DOWNLOAD_BASE),
                    "MIP_LEVEL": mip,
                    "NEURON_SAMPLING_RATIO": nr,
                    "VESSEL_SAMPLING_RATIO": vr,
                })

                proc = run_with_tee(
                    [sys.executable, str(SCRIPT_DIR / "sample_random_volume.py")],
                    str(SCRIPT_DIR),
                )
                try:
                    ret = proc.wait()
                    proc._tee_thread.join(timeout=2)
                except KeyboardInterrupt:
                    log("Ctrl+C received. Terminating child process...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise
                if ret != 0:
                    log("ERROR: sample_random_volume.py failed")
                    continue

                vols = sorted(DOWNLOAD_BASE.glob("microns_*"), key=os.path.getmtime, reverse=True)
                if not vols:
                    log("ERROR: No microns_* directory.")
                    continue

                vol_dir = vols[0]
                proc = run_with_tee(
                    [sys.executable, str(SCRIPT_DIR / "accumulate_roi.py"), str(vol_dir)],
                    str(SCRIPT_DIR),
                )
                try:
                    ret = proc.wait()
                    proc._tee_thread.join(timeout=2)
                except KeyboardInterrupt:
                    log("Ctrl+C received. Terminating child process...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise
                if ret != 0:
                    log("ERROR: accumulate_roi.py failed")
                    continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(130)