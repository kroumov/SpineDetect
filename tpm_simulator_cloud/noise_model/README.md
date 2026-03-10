# 2P Noise Model (Standalone)

MATLAB noise model for two-photon microscopy: Poisson–Gaussian detector noise and pixel bleed. Processes multi-layer TIFFs in `input/`, writes to `output/` (same filename). Input must come from simulation_module_v4 (photon-scaled clean images).

## Usage

```matlab
cd noise_model
main
```

## Input

Multi-layer TIFFs in `input/`. Expects photon units (simulation_module_v4 output).

## Output

`output/<stem>.tif` — noisy TIFF, same filename as input.

## Parameters (noise_config.m)

| Parameter | Default | Description |
|-----------|---------|--------------|
| mu | 100 | Mean gain per photon |
| sigma | 2300 | Variance of gain per photon |
| mu0 | 0 | Readout offset |
| sigma0 | 2.7 | Readout noise std |
| darkcount | 0.05 | PMT dark count |
| bleedp | 0.3 | Pixel bleed probability; 0 disables |
| bleedw | 0.4 | Max bleed fraction |

## Layout

- `main.m` — entry point (batch process input/*.tif* → output/)
- `run_noise_from_paths.m` — run from UTF-8 manifest (paths + optional params)
- `noise_config.m` — parameters
- `applyNoiseModel.m`, `PoissonGaussNoiseModel.m`, `check_noise_params.m`, `pixel_bleed.m` — from NAOMi (identical)

---

## Differences from Original NAOMi

Original NAOMi noise model: `naomi/adamshch-naomi_sim-65718ae7abb7/noise_model/`

### Files: Identical vs Added

| File | Status |
|------|--------|
| main.m | Identical |
| noise_config.m | Identical |
| applyNoiseModel.m | Identical |
| PoissonGaussNoiseModel.m | Identical |
| check_noise_params.m | Identical |
| pixel_bleed.m | Identical |
| **run_noise_from_paths.m** | Pipeline integration — not in original |

### run_noise_from_paths.m (Pipeline Integration)

- **Purpose**: Pipeline integration; avoids passing non-ASCII paths via MATLAB command line
- **Input**: UTF-8 manifest file
  - Line 1: input_tiff_path (absolute)
  - Line 2: output_dir (absolute)
  - Lines 3–9 (optional): mu, sigma, mu0, sigma0, darkcount, bleedp, bleedw — override noise_config
- **Flow**: Read manifest → load TIFF → applyNoiseModel → write TIFF to output_dir
- **Used by**: `tpm_simulator/pipelines/local/noise.py` — writes manifest, calls MATLAB with `run_noise_from_paths('noise_args.txt')`

### Integration: Original NAOMi vs tpm_simulator

| Aspect | Original NAOMi | tpm_simulator |
|--------|-----------------|-------|
| **Noise application** | Two modes: (1) Inside simulation (scan_volume.m applies PoissonGaussNoiseModel + pixel_bleed per frame); (2) Standalone noise_model batch on TIFFs | Standalone only: simulation_module_v4 outputs clean (photon-scaled) TIFF; noise pipeline applies noise separately |
| **sigscale** | Computed in naomi_bridge/TPM_Simulation_Parameters: `tpmSignalscale*(dt)*(sfrac^2)/(vol_sz*vres^2)`; applied to clean_img before noise | Applied inside simulation_module_v4 (naomi_bridge) before writing; noise_model receives pre-scaled input, does not apply sigscale |
| **Input** | Standalone: input/*.tif*; Simulation: clean frames in memory | degrade/*.tiff (from simulation_module_v4) |
| **Output** | Standalone: output/<stem>.tif; Simulation: HDF5/movie with noise | tpm_simulator/data/train/tpm/<folder>/neurons_*.tiff + noise_info.txt |
| **Config** | noise_config.m only | noise_config.m + noise.py overrides (manifest lines 3–9) |

### Parameter comparison: tpm_simulator vs original NAOMi

**noise_config.m** — identical to original NAOMi `noise_model/noise_config.m`:

| Parameter | noise_config.m (tpm_simulator = original) | Description |
|-----------|-----------------------------------|-------------|
| mu | 100 | Mean gain per photon |
| sigma | 2300 | Variance of gain per photon |
| mu0 | 0 | Readout offset |
| sigma0 | 2.7 | Readout noise std |
| darkcount | 0.05 | PMT dark count |
| bleedp | 0.3 | Pixel bleed probability |
| bleedw | 0.4 | Max bleed fraction |

**check_noise_params.m** defaults (tpm_simulator = original): mu=100, sigma=2300, sigma0=2.7, darkcount=0.05, sigscale=2e-7, bleedp=0.3, bleedw=0.4.

**noise.py overrides** — the pipeline passes params via manifest; these override `noise_config.m` when running `python noise.py`:

| Parameter | noise_config.m (original) | noise.py (pipeline) |
|-----------|---------------------------|---------------------------|
| mu | 100 | 100 |
| sigma | 2300 | **10** |
| sigma0 | 2.7 | **0.1** |
| darkcount | 0.05 | **0.01** |
| bleedp | 0.3 | **0** |
| bleedw | 0.4 | **0** |

Edit `NOISE_*` in `tpm_simulator/pipelines/local/noise.py` to tune pipeline noise.

### Core Algorithm (Unchanged)

- **PoissonGaussNoiseModel**: `poissrnd(clean_in + darkcount)` → log-normal (mu, sigma) → add `normrnd(mu0, sigma0)`
- **pixel_bleed**: With probability p, bleed fraction up to b_max from previous pixel (left/up)
- **applyNoiseModel**: Per-frame PoissonGaussNoiseModel → pixel_bleed; supports `type='dynode'` (not used in tpm_simulator)
