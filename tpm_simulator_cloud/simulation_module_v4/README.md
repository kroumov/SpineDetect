# TPM-from-volume (simulation_module_v4)

Slab-based two-photon microscopy (TPM) optical degradation simulation. Takes existing 3D volumes (neuron fluorescence and optional vessel masks) as input, slices them along Z into slabs, runs NAOMi optical propagation and scanning per slab, and stitches the 2D outputs into a TIFF frame stack. Unlike the original NAOMi, this module does not generate neuron or vessel geometry; it applies optical degradation to given volumes for the workflow: real or synthetic volume → simulated TPM imaging.

The module embeds a modified NAOMi codebase in `naomi/` and does not depend on external `../code`. It supports anisotropic voxels (e.g. XY 64 nm, Z 320 nm) and adjusts PSF and propagation parameters for thin slabs to avoid rounding and size issues in the original implementation. Optional neuropil background fluorescence is added via `add_neuropil`. Outputs are TIFF frame stacks and `degrade_info.txt` with NAOMi parameters.

## Usage

```matlab
cd simulation_module_v4

main()                                    % process all folders under input/
main('microns_864691136811782003')         % process specified folder
main('microns_864691136811782003', opts)   % with custom options from tpm_config()
```

## Input / Output

**Input:** Each subfolder under `input/` must contain `*neuron*.tif` (required) and optionally `*vessel*.tif`. Both are 3D TIFF stacks.

**Output:** `output/<folder_name>/<stem>.tiff` (2D frame stack) and `degrade_info.txt`.

**Data flow:**
```
input/<folder>/  →  process_one()  →  output/<folder>/<stem>.tiff
  *neuron*.tif        (per-slab: naomi_bridge → simulate_optical_propagation
  *vessel*.tif             → setup_scan_volume_frame → scan_volume_frame)
```

## Dependencies

- MATLAB R2017b+ (imresize3, Image Processing Toolbox)
- Bundled `naomi/` (no external NAOMi path required)
- MEX: compile before first run:
  ```matlab
  cd naomi/MEX
  mex -largeArrayDims array_SubSubTest.cpp
  mex -largeArrayDims array_SubModTest.cpp
  ```

## Parameters (tpm_config.m)

| Parameter | Default | Description |
|-----------|---------|-------------|
| fluor_scale | 0.8 | Fluorescence scaling (0–1) |
| voxel_um | [0.064, 0.064, 0.32] | Voxel size [x,y,z] µm (64×64×320 nm) |
| vol_depth_um | 10 | Focal depth under surface (µm) |
| slab_z | 64 | Slab Z size (px); XY inherits from input |
| focal_spacing | 3 | Slab step (px); overlap = slab_z − focal_spacing |
| NA | 1.0 | Excitation NA (high NA focus) |
| objNA | 1.0 | Objective NA (high NA focus) |
| psf_sz | [4, 4, 12] | PSF size [x,y,z] µm |
| zernikeWt | [0 0 0 0 0.01 0 0 0 0 0 0.02] | Zernike aberration weights |
| scan_avg | 1 | Scan averaging frames |
| sfrac | 1 | Scan fraction (pixel sampling) |
| neuropil_enable | true | Add neuropil background |
| neuropil_scale | 0.04 | Neuropil max as fraction of neur_vol max (0.05–0.15 typical) |
| neuropil_thresh | 0.05 | Background = voxels below this fraction of max |
| pad_xy | 4 | XY pad (px) before degrade, crop after |
| opt_type, condition | standard | NAOMi optics and condition |
| debug_psf | false | Save debug_psf.mat per slab |
| log_path | '' | Log file ('' = auto) |

## Comparison with Original NAOMi

Reference: `naomi/adamshch-naomi_sim-65718ae7abb7/code/` and `resources/`.

### Architecture and rationale

| Aspect | This module | Original NAOMi | Rationale |
|--------|-------------|----------------|-----------|
| Entry | main.m | TPM_Simulation_Script.m / TPM_Simulation_Script_LowRam.m | New entry point for volume-from-file workflow; original scripts assume geometry generation. |
| Dependency | naomi/ inside module | installNAOMi adds code/ | Self-contained deployment; no external NAOMi path. |
| Volume | Slab-based: slice → run per slab → stitch | Whole-volume | Reduces memory; enables processing of large volumes that exceed NAOMi’s whole-volume capacity. |
| Input | Existing TIFF (neurons + vessels) | simulate_neural_volume generates geometry | Module targets degradation of given volumes, not synthetic geometry. |
| Output | TIFF + degrade_info.txt | HDF5 / .mat / .tif | TIFF for downstream tools; degrade_info for reproducibility. |
| Time | Single frame (bg_act=1), no calcium dynamics | Full calcium dynamics + scanning | Only spatial degradation is needed; temporal simulation is out of scope. |

### Parameters and rationale

| Parameter | This module | Original NAOMi | Rationale |
|-----------|-------------|----------------|-----------|
| voxel_um | [0.064, 0.064, 0.32] µm | vres=2 → [0.5, 0.5, 0.5] µm | Match typical EM/chunk resolutions (64×64×320 nm). |
| vol_depth | 10 µm | 100 µm | Shallower depth for cortical imaging; affects vasc_sz/vcpx. |
| vres / vres_z | vres=1/vx, vres_z=1/vz (anisotropic) | Single vres (isotropic) | Support anisotropic voxels (XY vs Z). |
| slab_z, focal_spacing | 64 px, 3 px | — | Slab geometry for memory-efficient processing; overlap for smooth stitching. |
| scan_avg, sfrac | 1, 1 | 2, 2 | Single-frame spatial degradation; no temporal averaging. |
| scan_buff | 5 | 10 | Smaller buffer sufficient for slab-based scanning. |
| NA, objNA | 1.0, 1.0 | 0.6, 0.8 | High NA for sharper focus at fine voxel scale. |
| psf_sz | [4, 4, 12] µm | [20, 20, 50] µm | Smaller PSF to match finer voxels and slab dimensions. |
| zernikeWt | [0 0 0 0 0.01 0 0 0 0 0 0.02] | [0 0 0 0 0.1 0 0 0 0 0 0.12] | Reduced aberration weights for sharper PSF. |
| blur | 0 | 3 µm | Pipeline does not use blur kernel; avoid spurious lateral blur. |
| prop_sz | 5 µm | 10 µm | Smaller propagation step for thin slabs. |
| taillength | min(50, max(8, vol_sz_um(3))) | 50 µm | Adapt to slab Z extent; avoid exceeding slab bounds. |
| sampling | min(50, max(2, vol_sz_um(1)/4)) | 50 µm | Finer sampling for small lateral extent; max(2,...) prevents under-sampling. |
| mask exponent | 1.5 (mask^1.5) | 1 | Stronger tissue attenuation for more realistic degradation. |
| fluor_scale | 0.8 | — | Normalize input intensity to NAOMi’s expected range. |
| neuropil_*, pad_xy | Present (scale=0.04, thresh=0.05) | Absent | Neuropil: spatially correlated background; pad_xy: reduce edge artifacts from PSF convolution. |
| sigscale | vol_sz_um(1)*vol_sz_um(2)/(vx*vy) | vol_sz(1)*vol_sz(2)*(vres^2) | Anisotropic lateral area for correct photon scaling. |
| vcpx | vasc_sz / voxel_um per axis | round(vasc_sz * vres) | Anisotropic voxel counts from physical dimensions. |
| Vessel size mismatch | Symmetric center crop or zero-pad | TMPvasc temporary volume | Preserve center alignment when input vessel size differs from vcpx; avoid imresize distortion. |

### Code: Additions, modifications, omissions and rationale

**Additions**

| Path | Rationale |
|------|-----------|
| main.m | Entry for folder-based batch processing. |
| tpm_config.m | Central parameter store; original uses script-level variables. |
| process_one.m | Orchestrates slab slicing, naomi_bridge calls, and frame stitching. |
| naomi_bridge.m | Converts raw volume to NAOMi format; invokes optical propagation and scanning; single-frame path. |
| compute_vcpx.m | Computes vcpx from slab geometry and voxel_um (anisotropic); original uses hardcoded or vres-based values. |
| extract_slab_vessels.m | Extracts vessel slab from full propagation volume; symmetric center placement to preserve alignment. |
| load_volume_tiff.m, load_input_folder.m | Load TIFF volumes; original expects .mat or generated geometry. |
| get_slab_z_ranges.m | Computes overlapping slab Z ranges. |
| find_neurons_vessels_paths.m | Locates neuron/vessel files in input folder. |
| path_to_seed.m | Deterministic RNG seed from path for reproducibility. |
| log_append.m, warn_append.m | Structured logging; warnings to log and stdout. |
| run_from_path_files.m | Manifest-based runs; avoids non-ASCII path issues. |
| create_test_input.m | Generates synthetic test volumes for validation. |
| install_matlab_engine_from_matlab.m | Helper for Python–MATLAB engine setup. |
| add_neuropil.m | Adds spatially correlated neuropil via masked_3DGP; simulates background fluorescence. |
| array_SubSubTest.m, array_SubModTest.m | MATLAB fallbacks when MEX is absent; original provides .cpp only. |

**Modifications**

| Path | Change | Rationale |
|------|--------|-----------|
| check_vol_params.m | Skip vol_sz(3) rounding when vol_sz(3)&lt;15 or mod(vol_sz(3),10)==0 | Original rounds to multiple of 10; thin slabs (e.g. 10.24 µm) become 20 µm and cause phzA size mismatch in genCorticalLightPathLite. |
| simulate_optical_propagation.m | Save/restore vol_sz after check_vol_params; vcpx anisotropic; TMPvasc warn_append; colpx rounding; TMPves_pad third dim | Preserve actual slab size; support vres_z; robust handling when vessel size ≠ vcpx; fix indexing. |
| genCorticalLightPathLite.m | vres_z; anisotropic psfpx/proppx; N even; phz size check; roff/coff, c2, p1, p2 rounding; interp1 fallback | Anisotropic Z; correct pixel sizes; meshgrid symmetry; avoid size/length mismatches in thin slabs. |
| scan_volume_frame.m | array_SubSubTest/array_SubModTest return handling | Graceful fallback when MEX is missing. |

**Omissions**

| Path | Rationale |
|------|-----------|
| TPM_Simulation_Script.m, TPM_Simulation_Script_LowRam.m | Replaced by main.m + process_one + naomi_bridge. |
| simulate_neural_volume, generateTimeTraces, scan_volume | Geometry and temporal simulation not used; module takes existing volumes. |
| GUI/, AnalysisAndPlotting/, experimental/ | Visualization and analysis; not required for degradation. |
| installNAOMi.m, mex_compiling.m | Module manages paths and MEX internally. |

### Resources

Original `resources/` holds Neuroglancer project XML configs. This module does not use them; it only invokes simulation logic from `code/`.

## Logging

Each run writes `logs/log_<folder_name>_<yyyymmdd_HHMMSS>.log`. Set `opts.log_path` to override.

## Layout

- main.m, tpm_config.m
- scripts/: process_one, naomi_bridge, compute_vcpx, extract_slab_vessels, load_volume_tiff, load_input_folder, get_slab_z_ranges, find_neurons_vessels_paths, path_to_seed, log_append, run_from_path_files, warn_append, create_test_input
- naomi/: OpticsCode, ScanningCode, VolumeCode, TimeTraceCode, MiscCode, ExternalPackages, MEX

## Debugging

If output is incorrect: check for `genCorticalLightPathLite:interp1_length_mismatch`; verify tpm_config vs NAOMi defaults; confirm compute_vcpx vs vasc_sz/gaussianBeamSize; add prints in genCorticalLightPathLite, simulate_optical_propagation, scan_volume_frame.
