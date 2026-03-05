# TPM-from-volume (simulation_module_v4)

Slab-based TPM simulation: slice volume into slabs, run NAOMi per slab, stitch output. Default slab 100×100×64 voxels (64nm×64nm×320nm). No external NAOMi; bundled in `naomi/`.

**Module renamed from v3 to v4** — see "Changes in v4 vs v3" for detailed differences.

## Usage

```matlab
cd simulation_module_v4

% Process all folders under input/
main()

% Process specified folder (e.g. input/microns_864691136811782003)
main('microns_864691136811782003')

% With custom options
opts = tpm_config();
opts.focal_spacing = 10;
main('microns_864691136811782003', opts)
```

## Input

Each subfolder under `input/` must contain `*neuron*.tif` (required) and optionally `*vessel*.tif`. Neurons and vessels are 3D TIFF stacks.

## Output

`output/<folder_name>/<stem>.tiff` — 2D frame stack TIFF per folder (v4 layout). Also writes `degrade_info.txt` with NAOMi parameters.

## Dependencies

- MATLAB R2017b+ (imresize3, Image Processing Toolbox)
- `naomi/` — bundled NAOMi scripts, no `../code` needed
- MEX: compile before first run:
  ```matlab
  cd naomi/MEX
  mex -largeArrayDims array_SubSubTest.cpp
  mex -largeArrayDims array_SubModTest.cpp
  ```

## Logging

Each run writes a debug log to `logs/log_<folder_name>_<yyyymmdd_HHMMSS>.log`. Function-level I/O: main, process_one, load_input_folder, find_neurons_vessels_paths, load_volume_tiff, get_slab_z_ranges, compute_vcpx, path_to_seed, extract_slab_vessels, naomi_bridge. Set `opts.log_path` to override or disable (empty = auto path).

## Parameters (tpm_config.m)

- fluor_scale — fluorescence scaling (0–1)
- voxel_um — voxel size [x,y,z] in µm
- vol_depth_um — focal depth under surface (µm)
- verbose — print progress
- opt_type — NAOMi optics type (e.g. standard)
- condition — NAOMi condition
- scan_avg — scan averaging frames
- sfrac — scan fraction (pixel sampling)
- slab_z — slab Z size (px)
- focal_spacing — slab step in voxels; overlap = slab_z − focal_spacing
- psf_sz — PSF size [x,y,z] in µm
- neuropil_enable — add neuropil/background fluorescence
- neuropil_scale — neuropil max as fraction of neur_vol max
- neuropil_thresh — background threshold
- pad_xy — XY pad (px) before degrade, crop after
- debug_psf — if true, save debug_psf.mat per slab (for diagnostics)
- log_path — debug log file ('' = main creates logs/log_<stem>.log)

## Layout

- `main.m` — entry point
- `tpm_config.m` — default parameters
- `scripts/` — process_one, naomi_bridge, run_from_path_files, load_volume_tiff, load_input_folder, find_neurons_vessels_paths, get_slab_z_ranges, extract_slab_vessels, path_to_seed, compute_vcpx, create_test_input, warn_append
- `naomi/` — OpticsCode, ScanningCode, VolumeCode, TimeTraceCode, MiscCode, ExternalPackages, MEX

---

## Changes in v4 vs v3

### New files in v4

| Path | Purpose |
|------|---------|
| `scripts/run_from_path_files.m` | Run from UTF-8 manifest (avoids non-ASCII path issues) |
| `scripts/warn_append.m` | Append warnings to log and stdout |
| `naomi/VolumeCode/add_neuropil.m` | Add neuropil/background fluorescence via masked_3DGP |

### main.m

| Block | v4 | v3 |
|-------|----|----|
| output_dir | Override from opts; handles relative vs absolute; mkdir after opts | No override; mkdir right after setup |
| folder_path | Supports absolute path (filesep or drive letter) | `folder_path = fullfile(input_dir, folder_name)` only |
| Output layout | `out_subdir = fullfile(output_dir, folder_short)`; `stem.tiff` per folder | Flat: `tpm_<stem>.tif` |
| log_path | opts.log_path override for custom log path | Always auto path |

### tpm_config.m

| Parameter | v4 | v3 |
|----------|----|----|
| voxel_um | [0.064, 0.064, 0.32] (64×64×320 nm) | [0.128, 0.128, 0.08] |
| vol_depth_um | 10.0 | 20.0 |
| scan_avg | 1 | 2 |
| slab_z | 64 | (not in opts, hardcoded 50) |
| focal_spacing | 3 | 5 |
| psf_sz | [4, 4, 12] | [3, 3, 4] |
| neuropil_enable | true | (absent) |
| neuropil_scale | 0.08 | (absent) |
| neuropil_thresh | 0.05 | (absent) |
| pad_xy | 4 | (absent) |

### process_one.m

| Block | v4 | v3 |
|-------|----|----|
| Neuropil | Calls add_neuropil() when neuropil_enable; uses VolumeCode, MiscCode | No neuropil |
| Pad XY | pad_xy edge replicate; crop after degrade | No pad/crop |
| Voxel/slab | voxel_um, vol_depth_um, slab_z, focal_spacing from opts | Hardcoded values |
| Vessels | Crop/pad to match vcpx; symmetric center crop or zero-pad; warn_append | imresize3 resize; warning() |
| Z crop | warn_append | warning |
| Frame read | pad_xy crop after reading H5 | No crop |
| TIFF | TIFF tags: ResolutionUnit, XResolution, YResolution, ImageDescription (ImageJ/Napari) | Basic TIFF only |
| degrade_info.txt | Writes degrade_info.txt with NAOMi params | No metadata file |

### naomi_bridge.m

| Block | v4 | v3 |
|-------|----|----|
| vol_sz_um | `vol_sz_um = [nx*vx, ny*vy, nz*vz]` (anisotropic); vres_z = 1/vz | vol_sz_um = [nx,ny,nz]/vres (isotropic) |
| vol_params | vol_params.vres_z; check_vol_params vol_sz restore; vol_params.log_path | No vres_z; no restore |
| PSF | NA=0.8, objNA=0.8; taillength = min(..., max(8, vol_sz_um(3))); sampling = min(..., max(2, vol_sz_um(1)/4)); prop_sz = 5; zernikeWt = [0 0 0 0 0.03 0 0 0 0 0 0.04]; blur=0 | No NA override; sampling = min(..., max(10, vol_sz_um(1)/4)); zernikeWt = [0 0 0 0 0.1 0 0 0 0 0 0.12] |
| vcpx | `vcpx = [round(vasc_sz(1)/vx), round(vasc_sz(2)/vy), round(vasc_sz(3)/vz)]` (anisotropic) | vcpx = round(vasc_sz * vres) |
| sigscale | vol_sz_um(1)*vol_sz_um(2)/(vx*vy) | vol_sz_um(1)*vol_sz_um(2)*(vres^2) |
| Z pad | warn_append | warning |
| Debug | Saves debug_psf.mat when opts.debug_psf=true | No debug save |

### compute_vcpx.m

| Block | v4 | v3 |
|-------|----|----|
| vol_sz_um | `vol_sz_um = [slab_nx*vx, slab_ny*vy, slab_nz*vz]`; vres_xy, vres_z | vol_sz_um = [slab_nx,slab_ny,slab_nz]/vres |
| vcpx | `vcpx = [round(vasc_sz(1)/vx), round(vasc_sz(2)/vy), round(vasc_sz(3)/vz)]` | vcpx = round(vasc_sz * vres) |

### extract_slab_vessels.m

| Block | v4 | v3 |
|-------|----|----|
| Z pad | warn_append | warning |
| Placement | tz = floor((sz-n)/2); symmetric placement ves_slab(:,:,1+tz:tz+n); XY fallback with warn_append | ves_slab(:,:,1:n); no symmetric placement |

### naomi/OpticsCode/simulate_optical_propagation.m

| Block | v4 | v3/code |
|-------|----|---------|
| vol_sz | Saves vol_sz_orig; restores vol_params.vol_sz after check_vol_params | No save/restore |
| vres_z | vres_z from vol_params.vres_z; anisotropic Z | Single vres |
| vcpx | `vcpx = [round(vasc_sz(1)*vres), round(vasc_sz(2)*vres), round(vasc_sz(3)*vres_z)]`; warn_append for TMPvasc | vcpx = round(vol_params.vasc_sz*vres) |

### naomi/OpticsCode/genCorticalLightPathLite.m

| Block | v4 | v3/code |
|-------|----|---------|
| vres_z | vres_z from vol_params.vres_z | No vres_z |
| psfpx, proppx | `psfpx = [psf_sz(1)*vres, psf_sz(2)*vres, psf_sz(3)*vres_z]`; `proppx = prop_sz*vres_z` | psfpx = psf_sz*vres; proppx = prop_sz*vres |
| N | N = ceil(N(1)/2)*2 for meshgrid | N = N(1) |
| zA, zB, zC | Use vres_z and proppx/vres_z | Use vres |

### naomi/VolumeCode/check_vol_params.m

| Block | v4 | v3 |
|-------|----|----|
| Rounding | Skips rounding when vol_sz(3) < 15 or mod(vol_sz(3),10)==0 | Always rounds vol_sz(3) to multiple of 10 |

### Unchanged scripts

load_input_folder.m, get_slab_z_ranges.m, find_neurons_vessels_paths.m, log_append.m, load_volume_tiff.m, path_to_seed.m, create_test_input.m, install_matlab_engine_from_matlab.m — same in v3 and v4.

---

## Differences from Original NAOMi

Original NAOMi lives in `naomi/adamshch-naomi_sim-65718ae7abb7/code/` and `resources/`. This module changes:

### Architecture and flow

- Entry: `main.m` instead of `TPM_Simulation_Script.m` / `TPM_Simulation_Script_LowRam.m`
- Dependency: uses `naomi/` inside the module instead of `../code`
- Volume handling: slab-based (slice, run per slab, stitch) instead of whole-volume
- Output: TIFF frame stack instead of HDF5 / multi-frame

### Parameter comparison: tpm_config.m vs original NAOMi

| Parameter | simulation_module_v4 | simulation_module_v3 | Original NAOMi (code/) |
|-----------|----------------------|----------------------|-------------------------|
| voxel_um | [0.064, 0.064, 0.32] (64×64×320 nm) | [0.128, 0.128, 0.08] | vres=2 → [0.5, 0.5, 0.5] µm |
| vol_depth_um | 10 | 20 | 100 (TPM_Simulation_Parameters) |
| psf_sz | [4, 4, 12] µm | [3, 3, 4] | [20, 20, 50] (standard) |
| slab_z | 64 px | 50 (hardcoded) | — (whole volume) |
| focal_spacing | 3 px | 5 | — |
| scan_avg | 1 | 2 | 2 (check_scan_params) |
| sfrac | 1 | 1 | 2 (check_scan_params) |
| fluor_scale | 0.8 | 0.8 | — |
| NA (PSF) | 0.8 (naomi_bridge override) | — | 0.6 (standard), objNA=0.8 |
| zernikeWt | [0 0 0 0 0.03 0 0 0 0 0 0.04] (v4) | [0 0 0 0 0.1 0 0 0 0 0 0.12] | [0 0 0 0 0.1 0 0 0 0 0 0.12] (check_psf_params) |
| blur | 0 (v4) | — | 3 (check_psf_params) |
| prop_sz | 5 (naomi_bridge) | min(10, focal_spacing_um) | 10 (check_psf_params) |
| taillength | min(50, max(8, slab_z_um)) | same | 50 (check_psf_params) |
| neuropil_enable | true | (absent) | — |
| pad_xy | 4 | (absent) | — |

### Parameter overrides (naomi_bridge.m)

Before calling NAOMi, these are overridden for thin slabs:

- taillength — min(50, max(8, slab_z_um)); propagation extent beyond slab
- prop_sz — min(10, focal_spacing_um); propagation step (µm)
- sampling — min(50, max(2, vol_sz_um(1)/4)); mask sampling (v4 uses max(2,...), v3 used max(10,...))
- psf_sz — from tpm_config

### Logic

- vcpx: computed by `compute_vcpx()` from vol_sz, voxel_um, vol_depth_um instead of hardcoded [1164, 1164, 400]
- vcpx_z_full: (num_slabs - 1) * focal_spacing + vcpx(3)

### Code changes in naomi/

- `OpticsCode/genCorticalLightPathLite.m`: phz size check; N, N2 floor; roff/coff round; padPre/padPost for padarray; c2, p1, p2 round; interp1 fallback when length mismatches.
- `OpticsCode/simulate_optical_propagation.m`: colpx(1), colpx(2) round; TMPves_pad third dim szB(3) instead of size(TMPves,3).
- `ScanningCode/scan_volume_frame.m`: array_SubSubTest/array_SubModTest return handling for .m fallback when MEX absent.
- `MEX/array_SubSubTest.m`, `array_SubModTest.m`: pure MATLAB fallbacks (NAOMi provides .cpp only).

### Functional summary

| Category | Changes |
|----------|---------|
| Anisotropic voxels | v4 supports different X/Y/Z voxel sizes (voxel_um, vres_z) in naomi_bridge, compute_vcpx, simulate_optical_propagation, genCorticalLightPathLite |
| Thin slabs | v4 avoids rounding thin slabs in check_vol_params; restores vol_sz after check_vol_params |
| Neuropil | v4 adds add_neuropil and neuropil_* options |
| Pad/crop | v4 adds pad_xy edge replicate and crop in process_one |
| Vessels | v4 uses crop/pad instead of imresize3; symmetric center handling |
| PSF | v4 sets NA=0.8, objNA=0.8, reduced Zernike weights, blur=0 |
| Output | v4 writes TIFF metadata and degrade_info.txt; per-folder output layout |
| Logging | v4 adds warn_append; more logging in bridge and scripts |
| Entry points | v4 adds run_from_path_files for manifest-based runs |

### Debugging

If output looks wrong: check for `genCorticalLightPathLite:interp1_length_mismatch`; verify tpm_config vs NAOMi defaults; confirm compute_vcpx vs vasc_sz/gaussianBeamSize; add prints in genCorticalLightPathLite, simulate_optical_propagation, scan_volume_frame.
