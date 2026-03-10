function opts = tpm_config(log_path)
% TPM_CONFIG  Default parameters for TPM-from-volume.
%
% opts = tpm_config(log_path)
% log_path: optional (unused, kept for API compatibility)

if nargin < 1, log_path = ''; end  %#ok<NASGU>

opts = struct();
opts.fluor_scale     = 0.9;   % fluorescence scaling (0–1)
opts.voxel_um        = [0.064, 0.064, 0.32];  % voxel size [x,y,z] in µm (64x64x320 nm)
opts.vol_depth_um    = 10.0;   % focal depth under surface (µm). Affects vasc_sz/vcpx; fallback crop/pad handles size mismatch
opts.verbose         = true;  % print progress
opts.opt_type        = 'standard';   % NAOMi optics type
opts.condition       = 'standard';  % NAOMi condition
opts.scan_avg        = 1;     % scan averaging frames
opts.sfrac           = 1;     % scan fraction (pixel sampling)
opts.slab_z          = 64;    % slab Z size (px)
opts.focal_spacing   = 3;     % slab step in voxels (stride)
opts.NA              = 1.0;   % excitation NA (high NA focus)
opts.objNA           = 1.0;   % objective NA (high NA focus)
opts.psf_sz          = [8, 8, 25];  % PSF size [x,y,z] in µm
opts.zernikeWt       = [0 0 0 0 0.01 0 0 0 0 0 0.02];  % Zernike aberration weights
opts.neuropil_enable = true;  % add neuropil/background fluorescence (uses NAOMi masked_3DGP)
opts.neuropil_scale  = 0.04;  % neuropil max as fraction of neur_vol max (0.05–0.15 typical)
opts.neuropil_thresh = 0.04;  % background = voxels below this fraction of max
opts.pad_xy          = 4;     % XY pad (px) before degrade, crop same amount after; 0 = no pad/crop
opts.max_slabs       = 'none';  % test: process N slabs; 'none' = all
opts.max_slabs_start = 8;       % when max_slabs set: start from slab index (1-based); e.g. start=5, max=3 -> slabs 5,6,7
opts.debug_psf      = false; % if true, save debug_psf.mat per slab (for diagnostics)
opts.log_path       = '';    % if set, append debug log to file; '' = main creates logs/log_<folder>_<timestamp>.log
end
