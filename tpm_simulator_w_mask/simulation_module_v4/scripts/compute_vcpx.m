function [vcpx, vasc_sz, vol_sz_um] = compute_vcpx(voxel_um, vol_depth_um, slab_nx, slab_ny, slab_nz, psf_sz, opt_type, condition, log_path, opts)
% COMPUTE_VCPX  Compute vessel propagation volume size (vcpx) for NAOMi.
%
% [vcpx, vasc_sz, vol_sz_um] = compute_vcpx(voxel_um, vol_depth_um, slab_nx, slab_ny, slab_nz, psf_sz, opt_type, condition, log_path, opts)
% opts: optional struct from tpm_config(); NA, objNA override psf_params when present
%
% vcpx: [X,Y,Z] voxel counts for vessel volume
% vasc_sz: propagation volume size in µm
% vol_sz_um: slab size in µm
% log_path: optional, append debug info

if nargin < 9, log_path = ''; end
if nargin < 10, opts = struct(); end

base_dir = fileparts(fileparts(mfilename('fullpath')));  % module root
naomi_path = fullfile(base_dir, 'naomi');
if isfolder(naomi_path)
  addpath(genpath(naomi_path));
end

if nargin < 6, psf_sz = []; end
if nargin < 7, opt_type = 'standard'; end
if nargin < 8, condition = 'standard'; end

if ~isempty(log_path) && exist('log_append','file')
  log_append(log_path, '[COMPUTE_VCPX] IN voxel_um=[%.3g,%.3g,%.3g] vol_depth_um=%.1f slab=[%d,%d,%d] psf_sz=%s opt_type=%s\n', ...
    voxel_um(1), voxel_um(min(2,end)), voxel_um(min(3,end)), vol_depth_um, slab_nx, slab_ny, slab_nz, mat2str(psf_sz), opt_type);
end

if numel(voxel_um) == 1
  voxel_um = [voxel_um, voxel_um, voxel_um];
end
vx = voxel_um(1);
vy = voxel_um(min(2, numel(voxel_um)));
vz = voxel_um(min(3, numel(voxel_um)));
vres_xy = 1 / vx;
vres_z = 1 / vz;

% vol_sz in µm (anisotropic: each dim uses its own voxel size)
vol_sz_um = double([slab_nx * vx, slab_ny * vy, slab_nz * vz]);
if ~isempty(log_path) && exist('log_append','file')
  log_append(log_path, '[COMPUTE_VCPX] vol_sz_um=[%.2f,%.2f,%.2f] vres_xy=%.4g vres_z=%.4g (anisotropic)\n', vol_sz_um(1), vol_sz_um(2), vol_sz_um(3), vres_xy, vres_z);
end

% Get psf_params for gaussianBeamSize (must match naomi_bridge overrides)
params_init = TPM_Simulation_Parameters([], opt_type, condition);
psf_params = params_init.psf_params;
if ~isempty(psf_sz)
  psf_params.psf_sz = psf_sz;
end
if isfield(opts, 'NA') && ~isempty(opts.NA)
  psf_params.NA = opts.NA(1);
end
if isfield(opts, 'objNA') && ~isempty(opts.objNA)
  psf_params.objNA = opts.objNA(1);
end
if ~isempty(log_path) && exist('log_append','file')
  log_append(log_path, '[COMPUTE_VCPX] psf_sz=[%.1f,%.1f,%.1f] um\n', psf_params.psf_sz(1), psf_params.psf_sz(2), psf_params.psf_sz(3));
end

% vasc_sz = gaussianBeamSize(...) + vol_sz + [0,0,1]*vol_depth
vol_params = struct();
vol_params.vol_sz = vol_sz_um;
vol_params.vol_depth = vol_depth_um;
gbs = gaussianBeamSize(psf_params, vol_depth_um + vol_sz_um(3)/2);
vasc_sz = gbs + vol_sz_um + [0 0 1]*vol_depth_um;
vasc_sz_min_z = vol_depth_um + vol_sz_um(3)/2 + psf_params.psf_sz(3)/2;
vasc_sz(3) = max(vasc_sz(3), vasc_sz_min_z);
if ~isempty(log_path) && exist('log_append','file')
  log_append(log_path, '[COMPUTE_VCPX] gaussianBeamSize=[%.1f,%.1f,%.1f] vasc_sz_min_z=%.2f vasc_sz=[%.2f,%.2f,%.2f] um\n', ...
    gbs(1), gbs(2), gbs(3), vasc_sz_min_z, vasc_sz(1), vasc_sz(2), vasc_sz(3));
end

% vcpx: voxels per dim (anisotropic: vasc_sz in µm, divide by voxel size per dim)
vcpx = [round(vasc_sz(1) / vx), round(vasc_sz(2) / vy), round(vasc_sz(3) / vz)];
if ~isempty(log_path) && exist('log_append','file')
  log_append(log_path, '[COMPUTE_VCPX] OUT vcpx=[%d,%d,%d] (anisotropic)\n', vcpx(1), vcpx(2), vcpx(3));
end
end
