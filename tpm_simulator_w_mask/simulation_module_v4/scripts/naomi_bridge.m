function err = naomi_bridge(input_mat_path)
% NAOMI_BRIDGE  Run TPM-from-volume using NAOMi original pipeline (single frame).
%
% err = naomi_bridge(input_mat_path)
%
% Uses NAOMi pipeline: simulate_optical_propagation -> setup_scan_volume_frame
% -> scan_volume_frame. Input raw volume is converted to vol_out format
% (gp_vals empty, bg_proc = whole volume). Single frame only at center z.
%
% Loads input_mat_path which must contain:
%   neur_vol      - 3D single/double (H,W,Z), fluorescence
%   neur_ves      - 3D same size or [] (optional)
%   voxel_um      - [vx,vy,vz] in µm
%   vol_depth_um  - depth in µm
%   output_path   - base path; bridge writes .h5 (NAOMi LowRam style), Python converts to .tif
%   opt_type, condition, etc.
%
% On success err = 0; on error err = 1.

err = 1;
if nargin < 1 || isempty(input_mat_path)
  warning('naomi_bridge:NoInput', 'Usage: naomi_bridge(path_to_input.mat)');
  return;
end
input_mat_path = char(input_mat_path);
if ~isfile(input_mat_path)
  warning('naomi_bridge:NotFound', 'File not found: %s', input_mat_path);
  return;
end

base_dir = fileparts(fileparts(mfilename('fullpath')));  % module root
naomi_path = fullfile(base_dir, 'naomi');
if ~isfolder(naomi_path)
  warning('naomi_bridge:NoNAOMi', 'naomi folder not found: %s', naomi_path);
  return;
end

orig_pwd = pwd;
try
  addpath(genpath(fullfile(naomi_path, 'MEX')));
  addpath(genpath(fullfile(naomi_path, 'MiscCode')));
  addpath(genpath(fullfile(naomi_path, 'OpticsCode')));
  addpath(genpath(fullfile(naomi_path, 'TimeTraceCode')));
  addpath(genpath(fullfile(naomi_path, 'VolumeCode')));
  addpath(genpath(fullfile(naomi_path, 'ScanningCode')));
  addpath(genpath(fullfile(naomi_path, 'ExternalPackages')));
  addpath(naomi_path);
  cd(orig_pwd);
catch e
  cd(orig_pwd);
  warning('naomi_bridge:Install', 'addpath failed: %s', e.message);
  return;
end

S = load(input_mat_path);
log_path = get_opt(S, 'log_path', '');
if ischar(log_path) && ~isempty(log_path)
  log_path = log_path(:).';
  fid_log = fopen(log_path, 'a');
  if fid_log < 0, fid_log = 0; end
else
  fid_log = 0;
end
mlog = @(varargin) mlog_impl(fid_log, varargin{:});
mlog('[BRIDGE ENTER] naomi_bridge( input_mat_path=%s ) NAOMi pipeline, single frame\n', input_mat_path);
mlog('[BRIDGE] pwd=%s\n', pwd);

neur_vol = S.neur_vol;
if isfield(S, 'neur_ves') && ~isempty(S.neur_ves)
  neur_ves = S.neur_ves;
  mlog('[STEP neur_ves] size=[%d,%d,%d] class=%s\n', size(neur_ves,1), size(neur_ves,2), size(neur_ves,3), class(neur_ves));
else
  neur_ves = [];
  mlog('[STEP neur_ves] empty\n');
end
voxel_um = S.voxel_um;
if numel(voxel_um) == 1, voxel_um = [voxel_um voxel_um voxel_um]; end
vol_depth_um = S.vol_depth_um;
output_path = char(S.output_path);
is_abs = ~isempty(output_path) && (output_path(1)=='/' || output_path(1)=='\' || (ispc && numel(output_path)>=2 && output_path(2)==':'));
mlog('[STEP output_path] output_path=%s (absolute=%d)\n', output_path, is_abs);
verbose = get_opt(S, 'verbose', true);
opt_type = get_opt(S, 'opt_type', 'standard');
condition = get_opt(S, 'condition', 'standard');
if isstring(opt_type), opt_type = char(opt_type); end
if iscell(opt_type), opt_type = char(opt_type{1}); end
if isstring(condition), condition = char(condition); end
if iscell(condition), condition = char(condition{1}); end

V = single(neur_vol);
[nx, ny, nz] = size(V);

vx = voxel_um(1);
vy = voxel_um(min(2, numel(voxel_um)));
vz = voxel_um(min(3, numel(voxel_um)));
vres = 1 / vx;   % XY resolution (pixels/um)
vres_z = 1 / vz; % Z resolution for anisotropic voxels

% vol_sz in µm (anisotropic: each dim uses its own voxel size)
vol_sz_um = double([nx * vx, ny * vy, nz * vz]);

mlog('[STEP load_mat] vol_size=[%d,%d,%d], voxel_um=[%.4g,%.4g,%.4g], vol_depth_um=%.4g\n', nx, ny, nz, vx, vy, vz, vol_depth_um);
mlog('[STEP load_mat] vol_sz_um=[%.2f,%.2f,%.2f] vres=%.4g vres_z=%.4g (anisotropic)\n', vol_sz_um(1), vol_sz_um(2), vol_sz_um(3), vres, vres_z);

vol_params = struct();
vol_params.vol_sz   = vol_sz_um;
vol_params.vres     = vres;
vol_params.vres_z   = vres_z;  % Z resolution for anisotropic
vol_params.vol_depth = vol_depth_um;
vol_params = check_vol_params(vol_params);
% Restore actual vol_sz: check_vol_params rounds vol_sz(3) to multiple of 10,
% which breaks thin slabs (e.g. 10.24um -> 20um causes phzA size mismatch)
vol_params.vol_sz   = vol_sz_um;
vol_params.verbose  = 0;
if ~isempty(log_path)
  vol_params.log_path = log_path;
end

params_init = TPM_Simulation_Parameters([], opt_type, condition);
psf_params = params_init.psf_params;
if isfield(S, 'psf_sz') && ~isempty(S.psf_sz)
  psf_params.psf_sz = S.psf_sz(1:3);
  mlog('[STEP psf_sz] override to [%.1f,%.1f,%.1f] um\n', psf_params.psf_sz(1), psf_params.psf_sz(2), psf_params.psf_sz(3));
end
if isfield(S, 'NA') && ~isempty(S.NA)
  psf_params.NA = S.NA(1);
else
  psf_params.NA = 1.0;
end
if isfield(S, 'objNA') && ~isempty(S.objNA)
  psf_params.objNA = S.objNA(1);
else
  psf_params.objNA = 1.0;
end
mlog('[STEP NA] NA=%.1f objNA=%.1f\n', psf_params.NA, psf_params.objNA);
% Override for thin slabs (NAOMi default taillength=50)
psf_params.taillength = min(psf_params.taillength, max(8, vol_sz_um(3)));
mlog('[STEP taillength] %.1f um (slab z=%.1f um)\n', psf_params.taillength, vol_sz_um(3));
% Override for thin slabs (NAOMi default sampling=50)
psf_params.sampling = min(psf_params.sampling, max(2, vol_sz_um(1)/4));
mlog('[STEP sampling] %.1f um\n', psf_params.sampling);
% Override for thin slabs (NAOMi default prop_sz=10)
psf_params.prop_sz = min(psf_params.prop_sz, 5);
mlog('[STEP prop_sz] %.1f um\n', psf_params.prop_sz);
if isfield(S, 'zernikeWt') && ~isempty(S.zernikeWt)
  psf_params.zernikeWt = S.zernikeWt;
elseif strcmp(opt_type, 'standard')
  psf_params.zernikeWt = [0 0 0 0 0.01 0 0 0 0 0 0.02];
end
% blur: check_psf_params default 3um, but simulate_optical_propagation does not output
% opt_out.blur, so setup_scan_volume_frame gets g_blur=[]. Explicitly set 0.
blur_orig = psf_params.blur;
psf_params.blur = 0;
mlog('[STEP blur] psf_params.blur %.1f um -> 0 (disabled; pipeline does not use blur kernel)\n', blur_orig);

vol_params.vasc_sz = gaussianBeamSize(psf_params, vol_params.vol_depth + vol_params.vol_sz(3)/2) ...
  + vol_params.vol_sz + [0 0 1]*vol_params.vol_depth;
vasc_sz_min_z = vol_params.vol_depth + vol_params.vol_sz(3)/2 + psf_params.psf_sz(3)/2;
vol_params.vasc_sz(3) = max(vol_params.vasc_sz(3), vasc_sz_min_z);
% vcpx: anisotropic (vasc_sz in µm / voxel_um per dim)
vcpx = [round(vol_params.vasc_sz(1)/vx), round(vol_params.vasc_sz(2)/vy), round(vol_params.vasc_sz(3)/vz)];
mlog('[STEP vasc_sz] propagation vol [%d,%d,%d] vox (%.1f x %.1f x %.1f um) (anisotropic)\n', vcpx(1), vcpx(2), vcpx(3), vol_params.vasc_sz(1), vol_params.vasc_sz(2), vol_params.vasc_sz(3));

vol_out_prop = struct();
if ~isempty(neur_ves)
  nvpx = size(neur_ves);
  mlog('[STEP vol_out_prop] neur_ves_all size=[%d,%d,%d] vcpx=[%d,%d,%d] nvpx>=vcpx=%d nvpx==vcpx=%d\n', ...
    nvpx(1), nvpx(2), nvpx(3), vcpx(1), vcpx(2), vcpx(3), all(nvpx>=vcpx), isequal(nvpx, vcpx));
  vol_out_prop.neur_ves_all = single(neur_ves);
else
  vol_out_prop.neur_ves_all = zeros(vcpx(1), vcpx(2), vcpx(3), 'single');
  mlog('[STEP vol_out_prop] neur_ves_all zeros(vcpx)\n');
end

if verbose
  fprintf('[Step 1] Optical propagation (simulate_optical_propagation)...\n');
end
mlog('[STEP 1] calling simulate_optical_propagation... (this may take 5-15 min per slab for large vcpx)\n');
tic;
try
  PSF_struct = simulate_optical_propagation(vol_params, psf_params, vol_out_prop);
  if get_opt(S, 'debug_psf', false)
    out_dir = fileparts(output_path);
    debug_path = fullfile(out_dir, 'debug_psf.mat');
    save(debug_path, 'PSF_struct');
  end

catch e
  mlog('[STEP 1] ERROR simulate_optical_propagation: %s\n', e.message);
  mlog('[STEP 1] stack: %s\n', getReport(e, 'extended'));
  if fid_log > 0, fclose(fid_log); end
  rethrow(e);
end
clear vol_out_prop;
t1 = toc;
mlog('[STEP 1] DONE simulate_optical_propagation PSF_struct.psf size=[%d,%d,%d] elapsed=%.1f min\n', size(PSF_struct.psf,1), size(PSF_struct.psf,2), size(PSF_struct.psf,3), t1/60);

PSF_struct.mask = PSF_struct.mask.^1.5;

PSF = PSF_struct.psf;
Np1 = size(PSF, 1);
Np2 = size(PSF, 2);
Np3 = size(PSF, 3);
if Np1*Np2*Np3 <= 1 || any(isnan(PSF(:))) || max(PSF(:)) <= 0
  mlog('[STEP PSF] ERROR: invalid PSF\n');
  err = 1;
  if fid_log > 0, fclose(fid_log); end
  return;
end

if nz < Np3
  warn_append(log_path, 'PAD: volume Z (%d) < PSF Z (%d), padding slab to %d layers (symmetric)\n', nz, Np3, Np3);
  mlog('[STEP] volume Z (%d) < PSF Z (%d), pad slab to %d (symmetric)\n', nz, Np3, Np3);
  pad_total = Np3 - nz;
  pad_before = floor(pad_total / 2);
  pad_after = pad_total - pad_before;
  V = cat(3, zeros(nx, ny, pad_before, 'single'), V, zeros(nx, ny, pad_after, 'single'));
  nz = Np3;
end

vol_out_scan = struct();
vol_out_scan.neur_vol = V;
vol_out_scan.gp_vals = cell(0, 3);
vol_out_scan.bg_proc = cell(1, 2);
vol_out_scan.bg_proc{1, 1} = int32((1:numel(V))');
vol_out_scan.bg_proc{1, 2} = single(V(:));

params = TPM_Simulation_Parameters([], opt_type, condition);
scan_params = params.scan_params;
tpm_params = params.tpm_params;
spike_opts = params.spike_opts;
noise_params = params.noise_params;

if isfield(S, 'scan_avg') && ~isempty(S.scan_avg), scan_params.scan_avg = S.scan_avg(1); end
if isfield(S, 'sfrac') && ~isempty(S.sfrac), scan_params.sfrac = S.sfrac(1); end
% sigscale: lateral area in voxels = vol_sz_um(1)*vol_sz_um(2)/(vx*vy) (anisotropic)
noise_params.sigscale = tpmSignalscale(tpm_params)*(spike_opts.dt)*(scan_params.sfrac^2)/(vol_sz_um(1)*vol_sz_um(2)/(vx*vy));

scan_params = check_scan_params(scan_params);
scan_params.vol_sz = [nx, ny, nz];
scan_params.psf_sz = [Np1, Np2, Np3];
scan_params.verbose = 1;
scan_params.motion = 1;
scan_params.scan_buff = 5;

if verbose
  fprintf('[Step 2] Setup scan volume (setup_scan_volume_frame)...\n');
end
mlog('[STEP 2] calling setup_scan_volume_frame...\n');
try
  scan_vol = setup_scan_volume_frame(vol_out_scan, PSF_struct, scan_params);
catch e
  mlog('[STEP 2] ERROR setup_scan_volume_frame: %s\n', e.message);
  mlog('[STEP 2] stack: %s\n', getReport(e, 'extended'));
  if fid_log > 0, fclose(fid_log); end
  rethrow(e);
end
clear vol_out_scan;
mlog('[STEP 2] DONE setup_scan_volume_frame\n');
mlog('[STEP 2] scan_vol.g_blur empty=%d (no extra lateral blur applied)\n', isempty(scan_vol.g_blur));

% Single-frame neur_act: bg_act=1 for the one bg component
neur_act = struct();
neur_act.soma = zeros(0, 1, 'single');
neur_act.dend = zeros(0, 1, 'single');
neur_act.bg = ones(1, 1, 'single');

if isfield(S, 'rng_seed') && ~isempty(S.rng_seed)
  rng(double(S.rng_seed(1)));
end
if verbose
  fprintf('[Step 3] Scan single frame (scan_volume_frame)...\n');
end
mlog('[STEP 3] calling scan_volume_frame...\n');
try
  [clean_img, ~] = scan_volume_frame(scan_vol, neur_act, scan_params, 0);
catch e
  mlog('[STEP 3] ERROR scan_volume_frame: %s\n', e.message);
  mlog('[STEP 3] stack: %s\n', getReport(e, 'extended'));
  if fid_log > 0, fclose(fid_log); end
  rethrow(e);
end
mlog('[STEP 3] DONE scan_volume_frame clean_img size=[%d,%d]\n', size(clean_img,1), size(clean_img,2));


[out_dir, base_name, ~] = fileparts(output_path);
mlog('[STEP 4] fileparts output_path -> out_dir=%s base_name=%s\n', out_dir, base_name);
if ~isempty(out_dir) && ~exist(out_dir, 'dir')
  mlog('[STEP 4] mkdir out_dir: %s\n', out_dir);
  mkdir(out_dir);
end
mlog('[STEP 4] out_dir exists=%d\n', exist(out_dir, 'dir'));
h5path = fullfile(out_dir, [base_name '.h5']);
mlog('[STEP 4] h5path=%s\n', h5path);
if isfile(h5path), delete(h5path); mlog('[STEP 4] deleted existing h5\n'); end
mlog('[STEP 4] calling h5create...\n');
try
  clean_img = clean_img * noise_params.sigscale;  % scale for noise model (photon units)
  h5create(h5path, '/cleanimg', [size(clean_img,1), size(clean_img,2), 1]);
  mlog('[STEP 4] calling h5write...\n');
  h5write(h5path, '/cleanimg', clean_img, [1 1 1], [size(clean_img,1) size(clean_img,2) 1]);
catch e
  mlog('[STEP 4] ERROR h5: %s\n', e.message);
  mlog('[STEP 4] stack: %s\n', getReport(e, 'extended'));
  if fid_log > 0, fclose(fid_log); end
  rethrow(e);
end
mlog('[STEP 4] DONE h5write isfile=%d\n', isfile(h5path));
if verbose
  fprintf('[Step 4] Saved (HDF5): %s\n', h5path);
end

mlog('[BRIDGE EXIT] err=0, h5path=%s\n', h5path);
if fid_log > 0, fclose(fid_log); end
err = 0;
return;

function mlog_impl(fid, varargin)
  if fid > 0
    fprintf(fid, varargin{1}, varargin{2:end});
  end
  fprintf(1, varargin{1}, varargin{2:end});
end

function v = get_opt(S, f, default)
if isfield(S, f) && ~isempty(S.(f))
  v = S.(f);
else
  v = default;
end
end

end
