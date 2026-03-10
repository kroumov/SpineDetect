function process_one(input_path, output_path, opts)
% PROCESS_ONE  Slice volume into slabs, run bridge per slab, stitch output.
%
% process_one(input_path, output_path, opts)
% input_path: .tif file or folder (neurons + optional vessels)
% output_path: output .tif path
% opts: from tpm_config(), may override fields

if nargin < 3
  opts = tpm_config();
end

base_dir = fileparts(fileparts(mfilename('fullpath')));
input_path = char(input_path);
output_path = char(output_path);
log_path = get_opt(opts, 'log_path', '');

log_append(log_path, '[PROCESS_ONE] ENTER input_path=%s output_path=%s\n', input_path, output_path);

% Load volume
if isfile(input_path)
  neurons = load_volume_tiff(input_path, log_path);
  vessels = [];
else
  [neurons, vessels] = load_input_folder(input_path, true, opts.verbose, log_path);
end
log_append(log_path, '[PROCESS_ONE] load neurons size=[%d,%d,%d] vessels_empty=%d\n', ...
  size(neurons,1), size(neurons,2), size(neurons,3), isempty(vessels));

% Normalize
neurons = single(neurons);
if max(neurons(:)) > 1
  neurons = neurons / 255.0;
end
fluor_scale = get_opt(opts, 'fluor_scale', 0.8);
neurons = max(0, min(1, neurons)) * fluor_scale;

% Neuropil (background fluorescence)
if get_opt(opts, 'neuropil_enable', false)
  base_dir = fileparts(fileparts(mfilename('fullpath')));
  addpath(fullfile(base_dir, 'naomi', 'VolumeCode'));
  addpath(fullfile(base_dir, 'naomi', 'MiscCode'));
  neurons = add_neuropil(neurons, ...
    get_opt(opts, 'neuropil_scale', 0.08), ...
    get_opt(opts, 'neuropil_thresh', 0.05), ...
    [], log_path);
  log_append(log_path, '[PROCESS_ONE] neuropil added\n');
end

% Pad XY (edge replicate) before degrade; crop same amount after (pad_xy=0 disables)
pad_xy = get_opt(opts, 'pad_xy', 4);
orig_sz = size(neurons);
if pad_xy > 0
  neurons = padarray(neurons, [pad_xy, pad_xy, 0], 'replicate');
  log_append(log_path, '[PROCESS_ONE] pad_xy=%d: padded to [%d,%d,%d]\n', pad_xy, size(neurons,1), size(neurons,2), size(neurons,3));
end

% Voxel size and slab geometry
voxel_um = get_opt(opts, 'voxel_um', [0.064, 0.064, 0.32]);
if numel(voxel_um) == 1
  voxel_um = [voxel_um, voxel_um, voxel_um];
end
vol_depth_um = get_opt(opts, 'vol_depth_um', 8.0);
slab_z = get_opt(opts, 'slab_z', 32);
focal_spacing = get_opt(opts, 'focal_spacing', 3);

nz = size(neurons, 3);
ranges = get_slab_z_ranges(nz, slab_z, focal_spacing, log_path);
if isempty(ranges)
  error('process_one:VolumeTooSmall', 'Volume Z=%d too small for slab %d', nz, slab_z);
end
% Test: max_slabs limits slabs; max_slabs_start = first slab index (1-based)
max_slabs_opt = get_opt(opts, 'max_slabs', 'none');
max_slabs_start_opt = get_opt(opts, 'max_slabs_start', 1);
if ~isempty(max_slabs_opt) && ~(ischar(max_slabs_opt) && strcmpi(max_slabs_opt, 'none'))
  nlim = NaN;
  if isnumeric(max_slabs_opt), nlim = double(max_slabs_opt(1)); elseif ischar(max_slabs_opt), nlim = str2double(max_slabs_opt); end
  start_idx = 1;
  if ~isempty(max_slabs_start_opt) && ~(ischar(max_slabs_start_opt) && strcmpi(max_slabs_start_opt, 'none'))
    if isnumeric(max_slabs_start_opt), start_idx = max(1, round(double(max_slabs_start_opt(1)))); elseif ischar(max_slabs_start_opt), start_idx = max(1, round(str2double(max_slabs_start_opt))); end
  end
  if ~isnan(nlim) && nlim > 0
    end_idx = min(start_idx + nlim - 1, size(ranges, 1));
    if start_idx <= size(ranges, 1)
      ranges = ranges(start_idx:end_idx, :);
      log_append(log_path, '[PROCESS_ONE] max_slabs=%g max_slabs_start=%d: slabs %d-%d (%d total)\n', nlim, start_idx, start_idx, end_idx, size(ranges, 1));
    end
  end
end
last_z_end = ranges(end, 2);
if last_z_end < nz
  warn_append(log_path, 'CROP: volume Z=%d, last slab ends at %d, discarding layers %d-%d\n', nz, last_z_end, last_z_end+1, nz);
end

[vcpx, ~, ~] = compute_vcpx(voxel_um, vol_depth_um, size(neurons,1), size(neurons,2), slab_z, ...
  get_opt(opts, 'psf_sz', []), get_opt(opts, 'opt_type', 'standard'), get_opt(opts, 'condition', 'standard'), log_path, opts);
slab_vcpx = vcpx;
vcpx_xy = vcpx(1);
vcpx_z_full = (size(ranges, 1) - 1) * focal_spacing + vcpx(3);
log_append(log_path, '[PROCESS_ONE] vcpx=[%d,%d,%d] vcpx_z_full=%d num_slabs=%d\n', ...
  vcpx(1), vcpx(2), vcpx(3), vcpx_z_full, size(ranges, 1));

% Vessels: crop/pad to match vcpx when size differs
if ~isempty(vessels)
  vessels = single(vessels);
  if max(vessels(:)) > 1
    vessels = vessels / 255.0;
  end
  vessels = max(0, min(1, vessels));
  tgt = [vcpx_xy, vcpx_xy, vcpx_z_full];
  if ~isequal(size(vessels), tgt)
    vx = size(vessels, 1); vy = size(vessels, 2); vz = size(vessels, 3);
    if all([vx,vy,vz] >= tgt)
      % Larger: symmetric center crop
      d = floor(([vx,vy,vz] - tgt) / 2);
      vessels = vessels(1+d(1):d(1)+tgt(1), 1+d(2):d(2)+tgt(2), 1+d(3):d(3)+tgt(3));
      warn_append(log_path, 'FALLBACK: vessels [%d,%d,%d] > vcpx [%d,%d,%d], center-cropped\n', vx, vy, vz, tgt(1), tgt(2), tgt(3));
    elseif all([vx,vy,vz] <= tgt)
      % Smaller: symmetric zero-pad
      d = floor((tgt - [vx,vy,vz]) / 2);
      tmp = zeros(tgt, 'single');
      tmp(1+d(1):d(1)+vx, 1+d(2):d(2)+vy, 1+d(3):d(3)+vz) = vessels;
      vessels = tmp;
      warn_append(log_path, 'FALLBACK: vessels [%d,%d,%d] < vcpx [%d,%d,%d], zero-padded\n', vx, vy, vz, tgt(1), tgt(2), tgt(3));
    else
      % Mixed: crop larger dims, pad smaller dims
      tmp = zeros(tgt, 'single');
      rx = min(vx, tgt(1)); ry = min(vy, tgt(2)); rz = min(vz, tgt(3));
      dx = floor((vx - rx) / 2); dy = floor((vy - ry) / 2); dz = floor((vz - rz) / 2);
      tx = floor((tgt(1) - rx) / 2); ty = floor((tgt(2) - ry) / 2); tz = floor((tgt(3) - rz) / 2);
      tmp(1+tx:tx+rx, 1+ty:ty+ry, 1+tz:tz+rz) = vessels(1+dx:dx+rx, 1+dy:dy+ry, 1+dz:dz+rz);
      vessels = tmp;
      warn_append(log_path, 'FALLBACK: vessels [%d,%d,%d] ~= vcpx [%d,%d,%d], crop+pad\n', vx, vy, vz, tgt(1), tgt(2), tgt(3));
    end
  end
end

seed_base = path_to_seed(input_path, log_path);
[out_dir, stem, ext] = fileparts(output_path);
if isempty(out_dir)
  output_dir = fullfile(base_dir, 'output');
elseif ~contains(out_dir, filesep) || (ispc && ~contains(out_dir, ':'))
  output_dir = fullfile(base_dir, out_dir);
else
  output_dir = out_dir;
end
if ~isfolder(output_dir)
  mkdir(output_dir);
end

frames = {};
for i = 1:size(ranges, 1)
  z0 = ranges(i, 1);
  z1 = ranges(i, 2);
  if opts.verbose
    fprintf('[Step] Slab %d/%d: z=%d-%d\n', i, size(ranges, 1), z0, z1);
  end

  neur_slab = neurons(:, :, z0+1:z1);
  if ~isempty(vessels)
    ves_slab = extract_slab_vessels(vessels, i-1, focal_spacing, slab_vcpx, log_path);
  else
    ves_slab = [];
  end

  log_append(log_path, '[PROCESS_ONE] slab %d/%d z=[%d,%d) neur_slab=[%d,%d,%d] ves_slab_empty=%d\n', ...
    i, size(ranges, 1), z0, z1, size(neur_slab,1), size(neur_slab,2), size(neur_slab,3), isempty(ves_slab));

  slab_out = fullfile(output_dir, sprintf('%s_slab%03d.tif', stem, i-1));
  mat_path = [tempname, '.mat'];

  mat_dict = struct();
  mat_dict.neur_vol = neur_slab;
  mat_dict.neur_ves = ves_slab;
  mat_dict.voxel_um = voxel_um;
  mat_dict.vol_depth_um = vol_depth_um;
  mat_dict.output_path = slab_out;
  mat_dict.verbose = false;
  mat_dict.opt_type = get_opt(opts, 'opt_type', 'standard');
  mat_dict.condition = get_opt(opts, 'condition', 'standard');
  mat_dict.rng_seed = seed_base + i - 1;
  mat_dict.log_path = log_path;

  if isfield(opts, 'scan_avg') && ~isempty(opts.scan_avg)
    mat_dict.scan_avg = opts.scan_avg(1);
  end
  if isfield(opts, 'sfrac') && ~isempty(opts.sfrac)
    mat_dict.sfrac = opts.sfrac(1);
  end
  if isfield(opts, 'psf_sz') && ~isempty(opts.psf_sz)
    mat_dict.psf_sz = opts.psf_sz;
  end
  if isfield(opts, 'NA') && ~isempty(opts.NA)
    mat_dict.NA = opts.NA(1);
  end
  if isfield(opts, 'objNA') && ~isempty(opts.objNA)
    mat_dict.objNA = opts.objNA(1);
  end
  if isfield(opts, 'zernikeWt') && ~isempty(opts.zernikeWt)
    mat_dict.zernikeWt = opts.zernikeWt;
  end
  if get_opt(opts, 'debug_psf', false)
    mat_dict.debug_psf = true;
  end

  save(mat_path, '-struct', 'mat_dict', '-v7');
  try
    err = naomi_bridge(mat_path);
    if err ~= 0
      error('naomi_bridge slab %d returned error', i);
    end
  finally
    if isfile(mat_path)
      delete(mat_path);
    end
  end

  % Bridge writes .h5; read and convert to frame for stitching
  h5_path = [slab_out(1:end-4), '.h5'];
  if isfile(h5_path)
    fr = h5read(h5_path, '/cleanimg');
    fr = squeeze(fr);
    if ndims(fr) ~= 2
      fr = fr(:, :, 1);
    end
    if pad_xy > 0
      fr = fr(pad_xy+1:end-pad_xy, pad_xy+1:end-pad_xy);
    end
    frames{end+1} = single(fr);  %#ok<AGROW>
    delete(h5_path);
  else
    warn_append(log_path, 'FALLBACK: slab %d output not found: %s\n', i, h5_path);
  end
end

if isempty(frames)
  error('process_one:NoOutput', 'No slab outputs produced');
end

mov = cat(3, frames{:});
[out_pardir, out_name, out_ext] = fileparts(output_path);
if isempty(out_pardir)
  out_path = fullfile(output_dir, [out_name, out_ext]);
else
  out_path = output_path;
end
if ~isfolder(fileparts(out_path))
  mkdir(fileparts(out_path));
end

% Compute metadata: XY resolution um/px; TIFF typically uses Inch (2) for ResolutionUnit.
res_um_xy = voxel_um(1);
res_um_z = focal_spacing * voxel_um(3);
% TIFF stores pixels per unit (density); pixel_per_unit = 1/res_um_xy (px/um)
pixel_per_unit = 1 / res_um_xy;

t = Tiff(out_path, 'w');
tagstruct.ImageLength = size(mov, 1);
tagstruct.ImageWidth = size(mov, 2);
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32;
tagstruct.SamplesPerPixel = 1;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Compression = Tiff.Compression.None;

% Physical metadata (TIFF: XResolution/YResolution are Rational type)
tagstruct.ResolutionUnit = Tiff.ResolutionUnit.Inch;
tagstruct.XResolution = pixel_per_unit;
tagstruct.YResolution = pixel_per_unit;

% ImageJ/Napari-compatible Z axis and unit description
imgDescription = sprintf('ImageJ=1.53\nunit=um\nspacing=%.4f\nloop=false\nvoxel_um=[%.4f,%.4f,%.4f]', ...
  res_um_z, voxel_um(1), voxel_um(2), voxel_um(3));
tagstruct.ImageDescription = imgDescription;

for k = 1:size(mov, 3)
  t.setTag(tagstruct);
  t.write(mov(:, :, k));
  if k < size(mov, 3)
    t.writeDirectory();
  end
end
t.close();

% Write degrade_info.txt: chunk_info + NAOMi params (use orig_sz for output dimensions)
vol_sz_um = [orig_sz(1)*voxel_um(1), orig_sz(2)*voxel_um(2), orig_sz(3)*voxel_um(3)];
chunk_info_path = fullfile(input_path, 'chunk_info.txt');
degrade_info_path = fullfile(output_dir, 'degrade_info.txt');
chunk_content = '';
if isfolder(input_path) && isfile(chunk_info_path)
  fid = fopen(chunk_info_path, 'r');
  if fid >= 0
    chunk_content = fread(fid, '*char')';
    fclose(fid);
  end
end
psf_sz = get_opt(opts, 'psf_sz', [3, 3, 6]);
if numel(psf_sz) < 3, psf_sz = [psf_sz(1), psf_sz(min(2,end)), 6]; end
opt_type = get_opt(opts, 'opt_type', 'standard');
opt_type = char(opt_type); if iscell(opt_type), opt_type = char(opt_type{1}); end
condition = get_opt(opts, 'condition', 'standard');
condition = char(condition); if iscell(condition), condition = char(condition{1}); end
taillength_eff = min(50, max(8, vol_sz_um(3)));
sampling_eff = min(50, max(2, vol_sz_um(1)/4));
zernikeWt = get_opt(opts, 'zernikeWt', [0 0 0 0 0.01 0 0 0 0 0 0.02]);
zernike_str = mat2str(zernikeWt);
if ~strcmp(opt_type, 'standard')
  zernike_str = '[TPM_Simulation_Parameters default]';
end
na_val = get_opt(opts, 'NA', 1.0);
objna_val = get_opt(opts, 'objNA', 1.0);
naomi_section = sprintf([
  '\n--- NAOMi Degrade Info ---\n' ...
  'Timestamp:         %s\n' ...
  '\n[Input & Geometry]\n' ...
  'Input Size:        %dx%dx%d px\n' ...
  'Voxel Size:        [%.3g, %.3g, %.3g] um\n' ...
  'vol_sz_um:         [%.2f, %.2f, %.2f] um\n' ...
  'vol_depth_um:      %.2f um\n' ...
  'Slab Size:         %dx%dx%d px\n' ...
  'Focal Spacing:     %d px\n' ...
  'Num Slabs:         %d\n' ...
  'vcpx:              [%d, %d, %d]\n' ...
  'vcpx_z_full:       %d\n' ...
  '\n[PSF & Optics - tpm_config]\n' ...
  'PSF Size:          [%.1f, %.1f, %.1f] um\n' ...
  'opt_type:          %s\n' ...
  'condition:         %s\n' ...
  '\n[PSF & Optics - naomi_bridge overrides]\n' ...
  'NA:                %.1f\n' ...
  'objNA:             %.1f\n' ...
  'taillength:        %.1f um (min(50, max(8, vol_sz_um(3))))\n' ...
  'sampling:          %.1f um (min(50, max(2, vol_sz_um(1)/4)))\n' ...
  'prop_sz:           5 um\n' ...
  'blur:              0 um (disabled)\n' ...
  'zernikeWt:         %s\n' ...
  'mask_exponent:     1.5\n' ...
  'propcrop:          true\n' ...
  '\n[PSF - check_psf_params defaults]\n' ...
  'lambda:            0.92 um\n' ...
  'n:                 1.35\n' ...
  'obj_fl:            4.5 mm\n' ...
  'ss:                2\n' ...
  'fastmask:          true\n' ...
  '\n[Scan & Preprocessing]\n' ...
  'pad_xy:            %d\n' ...
  'max_slabs:         %s\n' ...
  'max_slabs_start:   %s\n' ...
  'scan_avg:          %d\n' ...
  'sfrac:             %g\n' ...
  'scan_buff:         5\n' ...
  'motion:            1\n' ...
  'fluor_scale:       %.2g\n' ...
  'neuropil_enable:   %d\n' ...
  'neuropil_scale:    %.2g\n' ...
  'neuropil_thresh:   %.2g\n' ...
  '\n[Output]\n' ...
  'Output Resolution: XY %.3g um/px, Z %.3g um/frame\n' ...
  '========================================\n' ...
  ], datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
  orig_sz(1), orig_sz(2), orig_sz(3), ...
  voxel_um(1), voxel_um(2), voxel_um(3), ...
  vol_sz_um(1), vol_sz_um(2), vol_sz_um(3), ...
  vol_depth_um, size(neurons,1), size(neurons,2), slab_z, ...
  focal_spacing, size(ranges, 1), vcpx(1), vcpx(2), vcpx(3), vcpx_z_full, ...
  psf_sz(1), psf_sz(2), psf_sz(3), opt_type, condition, ...
  na_val, objna_val, taillength_eff, sampling_eff, zernike_str, ...
  pad_xy, ...
  char(string(get_opt(opts, 'max_slabs', 'none'))), ...
  char(string(get_opt(opts, 'max_slabs_start', 1))), ...
  get_opt(opts, 'scan_avg', 1), get_opt(opts, 'sfrac', 1), ...
  fluor_scale, ...
  double(get_opt(opts, 'neuropil_enable', false)), ...
  get_opt(opts, 'neuropil_scale', 0.08), get_opt(opts, 'neuropil_thresh', 0.05), ...
  res_um_xy, res_um_z);
fid = fopen(degrade_info_path, 'w');
if fid >= 0
  fprintf(fid, '%s%s', strtrim(chunk_content), naomi_section);
  fclose(fid);
  log_append(log_path, '[PROCESS_ONE] wrote degrade_info.txt\n');
end

for i = 1:size(ranges, 1)
  slab_f = fullfile(output_dir, sprintf('%s_slab%03d.tif', stem, i-1));
  if isfile(slab_f)
    delete(slab_f);
  end
end

log_append(log_path, '[PROCESS_ONE] EXIT OK out_path=%s frames=%d\n', out_path, size(mov, 3));
if opts.verbose
  fprintf('Saved: %s (%d frames)\n', out_path, size(mov, 3));
end
end

function v = get_opt(s, f, default)
if isfield(s, f) && ~isempty(s.(f))
  v = s.(f);
else
  v = default;
end
end
