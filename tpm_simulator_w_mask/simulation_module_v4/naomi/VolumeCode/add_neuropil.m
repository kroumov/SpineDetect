function neur_vol = add_neuropil(neur_vol, neuropil_scale, neuropil_thresh, l_scale, log_path)
% ADD_NEUROPIL  Add neuropil/background fluorescence to a volume using 3D GP.
%
% neur_vol = add_neuropil(neur_vol, neuropil_scale, neuropil_thresh, l_scale, log_path)
%
% Adds spatially correlated background (neuropil) to neur_vol. Uses
% masked_3DGP_test (NAOMi) to generate correlated texture in background
% regions only. Simulates non-specific fluorescence from axons, dendrites,
% and extracellular matrix.
%
% Inputs:
%   neur_vol        - 3D single (H,W,Z), base fluorescence
%   neuropil_scale  - (optional) Max neuropil level as fraction of neur_vol max (default 0.08)
%   neuropil_thresh - (optional) Background = voxels below this fraction of max (default 0.05)
%   l_scale         - (optional) GP length scale [x,y,z] in voxels (default [6,6,2])
%   log_path        - (optional) path for debug log
%
% Output:
%   neur_vol        - neur_vol + neuropil (clipped to non-negative)
%
% 2026 - tpm_simulator NAOMi integration

if nargin < 2 || isempty(neuropil_scale)
  neuropil_scale = 0.08;
end
if nargin < 3 || isempty(neuropil_thresh)
  neuropil_thresh = 0.05;
end
if nargin < 4 || isempty(l_scale)
  l_scale = [6, 6, 2];  % XY more correlated, Z less (anisotropic)
end
if nargin < 5
  log_path = '';
end

grid_sz = size(neur_vol);
vmax = max(neur_vol(:));
if vmax <= 0
  return;
end

% Background mask: voxels below threshold
thresh_val = neuropil_thresh * vmax;
bg_mask = single(neur_vol < thresh_val);

% Generate correlated noise via 3D Gaussian process (NAOMi masked_3DGP_test)
% mu=0, p_scale small; we scale afterward
mu = 0;
p_scale = 0.15;  % GP variance (relative)
bin_mask = bg_mask;
threshold = 1e-10;
l_weights = 1;

try
  neuropil = masked_3DGP_test(grid_sz, l_scale, p_scale, mu, bin_mask, threshold, l_weights, 'single');
catch e
  if ~isempty(log_path) && exist('warn_append', 'file')
    warn_append(log_path, 'ADD_NEUROPIL: masked_3DGP_test failed: %s, skipping\n', e.message);
  end
  return;
end

% Clip to non-negative, scale to desired level
neuropil = max(0, neuropil);
np_max = max(neuropil(:));
if np_max > 0
  target_max = neuropil_scale * vmax;
  neuropil = neuropil * (target_max / np_max);
end

% Add only in background (avoid double-counting neurons)
neur_vol = neur_vol + neuropil .* bg_mask;
neur_vol = max(0, neur_vol);

if ~isempty(log_path) && exist('log_append', 'file')
  log_append(log_path, '[ADD_NEUROPIL] scale=%.3g thresh=%.3g l_scale=%s np_max=%.4g target=%.4g\n', ...
    neuropil_scale, neuropil_thresh, mat2str(l_scale), np_max, neuropil_scale * vmax);
end
end
