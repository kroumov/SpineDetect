function ves_slab = extract_slab_vessels(vessels_full, slab_idx, step, slab_vcpx, log_path)
% EXTRACT_SLAB_VESSELS  Extract vessel slab from full propagation volume.
%
% ves_slab = extract_slab_vessels(vessels_full, slab_idx, step, slab_vcpx, log_path)
% log_path: optional, for debug logging

if nargin < 4
  slab_vcpx = [1164, 1164, 400];
end
if nargin < 5, log_path = ''; end

[vx, vy, vz] = size(vessels_full);
sx = slab_vcpx(1);
sy = slab_vcpx(2);
sz = slab_vcpx(3);

z_start = slab_idx * step;
z_end = min(z_start + sz, vz);

ves_slab = zeros(slab_vcpx, 'single');
n = z_end - z_start;
log_append(log_path, '[EXTRACT_SLAB_VESSELS] IN slab_idx=%d step=%d slab_vcpx=[%d,%d,%d] vessels_full=[%d,%d,%d] z_start=%d z_end=%d n=%d\n', ...
  slab_idx, step, slab_vcpx(1), slab_vcpx(2), slab_vcpx(3), vx, vy, vz, z_start, z_end, n);
if n <= 0
  return;
end

if n < sz
  warn_append(log_path, 'PAD: slab %d vessel z layers %d < slab_vcpx(3)=%d, zero-padding %d layers (symmetric)\n', slab_idx+1, n, sz, sz-n);
end

tz = floor((sz - n) / 2);
if vx >= sx && vy >= sy
  % Larger: symmetric center crop
  cx = floor((vx - sx) / 2) + 1;
  cy = floor((vy - sy) / 2) + 1;
  src = vessels_full(cx:cx+sx-1, cy:cy+sy-1, z_start+1:z_end);
  ves_slab(:,:,1+tz:tz+n) = src;
else
  % Smaller: symmetric center placement (pad both sides)
  warn_append(log_path, 'FALLBACK: vessels [%d,%d] < slab_vcpx [%d,%d], symmetric center pad\n', vx, vy, sx, sy);
  mx = min(sx, vx);
  my = min(sy, vy);
  tx = floor((sx - mx) / 2);
  ty = floor((sy - my) / 2);
  tz = floor((sz - n) / 2);
  ves_slab(1+tx:tx+mx, 1+ty:ty+my, 1+tz:tz+n) = vessels_full(1:mx, 1:my, z_start+1:z_end);
end
log_append(log_path, '[EXTRACT_SLAB_VESSELS] OUT ves_slab size=[%d,%d,%d]\n', size(ves_slab,1), size(ves_slab,2), size(ves_slab,3));
end
