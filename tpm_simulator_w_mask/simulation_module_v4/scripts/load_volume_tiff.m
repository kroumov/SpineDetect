function vol = load_volume_tiff(path, log_path)
% LOAD_VOLUME_TIFF  Load 3D volume from TIFF file. Returns (H,W,Z) single.
%
% vol = load_volume_tiff(path, log_path)
% log_path: optional, for debug logging

if nargin < 2, log_path = ''; end

path = char(path);
log_append(log_path, '[LOAD_VOLUME_TIFF] IN path=%s\n', path);
if ~isfile(path)
  error('load_volume_tiff:NotFound', 'File not found: %s', path);
end

info = imfinfo(path);
n = numel(info);
h = info(1).Height;
w = info(1).Width;

vol = zeros(h, w, n, 'single');
for k = 1:n
  vol(:,:,k) = single(imread(path, k, 'Info', info));
end

if max(vol(:)) > 1
  vol = vol / 255.0;
end
vol = max(0, min(1, vol));

% TIFF stack: (H,W,N) = (Y,X,Z), third dim is Z. Do NOT permute when h < n,
% else 128x128x256 becomes 128x256x128 and nz wrongly becomes 128.
log_append(log_path, '[LOAD_VOLUME_TIFF] OUT size=[%d,%d,%d] class=%s\n', size(vol,1), size(vol,2), size(vol,3), class(vol));
end
