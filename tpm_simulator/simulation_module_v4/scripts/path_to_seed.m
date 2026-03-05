function seed = path_to_seed(path, log_path)
% PATH_TO_SEED  Deterministic seed from path for reproducible rng.
%
% seed = path_to_seed(path, log_path)
% log_path: optional, for debug logging

if nargin < 2, log_path = ''; end

path = char(path);
path = fullfile(path);
h = uint32(0);
for i = 1:numel(path)
  h = mod(h * 31 + uint32(path(i)), 2^32);
end
seed = mod(double(h), 2^31 - 1);
if seed == 0
  seed = 1;
end
log_append(log_path, '[PATH_TO_SEED] IN path=%s OUT seed=%d\n', path, seed);
end
