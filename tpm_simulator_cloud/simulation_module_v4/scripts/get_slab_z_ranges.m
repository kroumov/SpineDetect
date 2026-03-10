function ranges = get_slab_z_ranges(nz, slab_z, step, log_path)
% GET_SLAB_Z_RANGES  Return slab Z ranges for overlapping slabs.
%
% ranges = get_slab_z_ranges(nz, slab_z, step, log_path)
% log_path: optional, for debug logging

if nargin < 2, slab_z = 200; end
if nargin < 3, step = 20; end
if nargin < 4, log_path = ''; end

ranges = [];
z = 0;
while z + slab_z <= nz
  ranges(end+1, :) = [z, z + slab_z];  %#ok<AGROW>
  z = z + step;
end
log_append(log_path, '[GET_SLAB_Z_RANGES] IN nz=%d slab_z=%d step=%d OUT num_slabs=%d\n', nz, slab_z, step, size(ranges, 1));
end
