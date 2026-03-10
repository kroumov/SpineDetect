function [neurons_path, vessels_path] = find_neurons_vessels_paths(folder_path, log_path)
% FIND_NEURONS_VESSELS_PATHS  Find neurons and vessels TIFF in folder.
%
% [neurons_path, vessels_path] = find_neurons_vessels_paths(folder_path, log_path)
% log_path: optional, for debug logging

if nargin < 2, log_path = ''; end

folder_path = char(folder_path);
log_append(log_path, '[FIND_NEURONS_VESSELS_PATHS] IN folder_path=%s\n', folder_path);
if ~isfolder(folder_path)
  error('find_neurons_vessels_paths:NotDir', 'Not a directory: %s', folder_path);
end

d = dir(fullfile(folder_path, '*.tif'));
d = [d; dir(fullfile(folder_path, '*.tiff'))];
paths = unique(fullfile({d.folder}, {d.name}));

neurons_path = [];
vessels_path = [];
for i = 1:numel(paths)
  [~, stem, ~] = fileparts(paths{i});
  s = lower(stem);
  if contains(s, 'vessel')
    vessels_path = paths{i};
  elseif contains(s, 'neuron')
    neurons_path = paths{i};
  end
end
if isempty(neurons_path) && numel(paths) == 1
  neurons_path = paths{1};
end
log_append(log_path, '[FIND_NEURONS_VESSELS_PATHS] OUT neurons_path=%s vessels_path=%s num_files=%d\n', ...
  char(neurons_path), char(vessels_path), numel(paths));
end
