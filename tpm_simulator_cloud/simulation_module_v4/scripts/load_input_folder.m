function [neurons, vessels] = load_input_folder(folder_path, require_neurons, verbose, log_path)
% LOAD_INPUT_FOLDER  Load neurons and optional vessels from folder.
%
% [neurons, vessels] = load_input_folder(folder_path, require_neurons, verbose, log_path)
% log_path: optional, for debug logging

if nargin < 2, require_neurons = true; end
if nargin < 3, verbose = true; end
if nargin < 4, log_path = ''; end

log_append(log_path, '[LOAD_INPUT_FOLDER] IN folder_path=%s require_neurons=%d\n', folder_path, require_neurons);
[neurons_path, vessels_path] = find_neurons_vessels_paths(folder_path, log_path);
log_append(log_path, '[LOAD_INPUT_FOLDER] neurons_path=%s vessels_path=%s\n', ...
  char(neurons_path), char(vessels_path));
if isempty(neurons_path) && require_neurons
  error('load_input_folder:NoNeurons', 'No neurons TIFF (filename containing ''neuron'') in %s', folder_path);
end

neurons = [];
vessels = [];
if ~isempty(neurons_path)
  if verbose
    fprintf('  [Neurons] %s\n', char(neurons_path));
  end
  neurons = load_volume_tiff(neurons_path, log_path);
end
if ~isempty(vessels_path)
  if verbose
    fprintf('  [Vessels] %s\n', char(vessels_path));
  end
  vessels = load_volume_tiff(vessels_path, log_path);
else
  vessels = [];
end
log_append(log_path, '[LOAD_INPUT_FOLDER] OUT neurons size=[%d,%d,%d] vessels_empty=%d\n', ...
  size(neurons,1), size(neurons,2), size(neurons,3), isempty(vessels));
end
