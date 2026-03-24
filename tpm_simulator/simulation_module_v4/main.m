function out = main(folders, opts)
% MAIN  Run TPM simulation on folders under input/.
%
% Usage:
%   main()
%       -> process all folders under input/
%
%   main('microns_864691136811782003')
%   main({'microns_864691136811782003', 'microns_xxx'})
%       -> process specified folder(s) under input/
%
%   main(folders, opts)
%       -> with options (from tpm_config())
%
% Requires: naomi/ (standalone, no external NAOMi).

base_dir = fileparts(mfilename('fullpath'));
addpath(base_dir);
addpath(fullfile(base_dir, 'scripts'));
input_dir = fullfile(base_dir, 'input');
output_dir = fullfile(base_dir, 'output');
logs_dir = fullfile(base_dir, 'logs');

if nargin < 2
  opts = tpm_config();
end
if isfield(opts, 'output_dir') && ~isempty(opts.output_dir)
  output_dir = opts.output_dir;
  if ~contains(output_dir, filesep) && (~ispc || numel(output_dir) < 2 || output_dir(2) ~= ':')
    output_dir = fullfile(base_dir, output_dir);
  end
end
mkdir(output_dir);
mkdir(logs_dir);

if nargin < 1 || isempty(folders)
  % Process all folders under input/
  if ~isfolder(input_dir)
    error('main:NoInput', 'Input folder not found: %s', input_dir);
  end
  d = dir(input_dir);
  folders = {};
  for i = 1:numel(d)
    if d(i).isdir && ~strcmp(d(i).name, '.') && ~strcmp(d(i).name, '..')
      folders{end+1} = d(i).name;  %#ok<AGROW>
    end
  end
else
  if ischar(folders)
    folders = {folders};
  end
end

if isempty(folders)
  fprintf('No folders to process. Put subfolders with *neuron*.tif under input/\n');
  out = {};
  return;
end

out = {};
for i = 1:numel(folders)
  folder_name = folders{i};
  % Support absolute path or path under input/
  if contains(folder_name, filesep) || (ispc && numel(folder_name) >= 2 && folder_name(2) == ':')
    folder_path = folder_name;
  else
    folder_path = fullfile(input_dir, folder_name);
  end
  if ~isfolder(folder_path)
    fprintf('Skip (not found): %s\n', folder_path);
    continue;
  end
  [neurons_path, ~] = find_neurons_vessels_paths(folder_path);
  if isempty(neurons_path)
    fprintf('Skip (no neurons): %s\n', folder_path);
    continue;
  end
  [~, stem, ~] = fileparts(neurons_path);
  [~, folder_short, ~] = fileparts(folder_path);
  out_subdir = fullfile(output_dir, folder_short);
  mkdir(out_subdir);
  out_path = fullfile(out_subdir, [stem '.tiff']);
  ts = datestr(now, 'yyyymmdd_HHMMSS');
  if isfield(opts, 'log_path') && ~isempty(opts.log_path)
    log_path = opts.log_path;
  else
    log_path = fullfile(logs_dir, sprintf('log_%s_%s.log', folder_short, ts));
  end
  opts_run = opts;
  opts_run.log_path = log_path;
  try
    log_append(log_path, '[MAIN] ENTER folder=%s out_path=%s\n', folder_path, out_path);
    process_one(folder_path, out_path, opts_run);
    out{end+1} = out_path;  %#ok<AGROW>
    log_append(log_path, '[MAIN] EXIT OK out_path=%s\n', out_path);
    fprintf('Saved: %s\n', out_path);
  catch e
    log_append(log_path, '[MAIN] EXIT ERROR %s\n', e.message);
    fprintf('Error processing %s: %s\n', folder_path, e.message);
  end
end
end
