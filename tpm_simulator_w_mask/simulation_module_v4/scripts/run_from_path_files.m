function run_from_path_files(manifest_path)
% RUN_FROM_PATH_FILES  Read paths from a UTF-8 text file and call main().
%
% Avoids passing paths with non-ASCII characters via
% matlab -batch command line, which can cause encoding issues.
%
% manifest_path: path to a text file (relative to cwd or absolute).
%   Line 1: chunk_folder_path (absolute path to input folder)
%   Line 2: output_dir (absolute path to output base dir)
%   Line 3: log_path (absolute path to log file)

if nargin < 1 || isempty(manifest_path)
  error('run_from_path_files:NoArg', 'Usage: run_from_path_files(manifest_path)');
end

script_dir = fileparts(mfilename('fullpath'));
module_dir = fileparts(script_dir);
addpath(script_dir);
addpath(module_dir);

% Resolve manifest path (relative to module dir)
if ~contains(manifest_path, filesep) && (~ispc || numel(manifest_path) < 2 || manifest_path(2) ~= ':')
  manifest_path = fullfile(module_dir, manifest_path);
end

if ~isfile(manifest_path)
  error('run_from_path_files:NoFile', 'Manifest file not found: %s', manifest_path);
end

% Read with UTF-8 encoding
try
  fid = fopen(manifest_path, 'r', 'n', 'UTF-8');
  if fid < 0
    error('run_from_path_files:Open', 'Cannot open manifest: %s', manifest_path);
  end
  c = fread(fid, '*char')';
  fclose(fid);
catch e
  error('run_from_path_files:Read', 'Failed to read manifest: %s', e.message);
end

lines = strsplit(strtrim(c), {'\r\n', '\n', '\r'}, 'CollapseDelimiters', true);
lines = lines(~cellfun(@isempty, lines));

if numel(lines) < 3
  error('run_from_path_files:Format', 'Manifest must have at least 3 lines (chunk_path, output_dir, log_path)');
end

chunk_path = strtrim(lines{1});
output_dir = strtrim(lines{2});
log_path   = strtrim(lines{3});

opts = tpm_config();
opts.output_dir = output_dir;
opts.log_path   = log_path;

main(chunk_path, opts);
end
