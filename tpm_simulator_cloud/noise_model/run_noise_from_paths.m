% RUN_NOISE_FROM_PATHS  Apply noise to a single TIFF, paths from manifest.
%
% run_noise_from_paths(manifest_path)
% manifest_path: path to UTF-8 text file.
%   Line 1: input_tiff_path (absolute)
%   Line 2: output_dir (absolute, write same-named tiff here)

function run_noise_from_paths(manifest_path)
base = fileparts(mfilename('fullpath'));
addpath(base);

if nargin < 1 || isempty(manifest_path)
  error('run_noise_from_paths:NoArg', 'Usage: run_noise_from_paths(manifest_path)');
end

if ~isfile(manifest_path)
  error('run_noise_from_paths:NoFile', 'Manifest not found: %s', manifest_path);
end

fid = fopen(manifest_path, 'r', 'n', 'UTF-8');
if fid < 0
  error('run_noise_from_paths:Open', 'Cannot open manifest');
end
c = fread(fid, '*char')';
fclose(fid);

lines = strsplit(strtrim(c), {'\r\n', '\n', '\r'}, 'CollapseDelimiters', true);
lines = lines(~cellfun(@isempty, lines));
if numel(lines) < 2
  error('run_noise_from_paths:Format', 'Manifest needs at least 2 lines (input_path, output_dir)');
end

in_path = strtrim(lines{1});
out_dir = strtrim(lines{2});

fprintf('Input: %s\n', in_path);
fprintf('Output dir: %s\n', out_dir);

if ~isfile(in_path)
  error('run_noise_from_paths:NoInput', 'Input TIFF not found: %s', in_path);
end

mkdir(out_dir);

% Build noise params from manifest (lines 3-9) or fall back to noise_config
if numel(lines) >= 9
  p = struct();
  p.mu        = str2double(lines{3});
  p.sigma     = str2double(lines{4});
  p.mu0       = str2double(lines{5});
  p.sigma0    = str2double(lines{6});
  p.darkcount = str2double(lines{7});
  p.bleedp    = str2double(lines{8});
  p.bleedw    = str2double(lines{9});
  fprintf('Params from manifest: mu=%.1f sigma=%.1f sigma0=%.2f darkcount=%.3f bleedp=%.2f bleedw=%.2f\n', ...
    p.mu, p.sigma, p.sigma0, p.darkcount, p.bleedp, p.bleedw);
else
  p = noise_config();
  fprintf('Params from noise_config (manifest had %d lines)\n', numel(lines));
end
p = check_noise_params(p);

[~, name, ext] = fileparts(in_path);
out_path = fullfile(out_dir, [name, ext]);

info = imfinfo(in_path);
nf = numel(info);
if nf == 1
  img = imread(in_path);
  img = reshape(img, [size(img,1), size(img,2), 1]);
else
  img = zeros(info(1).Height, info(1).Width, nf, 'like', imread(in_path, 1));
  for k = 1:nf
    img(:,:,k) = imread(in_path, k);
  end
end

if isinteger(img)
  img = single(img) / double(intmax(class(img)));
else
  img = single(img);
end

fprintf('Loaded image size=[%d,%d,%d] class=%s range=[%.4g,%.4g]\n', ...
  size(img,1), size(img,2), size(img,3), class(img), min(img(:)), max(img(:)));

noisy_mov = applyNoiseModel(img, p);

fprintf('[run_noise_from_paths] NOISE applied, output range=[%.4g,%.4g]\n', min(noisy_mov(:)), max(noisy_mov(:)));

t = Tiff(out_path, 'w');
tagstruct.ImageLength = size(noisy_mov, 1);
tagstruct.ImageWidth  = size(noisy_mov, 2);
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32;
tagstruct.SamplesPerPixel = 1;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Compression = Tiff.Compression.None;
for k = 1:size(noisy_mov, 3)
  t.setTag(tagstruct);
  t.write(noisy_mov(:,:,k));
  if k < size(noisy_mov, 3)
    t.writeDirectory();
  end
end
t.close();
fprintf('Wrote %s\n', out_path);
end
