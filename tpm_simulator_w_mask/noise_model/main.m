% MAIN  Apply Poisson-Gaussian noise and pixel bleed to TIFFs in input/, write to output/.

base = fileparts(mfilename('fullpath'));
addpath(base);

input_dir  = fullfile(base, 'input');
output_dir = fullfile(base, 'output');
mkdir(output_dir);

p = noise_config();
p = check_noise_params(p);

d = dir(fullfile(input_dir, '*.tif*'));
if isempty(d)
  return;
end

for i = 1:numel(d)
  in_path  = fullfile(d(i).folder, d(i).name);
  [~, stem, ext] = fileparts(d(i).name);
  out_path = fullfile(output_dir, [stem, ext]);

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

  noisy_mov = applyNoiseModel(img, p);

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
end
