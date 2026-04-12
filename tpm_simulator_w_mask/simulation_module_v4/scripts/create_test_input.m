% CREATE_TEST_INPUT  Generate 100x100x50 test volume for TPM simulation.
% Saves to input/neurons_test.tif

base_dir = fileparts(fileparts(mfilename('fullpath')));  % module root
input_dir = fullfile(base_dir, 'input');
if ~isfolder(input_dir)
  mkdir(input_dir);
end

% 100x100x50 volume, single precision 0-1
vol = rand(100, 100, 50, 'single') * 0.5;  % random fluorescence-like
out_path = fullfile(input_dir, 'neurons_test.tif');

% Write TIFF stack
t = Tiff(out_path, 'w');
tagstruct.ImageLength = 100;
tagstruct.ImageWidth = 100;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32;
tagstruct.SamplesPerPixel = 1;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Compression = Tiff.Compression.None;
for k = 1:50
  t.setTag(tagstruct);
  t.write(vol(:, :, k));
  if k < 50
    t.writeDirectory();
  end
end
t.close();

fprintf('Created: %s (100x100x50)\n', out_path);
