load('og_psf.mat');
og_psf = PSF_struct.psf;
load('na_psf.mat');
na_psf = PSF_struct.psf;

figure()

iis = 8:1:13;

% og_psf = log(og_psf);
% na_psf = log(na_psf);

og_psf = rescale(og_psf);
na_psf = rescale(na_psf);

minVal = min(min([og_psf(:) na_psf(:)]));
maxVal = max(max([og_psf(:) na_psf(:)]));

maxVal = maxVal;

for ii = 1:length(iis)
    og_slice = squeeze(og_psf(:,iis(ii),:))';
    na_slice = squeeze(na_psf(:,iis(ii),:))';

    subplot(2, length(iis), ii);
    imagesc(og_slice);
    clim([minVal maxVal]);

    subplot(2, length(iis), length(iis) + ii);
    imagesc(na_slice);
    clim([minVal maxVal]);

end

colorbar

%%
data = PSF_struct.psf;
data = mat2gray(data); % scale to [0,1]
data = uint16(data * 65535);

filename = 'output.tif';

% Write first slice
imwrite(data(:,:,1), filename, 'tif', 'Compression', 'none');

% Append remaining slices
for k = 2:size(data,3)
    imwrite(data(:,:,k), filename, 'tif', ...
        'WriteMode', 'append', ...
        'Compression', 'none');
end

%%
figure()
histogram(na_psf(:))