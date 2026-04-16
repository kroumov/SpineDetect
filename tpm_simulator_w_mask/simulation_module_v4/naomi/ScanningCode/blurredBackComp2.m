function img_out = blurredBackComp2(TMPvol, idx, psf_lowres, pwr_ratio, mask, freq_opt, zscale, varargin)

% function img_out = addBlurredBackground(TMPvol, idx_top, idx_bot, psf_lowres, pwr_ratios)
%
% Create the background image adding in the 
%
% 2018 - Adam Charles and Alex Song

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input checking
  
if(nargin<7)
    zscale = [];
end
if(nargin<6)
    freq_opt = 0;
end
if(nargin<5)
    mask = [];
end
if(nargin<4)
    pwr_ratio = 1;
end

% Optional psf_sz: first varargin when 1x2/2x1 numeric (spatial PSF xy size)
% Fixes crop alignment when nearest_small_prime expands FFT size.
psf_sz = [];
if ~isempty(varargin) && isnumeric(varargin{1}) && numel(varargin{1}) <= 4
  v1 = varargin{1};
  if (isequal(size(v1),[1 2]) || isequal(size(v1),[2 1])) && all(v1(:) >= 1) && all(v1(:) < 1e4)
    psf_sz = [v1(1), v1(2)];
    varargin = varargin(2:end);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate out-of-plane fluorescence

% img_out = mean(TMPvol(:,:,idx),3);                                         % Get the total sum of the top and bottom layers, modulated by the laser power above and below the focus
% if(~isempty(varargin))
%     for i = 1:length(varargin)
%         img_out = img_out + mean(varargin{i}(:,:,idx),3);
%     end
% end

v = zeros(size(TMPvol,3),1); 
v(idx) = 1/numel(idx);
if(~isempty(zscale)&&length(zscale)>=length(idx))
  if(length(zscale)>length(idx))
    zscale = zscale(1:length(idx));
  end
  vscale = v;
  vscale(idx) = vscale(idx).*zscale(:);
  img_out = (1/numel(idx))*sum(bsxfun(@times,TMPvol(:,:,idx),reshape(zscale,1,1,[])),3);
else
  img_out = (1/numel(idx))*sum(TMPvol(:,:,idx),3);
end
% reshape(reshape(TMPvol, [], size(TMPvol,3))*v, size(TMPvol,1), size(TMPvol,2), []);

% img_out = reshape(reshape(TMPvol, [], size(TMPvol,3))*v, size(TMPvol,1), size(TMPvol,2), []);
if(~isempty(varargin))
  for i = 1:length(varargin)
    if(~isempty(zscale)&&length(zscale)>=length(idx))
      img_out = img_out +reshape(reshape(varargin{i}, [], size(varargin{i},3))*vscale, size(varargin{i},1), size(varargin{i},2), []);
    else
      img_out = img_out +reshape(reshape(varargin{i}, [], size(varargin{i},3))*v, size(varargin{i},1), size(varargin{i},2), []);
    end
  end
end


img_out = pwr_ratio*img_out;
if(~isempty(mask))
    img_out = img_out.*mask;               
end

if size(psf_lowres,3) ~= 1
  psf_lowres = mean(psf_lowres(:,:,idx),3);
end

if freq_opt
    sz      = size(psf_lowres);                                            % Get the sizes of the post-convolution array
    img_out = ifft(ifft(fft(fft(img_out,sz(1),1),sz(2),2).*psf_lowres,...
                                              [], 1), [], 2, 'symmetric'); % 
    if ~isempty(psf_sz) && numel(psf_sz) >= 2
      % Use floor((psf_sz-1)/2) to match conv2(...,'same') when sz expanded by nearest_small_prime
      y_ix   = floor((psf_sz(1)-1)/2) + [1,size(TMPvol,1)];
      y_jx   = floor((psf_sz(2)-1)/2) + [1,size(TMPvol,2)];
    else
      y_ix   = ceil((sz(1)-size(TMPvol,1)-1)/2) + [1,size(TMPvol,1)];      % Original (buggy when sz expanded)
      y_jx   = ceil((sz(2)-size(TMPvol,2)-1)/2) + [1,size(TMPvol,2)];
    end
    img_out = img_out(y_ix(1):y_ix(2),y_jx(1):y_jx(2));                    % Crop the image
else
    img_out = conv2(img_out, psf_lowres,'same');                           % Blur with the low-resolution "psf"
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%