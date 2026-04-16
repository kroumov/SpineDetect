 function psf_params = TPM_Simulation_Parameters(opt_type, psf_params)
%
% [psf_params, tpm_params] = TPM_Simulation_Parameters(opt_type, , psf_params, tpm_params)
%  
% This function sets the default parameters for a few sets of imaging
% conditions
% 
%   - filename          - Parameter output filename
%
%   - opt_type          - Optics type. Defaults to standard
%          'standard'   - Gaussian illumination of back aperture
%          'bessel'     - Bessel beam illumination (Lu et al 2017)
%          'stefo'      - Temporally focused beam (Prevedel et al 2016)
%          'vtwins'     - vTwINS illumination
%
% 2020 - Adam Charles & Alex Song


switch opt_type
    case 'lowNA'
        psf_params.NA     = 0.2;                                             % Numerical aperture of PSF
        psf_params.psf_sz = [20 20 80];
    case 'standard'
        psf_params.NA     = 0.6;                                             % Numerical aperture of PSF
        psf_params.psf_sz = [20 20 50];
    case 'bessel'
        psf_params        = getDefaultPSFParams('bessel');    
        psf_params.psf_sz = [20 20 80];
        tpm_params.pavg   = 120;    
    case 'stefo'
        psf_params        = getDefaultPSFParams('temporal-focusing');    
        psf_params.psf_sz = [20 20 50];
        tpm_params.pavg   = 400;
    case 'vtwins'
        psf_params        = getDefaultPSFParams('vtwins');
        psf_params.psf_sz = [20 20 80];
    otherwise
        error('Given optics type is invalid')
end

psf_params.objNA     = 0.8;                                                % Numerical aperture of PSF
psf_params.zernikeWt = [0 0 0 0 0 0 0 0 0 0 0.06];                         % In units of wavelength, a small amount of spherical aberration and astigmatism added as uncorrected "system" aberrations

psf_params = check_psf_params(psf_params);                                 % Check point spread function parameters
tpm_params = check_tpm_params(tpm_params);


end
