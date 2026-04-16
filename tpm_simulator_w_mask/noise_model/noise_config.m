function p = noise_config()
% NOISE_CONFIG  Noise model parameters. Edit to tune.

p = struct();

p.mu        = 500;     % Mean gain per photon
p.sigma     = 2300;    % Variance of gain per photon
p.mu0       = 0;       % Readout offset
p.sigma0    = 1.5;     % Readout noise std
p.darkcount = 0.01;    % PMT dark count rate

p.bleedp    = 0.01;     % Pixel bleed probability (0–1); 0 disables
p.bleedw    = 0.05;     % Max bleed fraction when it occurs (0–1)

end
