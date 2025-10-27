function generate_rcs_data(filename, N)
    % ----------------------------------------------------
    % Synthetic Radar RCS Data Generator
    % Features for model: [Range, SNR]
    % Labels: target_class (1=small, 2=medium, 3=large)
    % ----------------------------------------------------

    % Radar & physical parameters
    PT = 100;            % Transmit power (Watts)
    G = 50;             % Antenna gain (linear)
    L = 1.5;            % System losses
    LAMBDA = 0.03;      % Wavelength (m)

    % Noise parameters
    k = 1.38e-23;       % Boltzmann constant
    T = 290;            % Noise temperature (Kelvin)
    B = 1e6;            % Bandwidth (Hz)
    F = 3;              % Noise figure (linear)
    Pn = k * T * B * F * L;  % Noise power (W)

    % RCS ranges (with overlap)
    sigma_ranges = [
        0.01, 2.0;      % Class 1 - small target
        1.0, 20.0;      % Class 2 - medium target
        15.0, 50.0;     % Class 3 - large target
    ];

    % ----------------------------------------------------
    % Preallocate arrays
    % ----------------------------------------------------
    sigma = zeros(N,1);
    range = zeros(N,1);
    SNR = zeros(N,1);
    target_class = zeros(N,1);

    % ----------------------------------------------------
    % Generate samples
    % ----------------------------------------------------
    for i = 1:N
        % Randomly assign target type (1, 2, or 3)
        cls = randi(3);
        target_class(i) = cls;

        % Draw random sigma from class range
        sigma_min = sigma_ranges(cls,1);
        sigma_max = sigma_ranges(cls,2);
        sigma_i = sigma_min + (sigma_max - sigma_min) * rand();
        sigma(i) = sigma_i;

        % Random Range between 1 km - 10 km
        R = 1000 + rand() * 9000;
        Range(i) = R;

        % Ideal received power
        Pr = PT * (G^2) * (LAMBDA^2) * sigma_i / ((4*pi)^3 * Range(i)^4 * L);

        % Add Gaussian noise
        SNR(i) = (Pr / Pn) * (1 + 0.05 * randn());
    end

    % ----------------------------------------------------
    % Save data to CSV file
    % Columns: [Range, SNR, target_class]
    % ----------------------------------------------------
    data = [Range(:), SNR(:), target_class(:)];
    csvwrite(filename, data);
    fprintf('CSV successfully created with %d samples: %s\n', N, filename);
end
