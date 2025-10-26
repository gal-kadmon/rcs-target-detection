function generate_rcs_data(filename, N)
    % ----------------------------------------------------
    % Synthetic Radar RCS Data Generator
    % Features for model: Pr_noisy
    % Labels: target_class
    % ----------------------------------------------------

    global PT G LAMBDA F B T L

    % Radar & physical parameters
    PT = 100;            % Transmit power (Watts)
    G = 90;             % Antenna gain (linear)
    L = 1.5;            % System losses
    LAMBDA = 0.03;      % Wavelength (m)
    R = 1000;           % Fixed range (m)

    % Noise parameters
    k = 1.38e-23;       % Boltzmann constant
    T = 290;            % Noise temperature (Kelvin)
    B = 1e6;            % Bandwidth (Hz)
    F = 3;              % Noise figure (linear)
    NOISE_FLOOR = k * T * B * F * L;

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
    Pr_noisy = zeros(N,1);
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

        % Ideal received power (constant range)
        Pr = PT * (G^2) * (LAMBDA^2) * sigma_i / ((4*pi)^3 * R^4 * L);

        % Add Gaussian noise
        Pr_noisy(i) = Pr + sqrt(NOISE_FLOOR) * randn();
    end

    % ----------------------------------------------------
    % Save data to CSV file
    % Columns: [Pr_noisy, sigma, target_class]
    % ----------------------------------------------------
    data = [Pr_noisy, sigma, target_class];
    csvwrite(filename, data);
    fprintf('CSV successfully created with %d samples: %s\n', N, filename);
end
