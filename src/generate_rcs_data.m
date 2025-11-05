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
    c0 = 3e8;           % Speed of light (m/s)

    % ----------------------------------------------------
    % Define RCS ranges for each target class
    % ----------------------------------------------------
    % sigma_ranges = [min, max] for each class
    sigma_ranges = [
        0.01, 2.0;      % Class 1 - small target (bird)
        1.0, 20.0;      % Class 2 - medium target (drown)
        15.0, 50.0;     % Class 3 - large target (airplane)
    ];

    sigma_mean_values = mean(sigma_ranges, 2);

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
    
    % Devide N to 3 groups - so target destribution is equal
    N1 = floor(N/3);
    N2 = floor(N/3);
    N3 = N - N1 - N2; 

    % Create random permutation of the 3 groups
    target_class = [ones(1, N1), 2*ones(1, N2), 3*ones(1, N3)];
    target_class = target_class(randperm(N));
    
    for i = 1:N

        cls = target_class(i);
        sigma_mean = sigma_mean_values(cls);    % select the RCS mean for target class  
        X = randn(2,1);                         % 2 standard normal variables
        draw_chi2 = sum(X.^2);                  % sum of squares -> chi2 with 2 DOF
        sigma_i = draw_chi2 * sigma_mean / 2;   % scale to match the desired mean
        sigma_i = min(max(sigma_i, sigma_ranges(cls,1)), sigma_ranges(cls,2)); % make sure val is within range 
        sigma(i) = sigma_i;                    

        % Random Range between 5 km - 40 km
        R = 5000 + rand() * 35000;
        true_range = R;

        % Ideal power received based on true range
        Pr = PT * (G^2) * (LAMBDA^2) * sigma_i / ((4*pi)^3 * true_range^4 * L);
        
        % Compute SNR
        SNR(i) = Pr / Pn;

        % Add Range Measurement Error as function of SNR
        delta_R = c0 / (2 * B * sqrt(SNR(i)));  % meters

        % Add Gaussian noise for measurement inaccuracy
        range_measured = true_range + delta_R * randn();
        range(i) = range_measured;

    end

    % ----------------------------------------------------
    % Save data to CSV file
    % Columns: [Range, SNR, target_class]
    % ----------------------------------------------------
    data = [range(:), SNR(:), target_class(:)];
    csvwrite(filename, data);
    fprintf('CSV successfully created with %d samples: %s\n', N, filename);
end
