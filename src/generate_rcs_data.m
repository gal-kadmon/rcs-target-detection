function generate_rcs_data(filename, N, target_ratio)
    % -------------------------------
    % Radar RCS synthetic data generator
    % -------------------------------
    
    global MIN_RANGE MAX_RANGE MIN_SIGMA MAX_SIGMA PT G LAMBDA F B T
    
    % Target range (m)
    MIN_RANGE = 50; MAX_RANGE = 1000;
    % Typical sigma values (m^2)
    MIN_SIGMA = 0.1; MAX_SIGMA = 1.0;
    % Radar pulse parameters
    PT = 1;           % Transmit power (Watts)
    G = 30;           % Antenna gain (linear)
    L = 1.5;          % System losses
    LAMBDA = 0.03;    % Wavelength (m)
    
    % Noise parameters
    k = 1.38e-23;     % Boltzmann constant
    T = 290;          % Noise temperature (Kelvin)
    B = 1e6;          % Bandwidth (Hz)
    F = 3;            % Noise figure (linear)
    
    % Noise floor (base noise power)
    NOISE_FLOOR = k*T*B*F*L;
    
    % -------------------------------
    % Arrays for data
    % -------------------------------
    ranges = zeros(N,1);
    target_present = zeros(N,1);
    sigma = zeros(N,1);
    RCS = zeros(N,1);
    SNR = zeros(N,1);
    
    for i = 1:N
        if rand() < target_ratio
            target_present(i) = 1;
            % Random range and sigma
            R = MIN_RANGE + (MAX_RANGE-MIN_RANGE)*rand();
            sigma_i = MIN_SIGMA + (MAX_SIGMA-MIN_SIGMA)*rand();
            ranges(i) = R;
            sigma(i) = sigma_i;
            
            % Received power calculation
            Pr = PT*G^2*LAMBDA^2*sigma_i/((4*pi)^3*R^4*L);
            
            % Add white Gaussian noise
            Pr_noisy = Pr + sqrt(NOISE_FLOOR)*randn();
            RCS(i) = Pr_noisy;
            
            % Compute SNR
            SNR(i) = 10*log10(max(Pr,1e-12)/NOISE_FLOOR) + 2*randn();
        else
            target_present(i) = 0;
            % Background random range
            R = MIN_RANGE + (MAX_RANGE-MIN_RANGE)*rand();
            ranges(i) = R;
            sigma(i) = 0;
            
            % Only white Gaussian noise
            Pr_noisy = sqrt(NOISE_FLOOR)*randn();
            RCS(i) = Pr_noisy;
            
            SNR(i) = 10*log10(max(abs(Pr_noisy),1e-12)/NOISE_FLOOR) + 2*randn();
        end
    end
    
    % -------------------------------
    % Save to CSV
    % -------------------------------
    data = [ranges, target_present, sigma, RCS, SNR];
    csvwrite(filename, data);
    fprintf('CSV successfully created: %s\n', filename);
end
