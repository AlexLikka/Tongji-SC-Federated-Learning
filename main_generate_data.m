clear all;
clc;
%¿É¸Ä1000
N_sample = 1000;
K = 6;              % number of users
SNR_values = 0:2:10;
Trainsets = cell(N_sample * length(SNR_values), 3);  % Adding one more dimension for SNR
for SNR_idx = 1:length(SNR_values)
    SNRdB = SNR_values(SNR_idx);
    for n = 1:N_sample

        %% System Settings
        N_BS = 64;          % number of BS antennas
        M_BS = 64;          % slots of downlink pilots, equal to N_BS
        N_RF = 16;
        D = 4;

        % K = 6;              % number of UEs
        N_UE = 4;           % number of UE antennas
        M_UE = 4;           % times of UE receiver for each slot, equal to N_UE
        T = N_UE * K;       % slots of uplink pilots

        % channel setting
        L = 4;
        angle_spread = 180;
        space = 0.5;

        FBar = generate_RFMatrix(N_BS, M_BS);   % BS precoding matrix
        WBar = generate_RFMatrix(N_UE, M_UE);   % UE Combiner matrix
        WTilde = generate_RFMatrix(N_UE*K, T); 
        FWilde = generate_RFMatrix(N_BS, N_RF*D); 

        UEChannels = cell(1, K);
        UE_Label_list = cell(1, K);
        UE_Receive_list = cell(1, K);

        %% downlink
        for k = 1 : K
            % generate UE channel
            Hk_original = UE_Channel(N_UE, N_BS, L, angle_spread, space);    
            % normalization, average power = 1
            Hk = (Hk_original / norm(Hk_original,'fro') * sqrt(N_BS * N_UE))';	 % output for k'th NN
            Hk_ext = extract_real_image(Hk);
            UE_Label_list{k} = Hk_ext;
            UEChannels{k} = Hk';
            % received signal at UE
            R_noiseless = Hk' * FBar;
            power = norm(R_noiseless,'fro')^2/N_UE/M_BS; 
            SNR = 10.^(SNRdB./10);
            sigma2 = power / SNR;
            noise = sqrt(sigma2/2)*(randn(N_UE,M_BS) + 1i*randn(N_UE,M_BS));
            Yk = WBar' * Hk' * FBar + WBar' * noise;      
            Rk = WBar*Yk*FBar';   % input for k'th NN
            Rk_ext = extract_real_image(Rk');
            UE_Receive_list{k} = Rk_ext;

            % LMMSE/LS Estimation
            [Hk_LS,Hk_LMMSE] = LMMSE_estimation(Yk, WBar, FBar, sigma2);
            MSEk_LS = norm(Hk_LS-Hk','fro')^2 / norm(Hk','fro')^2;    
            MSEk_LMMSE = norm(Hk_LMMSE-Hk','fro')^2 / norm(Hk','fro')^2;  
        end
        % Store the data with SNR value
        idx = (SNR_idx-1) * N_sample + n;
        Trainsets{idx, 1} = UE_Receive_list;
        Trainsets{idx, 2} = UE_Label_list;
        Trainsets{idx, 3} = SNRdB;  % Store the SNR value
    end
end

% Modify the input and output data storage to include the SNR dimension
input_data = zeros(128, 4, K, N_sample * length(SNR_values));
output_data = zeros(128, 4, K, N_sample * length(SNR_values));
SNR_data = zeros(N_sample * length(SNR_values), 1);  % Array to store SNR values

for n = 1:(N_sample * length(SNR_values))
    for k = 1:K
        tmp_input = Trainsets{n,1}{1,k};
        input_data(:, :, k, n) = tmp_input;
        tmp_output = Trainsets{n,2}{1,k};
        output_data(:, :, k, n) = tmp_output;
    end
    SNR_data(n) = Trainsets{n, 3};  % Retrieve the stored SNR value
end

save Train3000 input_data output_data SNR_data;

%%% Additional
%% function to extract Real and Image, formulate new matrix
function Rk_ext = extract_real_image(Rk)
Rk_ext=[real(Rk); imag(Rk)];
end
