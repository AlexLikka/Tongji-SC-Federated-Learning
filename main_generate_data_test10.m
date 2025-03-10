clear all;
clc;
N_sample = 600;
K = 6;              % number of users
Label_list = cell(1, K);
Receive_list = cell(1, K);
Testsets = cell(6 * N_sample, 2);
output_data_compl = zeros(64, 4, K, 6 * N_sample);

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
    SNRdB = 10;        % SNR~0-10dB
    UEChannels = cell(1, K);
    UE_Label_list = cell(1, K);
    UE_Receive_list = cell(1, K);
        
    %% downlink
    % UEChannels = cell(1, K);
    for k = 1 : K
	    % generate UE channel
	    Hk_original = UE_Channel(N_UE, N_BS, L, angle_spread, space);    
	    % normalization, average power = 1
 	    Hk = (Hk_original / norm(Hk_original,'fro') * sqrt(N_BS * N_UE))';	 % output for k'th NN
        Hk_ext = extract_real_image(Hk);
        UE_Label_list{k} = Hk_ext;
	    UEChannels{k} = Hk;
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

%% uplink
    for k = 1 : K
        H = horzcat(UEChannels{:});                        % output for BS NN
        R_noiseless = H * WTilde;
        power = norm(R_noiseless,'fro')^2/N_BS/T; 
        SNR = 10.^(SNRdB./10);
        sigma2 = power / SNR;
        noise = sqrt(sigma2/2)*(randn(N_BS,T) + 1i*randn(N_BS,T));
        % received signal after RF combining at BS
        Yr = FBar' * H * WTilde + FBar' * noise;      
        R =  FBar * Yr * WTilde';                          % input for BS NN
    end
    H_reshape = reshape(H,[64,4,6]);
    output_data_compl(:, :, :, n) = H_reshape;
    R_reshape = reshape(R,[64,4,6]);

    for k = 1 : K
        Hk_ext = extract_real_image(H_reshape(:,:,k));
        Label_list{k} = Hk_ext;
        Rk_ext = extract_real_image(R_reshape(:,:,k));
        Receive_list{k} = Rk_ext;
    end

    Testsets{round(SNRdB / 2 * N_sample + n),1,1} = Receive_list;
    Testsets{round(SNRdB / 2 * N_sample + n),1,2} = Label_list;
end

input_data = zeros(128, 4, K, N_sample);
output_data = zeros(128, 4, K, N_sample);
for n = 1:(N_sample * 6)
    for k = 1:K
        tmp_input = Testsets{n,1}{1,k};
        input_data(:, :, k, n) = tmp_input;
        tmp_output = Testsets{n,2}{1,k};
        output_data(:, :, k, n) = tmp_output;
    end
end
% input_data = reshape(input_data, [N_sample*K,1,64,8]);
% output_data = reshape(output_data, [N_sample*K,1,64,8]);
save Test600 input_data output_data;


%%% Additional
%% function to extract Real and Image, formulate new matrix
function Rk_ext = extract_real_image(Rk)
Rk_ext=[real(Rk); imag(Rk)];
% Rk_ext(:,[2 5]) = Rk_ext(:,[5 2]);
% Rk_ext(:,[4 7]) = Rk_ext(:,[7 4]); 
end
