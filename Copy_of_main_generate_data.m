clear all;
clc;

%% System Settings
N_BS = 64;          % number of BS antennas
M_BS = 64;          % slots of downlink pilots, equal to N_BS
N_RF = 16;
D = 4;

K = 6;              % number of UEs
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

SNRdB = 5;          % SNR~0-10dB
    
%% downlink
UEChannels = cell(1, K);
for k = 1 : K
	% generate UE channel
	Hk_original = UE_Channel(N_UE, N_BS, L, angle_spread, space);    
	% normalization, average power = 1
 	Hk = Hk_original / norm(Hk_original,'fro') * sqrt(N_BS * N_UE);	 % output for k'th NN
	UEChannels{k} = Hk';
	% received signal at UE
	R_noiseless = Hk * FBar;
	power = norm(R_noiseless,'fro')^2/N_UE/M_BS; 
	SNR = 10.^(SNRdB./10);
	sigma2 = power / SNR;
    noise = sqrt(sigma2/2)*(randn(N_UE,M_BS) + 1i*randn(N_UE,M_BS));
	Yk = WBar' * Hk * FBar + WBar' * noise;      
 	Rk = WBar*Yk*FBar';   % input for k'th NN
    
    % LMMSE/LS Estimation
    [Hk_LS,Hk_LMMSE] = LMMSE_estimation(Yk, WBar, FBar, sigma2);
    MSEk_LS = norm(Hk_LS-Hk,'fro')^2 / norm(Hk,'fro')^2;    
    MSEk_LMMSE = norm(Hk_LMMSE-Hk,'fro')^2 / norm(Hk,'fro')^2;  
end




%% uplink
H = horzcat(UEChannels{:});                        % output for BS NN
R_noiseless = H * WTilde;
power = norm(R_noiseless,'fro')^2/N_BS/T; 
SNR = 10.^(SNRdB./10);
sigma2 = power / SNR;
noise = sqrt(sigma2/2)*(randn(N_BS,T) + 1i*randn(N_BS,T));
% received signal after RF combining at BS
Yr = FBar' * H * WTilde + FBar' * noise;      
R =  FBar * Yr * WTilde';                          % input for BS NN


