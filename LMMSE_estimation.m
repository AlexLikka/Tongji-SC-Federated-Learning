function [H_LS,H_LMMSE] = LMMSE_estimation(Y, WBar, FBar, noiseVar)
    M = size(WBar, 2);
    N = size(FBar, 2);
   % LS estimation
   vecY = Y(:);
   Wp = kron(FBar.', WBar');
   h_LS = pinv(Wp'*Wp)*Wp'*vecY;
   H_LS = reshape(h_LS, M, N);
  
   % LMMSE estimation
   h_LMMSE = Wp'* pinv( Wp*Wp' + (eye(M*N)+1i*eye(M*N)).* noiseVar ) * vecY;
   H_LMMSE = reshape(h_LMMSE, M, N);
   
   
   
