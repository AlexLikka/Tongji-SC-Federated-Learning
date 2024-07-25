function RF = generate_RFMatrix(M, N)
    % M: 行数
    % N: 列数
    % N_BS: 天线数

    % 初始化矩阵
    RF = zeros(M, N);

    % 填充矩阵
    for m = 1:M
        for n = 1:N
            RF(m, n) = exp(-2 * pi * 1i * (m-1) * (n-1) / N) / sqrt(M);
        end
    end
end