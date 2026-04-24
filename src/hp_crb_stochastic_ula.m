function crb_mean_mse_rad2 = hp_crb_stochastic_ula(theta_deg, P, sigma2noise, M, N)
% Approximate stochastic CRB for zero-mean complex Gaussian snapshots x~CN(0,R).
% R = A P A^H + sigma2 I, ULA with d=lambda/2.
% Output: mean diag(CRB) in rad^2.

    K = numel(theta_deg);
    d_lambda = 0.5;
    m = (0:M-1).';
    % The experiment driver passes degrees, while the CRB is parameterized
    % in radians.
    theta_rad = deg2rad(theta_deg(:).');
    A = zeros(M,K);
    dA = zeros(M,K);
    for k = 1:K
        a = exp(-1j*2*pi*d_lambda*sin(theta_rad(k))*m);
        da = a .* (-1j*2*pi*d_lambda*cos(theta_rad(k))*m);
        A(:,k) = a;
        dA(:,k) = da;
    end
    R = A*P*A' + sigma2noise*eye(M);
    Rinv = R \ eye(M);
    FIM = zeros(K,K);
    for i = 1:K
        dRi = dA(:,i) * (P(i,:)*A') + A * (P(:,i) * dA(:,i)');
        for j = 1:K
            dRj = dA(:,j) * (P(j,:)*A') + A * (P(:,j) * dA(:,j)');
            FIM(i,j) = N * real(trace(Rinv * dRi * Rinv * dRj));
        end
    end
    C = pinv(FIM);
    crb_mean_mse_rad2 = mean(real(diag(C)));
end
