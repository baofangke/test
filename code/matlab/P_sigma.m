function result = P_sigma(A, G, U1, U2, U3, Sigma1, Sigma2, Sigma3)
% P_SIGMA Computes the P_Sigma operator on tensor A
%
% P_Sigma(A) = A ×₁ (P₁·Σ₁) ×₂ (P₂·Σ₂) ×₃ (P₃·Σ₃) + sum_j G ×_{k≠j} U_k ×_j W_j
%
% Inputs:
%   A - Input tensor (p1 x p2 x p3)
%   G - Core tensor (r1 x r2 x r3)
%   U1, U2, U3 - Factor matrices (p_j x r_j)
%   Sigma1, Sigma2, Sigma3 - Covariance matrices (p_j x p_j)
%
% Output:
%   result - Transformed tensor (p1 x p2 x p3)

% Compute projection operators: P_U^Sigma = U * inv(U' * Sigma * U) * U'
% We use the backslash operator for stability.
temp1 = U1' * Sigma1 * U1;
P1 = U1 * (temp1 \ U1');
temp2 = U2' * Sigma2 * U2;
P2 = U2 * (temp2 \ U2');
temp3 = U3' * Sigma3 * U3;
P3 = U3 * (temp3 \ U3');

% Part 1: A ×₁ (P₁·Σ₁) ×₂ (P₂·Σ₂) ×₃ (P₃·Σ₃)
part1 = ttm(tensor(A), {P1*Sigma1, P2*Sigma2, P3*Sigma3}, [1, 2, 3]);
part1 = double(part1);

% Part 2: Sum over j=1,2,3
part2 = zeros(size(A));

for j = 1:3
    % Compute W_j
    W_j = compute_W_Psigma(A, G, U1, U2, U3, Sigma1, Sigma2, Sigma3, P1, P2, P3, j);
    
    % Compute G ×_{k≠j} U_k ×_j W_j
    if j == 1
        temp = ttm(tensor(G), {U2, U3}, [2, 3]);
        temp = ttm(temp, W_j, 1);
    elseif j == 2
        temp = ttm(tensor(G), {U1, U3}, [1, 3]);
        temp = ttm(temp, W_j, 2);
    else % j == 3
        temp = ttm(tensor(G), {U1, U2}, [1, 2]);
        temp = ttm(temp, W_j, 3);
    end
    
    part2 = part2 + double(temp);
end

result = part1 + part2;
end

function W_j = compute_W_Psigma(A, G, U1, U2, U3, Sigma1, Sigma2, Sigma3, P1, P2, P3, j)
% Compute W_j for P_sigma operator
% W_j = (I - P_j·Σ_j)·A_j·(⊗_{k≠j} Σ_k U_k)·G_j^T·(...)^{-1}

sz_A = size(A);
sz_G = size(G);
p = [sz_A(1), sz_A(2), sz_A(3)];

if j == 1
    Sigma_j = Sigma1; P_j = P1;
    Sigma_k_U_k_kron = kron(Sigma3*U3, Sigma2*U2);
    U_k = {U2, U3};
    Sigma_k = {Sigma2, Sigma3};
    p_j = p(1);
elseif j == 2
    Sigma_j = Sigma2; P_j = P2;
    Sigma_k_U_k_kron = kron(Sigma3*U3, Sigma1*U1);
    U_k = {U1, U3};
    Sigma_k = {Sigma1, Sigma3};
    p_j = p(2);
else
    Sigma_j = Sigma3; P_j = P3;
    Sigma_k_U_k_kron = kron(Sigma2*U2, Sigma1*U1);
    U_k = {U1, U2};
    Sigma_k = {Sigma1, Sigma2};
    p_j = p(3);
end

A_j = tenmat(tensor(A), j);
A_j = double(A_j);
G_j = tenmat(tensor(G), j);
G_j = double(G_j);

complement = eye(p_j) - P_j * Sigma_j;

inner_term = kron(U_k{2}' * Sigma_k{2} * U_k{2}, ...
                  U_k{1}' * Sigma_k{1} * U_k{1});

% Compute inv_term without using inv
% inv_term = inv(G_j * inner_term * G_j');
% Instead, solve for temp: (G_j * inner_term * G_j') * temp = G_j, then use temp'
temp = (G_j * inner_term * G_j') \ G_j;

W_j = complement * A_j * Sigma_k_U_k_kron * temp';
end