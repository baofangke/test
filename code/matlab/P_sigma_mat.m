function result = P_sigma_mat(A, U1, U2, Sigma1, Sigma2)
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
[p, q] = size(A);

% Compute projection operators: P_U^Sigma = U * inv(U' * Sigma * U) * U'
% We use the backslash operator for stability.
temp1 = U1' * Sigma1 * U1;
P1 = U1 * (temp1 \ U1');
temp2 = U2' * Sigma2 * U2;
P2 = U2 * (temp2 \ U2'); 
% Part 1: A ×₁ (P₁·Σ₁) ×₂ (P₂·Σ₂) ×₃ (P₃·Σ₃)
part1 = ttm(tensor(A), {P1*Sigma1, P2*Sigma2}, [1, 2]);
part1 = double(part1);

% Part 2: Sum over j=1,2,3 
part21 = double(ttm(tensor(A), {P1*Sigma1, eye(q) - P2*Sigma2}, [1, 2]));
part22 = double(ttm(tensor(A), {eye(p) - P1*Sigma1, P2*Sigma2}, [1, 2])); 
result = part1 + part21 + part22;
end
