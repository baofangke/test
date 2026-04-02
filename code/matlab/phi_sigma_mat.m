function result = phi_sigma_mat(A, U1, U2, Sigma1, Sigma2)
% PHI_SIGMA Computes the Phi_Sigma operator on tensor A
%
% Phi_Sigma(A) = A ×₁ P₁^Σ ×₂ P₂^Σ ×₃ P₃^Σ + sum_j G ×_{k≠j} U_k ×_j Σ_j^{-1}W_j^*
%
% Inputs:
%   A - Input tensor (p1 x p2 x p3)
%   G - Core tensor (r1 x r2 x r3)
%   U1, U2, U3 - Factor matrices (p_j x r_j)
%   Sigma1, Sigma2, Sigma3 - Covariance matrices (p_j x p_j)
%
% Output:
%   result - Transformed tensor (p1 x p2 x p3)


% %
% A = double(T);
% U1 = data_ind.u1;
% U2 = data_ind.u2;
% Sigma1 = data_ind.S1;
% Sigma2 = data_ind.S2 / data_ind.avefro; 

% Compute projection operators: P_U^Sigma = Sigma^{-1} * U * U' * Sigma
    temp1 = U1' * Sigma1 * U1;
    P1 = U1 * (temp1 \ U1'); 
    temp2 = U2' * Sigma2 * U2;
    P2 = U2 * (temp2 \ U2');  

    % Part 1: A ×₁ P₁ ×₂ P₂ ×₃ P₃
    part1 = ttm(tensor(A), {P1, P2}, [1, 2]);
    part1 = double(part1);

    % Part 2: Sum over j=1,2,3 
    part21 = double(ttm(tensor(A), {P1, inv(Sigma2) - P2}, [1, 2]));
    part22 = double(ttm(tensor(A), {inv(Sigma1) - P1, P2}, [1, 2])); 

    result = part1 + part21 + part22; 
end
