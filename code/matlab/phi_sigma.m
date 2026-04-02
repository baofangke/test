function result = phi_sigma(A, G, U1, U2, U3, Sigma1, Sigma2, Sigma3)
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

% Compute projection operators: P_U^Sigma = Sigma^{-1} * U * U' * Sigma
    temp1 = U1' * Sigma1 * U1;
    P1 = U1 * (temp1 \ U1'); 
    temp2 = U2' * Sigma2 * U2;
    P2 = U2 * (temp2 \ U2'); 
    temp3 = U3' * Sigma3 * U3;
    P3 = U3 * (temp3 \ U3');

% Part 1: A ×₁ P₁ ×₂ P₂ ×₃ P₃
part1 = ttm(tensor(A), {P1, P2, P3}, [1, 2, 3]);
part1 = double(part1);

% Part 2: Sum over j=1,2,3
W1_star = compute_W_star(A, 1, U1, U2, U3, Sigma1, Sigma2, Sigma3, ...
                             P1, P2, P3, G);
W2_star = compute_W_star(A, 2, U1, U2, U3, Sigma1, Sigma2, Sigma3, ...
                         P1, P2, P3, G);
W3_star = compute_W_star(A, 3, U1, U2, U3, Sigma1, Sigma2, Sigma3, ...
                         P1, P2, P3, G);  
                     
G_t = tensor(G); 
% Compute G ×_{k≠j} U_k ×_j Σ_j Σ_j^{-1} W_j^*
term2_j1 = ttm(G_t, {Sigma1 \ W1_star, U2, U3}, [1, 2, 3]); 
term2_j2 = ttm(G_t, {U1, Sigma2 \ W2_star, U3}, [1, 2, 3]); 
term2_j3 = ttm(G_t, {U1, U2, Sigma3 \ W3_star}, [1, 2, 3]); 
part2 = term2_j1 + term2_j2 + term2_j3;


result = part1 + part2;
end
