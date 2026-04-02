%% Compute W_j^* for each mode
function W_star = compute_W_star(A, j, U1, U2, U3, Sigma1, Sigma2, Sigma3, ...
                                  P1, P2, P3, G)
    % Inputs:
    %   A: input tensor
    %   j: mode index (1, 2, or 3)
    %   U1, U2, U3: orthogonal matrices
    %   Sigma1, Sigma2, Sigma3: diagonal/weight matrices
    %   P1, P2, P3: projection matrices
    %   G: core tensor (will be matricized for mode j)
    
    [p1, p2, p3] = size(A);
    
    % Compute G_j: mode-j matricization of tensor G
    G_j = tenmat(tensor(G), j);
    G_j = double(G_j);
    
    if j == 1
        % W_1^* computation
        p_j = p1;
        Sigma_j = Sigma1;
        P_j = P1;
        
        % Mode-1 unfolding: A_1 is p1 × (p2*p3)
        A_j = tenmat(tensor(A), 1);
        A_j = double(A_j);
        
        % Kronecker product of other modes: U_3 ⊗ U_2
        U_kron = kron(U3, U2);
        
        % Product of (U_k' * Sigma_k * U_k) for k ≠ j
        temp2 = U2' * Sigma2 * U2;
        temp3 = U3' * Sigma3 * U3;
        U_Sigma_prod = kron(temp3, temp2);
        
    elseif j == 2
        % W_2^* computation
        p_j = p2;
        Sigma_j = Sigma2;
        P_j = P2;
        
        % Mode-2 unfolding: A_2 is p2 × (p1*p3)
        A_j = tenmat(tensor(A), 2);
        A_j = double(A_j);
        
        % Kronecker product: U_3 ⊗ U_1
        U_kron = kron(U3, U1);
        
        % Product of (U_k' * Sigma_k * U_k) for k ≠ j
        temp1 = U1' * Sigma1 * U1;
        temp3 = U3' * Sigma3 * U3;
        U_Sigma_prod = kron(temp3, temp1);
        
    else % j == 3
        % W_3^* computation
        p_j = p3;
        Sigma_j = Sigma3;
        P_j = P3;
        
        % Mode-3 unfolding: A_3 is p3 × (p1*p2)
        A_j = tenmat(tensor(A), 3);
        A_j = double(A_j);
        
        % Kronecker product: U_2 ⊗ U_1
        U_kron = kron(U2, U1);
        
        % Product of (U_k' * Sigma_k * U_k) for k ≠ j
        temp1 = U1' * Sigma1 * U1;
        temp2 = U2' * Sigma2 * U2;
        U_Sigma_prod = kron(temp2, temp1);
    end
    
    % Compute (I - Σ_j * P_j)
    I_minus_SigmaP = eye(p_j) - Sigma_j * P_j;
    
    % Compute middle term: A_j * (⊗_{k≠j} U_k) * G_j'
    middle_term = A_j * U_kron * G_j';
    
    % Compute inverse term: (G_j ⊗_{k≠j}(U_k' * Σ_k * U_k) G_j')^{-1}
    inverse_term = G_j * U_Sigma_prod * G_j';
    inverse_term = inv(inverse_term);
    
    % Final computation: W_j^*
    W_star = I_minus_SigmaP * middle_term * inverse_term;
end

