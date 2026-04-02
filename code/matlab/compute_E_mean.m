function E_mean = compute_E_mean(A, B, C, init_vec, p, n, blockSize, y)
    E_sum = zeros(p, p, p);
    idx = 1;
    while idx <= n
        nb = min(blockSize, n - idx + 1);
        S_block = randn(p, p, p, nb);
        X_block = ttm(tensor(S_block), {A, B, C}, [1, 2, 3]); 
        X_block = X_block.data;
        X_mat = reshape(X_block, [], nb);
        resid = y(idx:idx+nb-1) - X_mat' * init_vec;
        E_sum = E_sum + sum(X_block .* reshape(resid, [1, 1, 1, nb]), 4);
        randn(nb, 1); % keep RNG in sync with data generation
        idx = idx + nb;
    end
    E_mean = E_sum / n;
end