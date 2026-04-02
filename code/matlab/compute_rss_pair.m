function [rss_a, rss_b] = compute_rss_pair(A, B, C, init_vec_a, init_vec_b, p, n, blockSize, y)
    rss_a = 0;
    rss_b = 0;
    idx = 1;
    while idx <= n
        nb = min(blockSize, n - idx + 1);
        S_block = randn(p, p, p, nb);
        X_block = ttm(tensor(S_block), {A, B, C}, [1, 2, 3]);
        X_mat = reshape(X_block.data, [], nb);
        y_block = y(idx:idx+nb-1);

        resid_a = y_block - X_mat' * init_vec_a;
        resid_b = y_block - X_mat' * init_vec_b;
        rss_a = rss_a + sum(resid_a.^2);
        rss_b = rss_b + sum(resid_b.^2);

        randn(nb, 1); % keep RNG in sync with data generation
        idx = idx + nb;
    end
end
