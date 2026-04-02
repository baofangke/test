function [model, info] = fit_tucker_tensor_regression(Xtrain, ytrain, ranks, varargin)
%FIT_TUCKER_TENSOR_REGRESSION Fit low multilinear-rank tensor regression.
%
%   MODEL = FIT_TUCKER_TENSOR_REGRESSION(XTRAIN, YTRAIN, RANKS) fits a
%   Tucker low-rank coefficient tensor for scalar responses YTRAIN. XTRAIN
%   may be n-by-p1-by-p2-by-p3 or p1-by-p2-by-p3-by-n.
%
%   Optional name/value pairs:
%       'MaxIter'  Maximum projected-gradient iterations, default 200
%       'Tol'      Relative stopping tolerance, default 1e-6

    narginchk(3, 7);

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'MaxIter', 200, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1);
    addParameter(parser, 'Tol', 1e-6, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    parse(parser, varargin{:});

    ytrain = ytrain(:);
    [Xtrain, dims] = normalize_training_array(Xtrain, numel(ytrain));
    validateattributes(ranks, {'numeric'}, {'vector', 'numel', 3, 'integer', 'positive'}, ...
        mfilename, 'ranks');
    ranks = min(ranks(:)', dims);

    n = numel(ytrain);
    Xmean = reshape(mean(Xtrain, 1), dims);
    Xcentered = bsxfun(@minus, Xtrain, reshape(Xmean, [1, dims]));
    ymean = mean(ytrain);
    ycentered = ytrain - ymean;

    Z = reshape(Xcentered, n, []);
    if isempty(Z)
        error('fit_tucker_tensor_regression:EmptyDesign', ...
            'Xtrain must contain at least one observation.');
    end

    if exist('lsqminnorm', 'file') == 2
        beta0 = lsqminnorm(Z, ycentered);
    else
        beta0 = pinv(Z) * ycentered;
    end
    B = reshape(beta0, dims);
    [B, U1, U2, U3, core] = project_tucker_tensor(B, ranks);

    s = svd(Z, 'econ');
    if isempty(s)
        smax = 0;
    else
        smax = s(1);
    end
    lipschitz = max((smax * smax) / max(n, 1), 1e-8);
    step = 1.0 / lipschitz;

    prevObj = inf;
    nIter = 0;

    for iteration = 1:parser.Results.MaxIter
        residual = Z * B(:) - ycentered;
        grad = reshape((Z' * residual) / n, dims);
        candidate = B - step * grad;
        [candidate, U1, U2, U3, core] = project_tucker_tensor(candidate, ranks);

        residualNew = Z * candidate(:) - ycentered;
        obj = 0.5 * mean(residualNew .^ 2);
        relChange = norm(candidate(:) - B(:)) / max(norm(B(:)), 1e-12);

        if ~isfinite(prevObj)
            objChange = inf;
        else
            objChange = abs(prevObj - obj) / max(abs(prevObj), 1.0);
        end

        B = candidate;
        prevObj = obj;
        nIter = iteration;

        if relChange < parser.Results.Tol || objChange < parser.Results.Tol
            break;
        end
    end

    intercept = ymean - dot(Xmean(:), B(:));
    trainPred = ymean + Z * B(:);
    trainMSE = mean((ytrain - trainPred) .^ 2);

    model = struct( ...
        'B', B, ...
        'core', core, ...
        'U1', U1, ...
        'U2', U2, ...
        'U3', U3, ...
        'ranks', ranks, ...
        'dims', dims, ...
        'Xmean', Xmean, ...
        'ymean', ymean, ...
        'intercept', intercept);

    info = struct( ...
        'step_size', step, ...
        'lipschitz', lipschitz, ...
        'n_iter', nIter, ...
        'objective', prevObj, ...
        'train_mse', trainMSE);
end

function [Xtrain, dims] = normalize_training_array(Xtrain, nObs)
    if ndims(Xtrain) ~= 4
        error('fit_tucker_tensor_regression:InvalidX', ...
            'Xtrain must be a four-dimensional array.');
    end

    sz = size(Xtrain);
    if sz(1) == nObs
        dims = sz(2:4);
        return;
    end

    if sz(4) == nObs
        Xtrain = permute(Xtrain, [4, 1, 2, 3]);
        dims = size(Xtrain);
        dims = dims(2:4);
        return;
    end

    error('fit_tucker_tensor_regression:InvalidXSize', ...
        'Xtrain must have size n-by-p1-by-p2-by-p3 or p1-by-p2-by-p3-by-n.');
end

function [Bproj, U1, U2, U3, core] = project_tucker_tensor(B, ranks)
    dims = size(B);
    ranks = min(ranks(:)', dims);

    U1 = leading_left_vectors(reshape(B, dims(1), []), ranks(1));
    U2 = leading_left_vectors(reshape(permute(B, [2, 1, 3]), dims(2), []), ranks(2));
    U3 = leading_left_vectors(reshape(permute(B, [3, 1, 2]), dims(3), []), ranks(3));

    core = reshape(double(ttm(tensor(B), {U1', U2', U3'}, [1, 2, 3])), ranks);
    Bproj = reconstruct_tucker_tensor(core, U1, U2, U3);
end

function U = leading_left_vectors(M, rankValue)
    [U, ~, ~] = svd(M, 'econ');
    keep = min(rankValue, size(U, 2));
    U = U(:, 1:keep);
end

function B = reconstruct_tucker_tensor(core, U1, U2, U3)
    r1 = size(core, 1);
    r2 = size(core, 2);
    r3 = size(core, 3);

    tmp = U1 * reshape(core, r1, []);
    tmp = reshape(tmp, size(U1, 1), r2, r3);

    tmp = permute(tmp, [2, 1, 3]);
    tmp = U2 * reshape(tmp, r2, []);
    tmp = reshape(tmp, size(U2, 1), size(U1, 1), r3);
    tmp = permute(tmp, [2, 1, 3]);

    tmp = permute(tmp, [3, 1, 2]);
    tmp = U3 * reshape(tmp, r3, []);
    tmp = reshape(tmp, size(U3, 1), size(U1, 1), size(U2, 1));
    B = permute(tmp, [2, 3, 1]);
end
