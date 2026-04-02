function [thetahat, sdest] = matProj(X, y, T, r)
%MATPROJ Cross-fitted projection estimator for matrix regression.
%   [THETAHAT, SDEST] = MATPROJ(X, Y, T, R) computes the cross-fitted
%   estimator THETAHAT for the target matrix T and its estimated standard
%   deviation SDEST. X can be either p-by-q-by-N or N-by-p-by-q, and may
%   be numeric or a Tensor Toolbox tensor.

    narginchk(4, 4);

    y = y(:);
    T = toNumericMatrix(T, 'T');
    [p, q] = size(T);

    X = normalizeDesignArray(X, numel(y), p, q);
    N = size(X, 3);
    if N < 2
        error('matProj:TooFewSamples', ...
            'At least two samples are required.');
    end

    validateattributes(r, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'r');
    r = min(r, min(p, q));

    n1 = floor(N / 2);
    n2 = N - n1;
    if n1 == 0 || n2 == 0
        error('matProj:InvalidSplit', ...
            'The sample split must leave observations in both folds.');
    end

    index1 = sort(randperm(N, n1));
    index2 = setdiff(1:N, index1, 'stable');

    data1 = buildFold(X(:, :, index1), y(index1), r);
    data2 = buildFold(X(:, :, index2), y(index2), r);

    [data1.theta, data1.Phi, data1.evalS1, data1.evalS2, data1.evalAvefro, resid1] = foldStatistic(data1, data2, T);
    [data2.theta, data2.Phi, data2.evalS1, data2.evalS2, data2.evalAvefro, resid2] = foldStatistic(data2, data1, T);
    thetahat = (data1.theta + data2.theta) / 2;

    sigma2 = (sum(resid1 .^ 2) + sum(resid2 .^ 2)) / N;

    Phi = (data1.Phi(:)' * kron(data1.evalS2 / data1.evalAvefro, data1.evalS1) * data1.Phi(:) + ...
        data2.Phi(:)' * kron(data2.evalS2 / data2.evalAvefro, data2.evalS1) * data2.Phi(:)) / 2;
    sdest = sqrt(max(sigma2 * Phi, 0));
end

function data = buildFold(Xfold, yfold, r)
    [Xcentered, ycentered, Xmean, ymean] = centerFold(Xfold, yfold);
    [S_mode1, S_mode2, avg_fro_sq] = secondMomentMatrices(Xcentered);
    [init, u1, u2] = fitInitialEstimator(Xcentered, ycentered, r);

    data = struct( ...
        'X', Xfold, ...
        'y', yfold(:), ...
        'Xmean', Xmean, ...
        'ymean', ymean, ...
        'S1', S_mode1, ...
        'S2', S_mode2, ...
        'avefro', avg_fro_sq, ...
        'init', init, ...
        'u1', u1, ...
        'u2', u2, ...
        'theta', [], ...
        'Phi', [], ...
        'evalS1', [], ...
        'evalS2', [], ...
        'evalAvefro', []);
end

function [Xcentered, ycentered, Xmean, ymean] = centerFold(Xfold, yfold, XrefMean, yrefMean)
    if nargin < 3
        Xmean = mean(Xfold, 3);
        ymean = mean(yfold(:));
    else
        Xmean = XrefMean;
        ymean = yrefMean;
    end

    Xcentered = Xfold - Xmean;
    ycentered = yfold(:) - ymean;
end

function [S_mode1, S_mode2, avg_fro_sq] = secondMomentMatrices(Xfold)
    [p, q, n] = size(Xfold);

    X_mode1 = reshape(Xfold, p, []);
    S_mode1 = X_mode1 * X_mode1' / n;
    X_mode2 = reshape(permute(Xfold, [2, 1, 3]), q, []);
    S_mode2 = X_mode2 * X_mode2' / n;
    avg_fro_sq = mean(reshape(sum(Xfold .^ 2, [1, 2]), [], 1));
end

function [init, u1, u2] = fitInitialEstimator(Xcentered, ycentered, r)
    [init, ~] = fit_low_rank_trace_regression(Xcentered, ycentered, r, ...
        'MaxIter', 1000, 'Tol', 1e-7);
    [u, d, v] = svd(init, 'econ');

    rEff = min([r, size(u, 2), size(v, 2)]);
    init = u(:, 1:rEff) * d(1:rEff, 1:rEff) * v(:, 1:rEff)';
    u1 = u(:, 1:rEff);
    u2 = v(:, 1:rEff);
end

function [theta, Phi, evalS1, evalS2, evalAvefro, residVec] = foldStatistic(dataEval, dataTrain, T)
    [XevalCentered, yevalCentered] = centerFold(dataEval.X, dataEval.y, dataTrain.Xmean, dataTrain.ymean);
    [evalS1, evalS2, evalAvefro] = secondMomentMatrices(XevalCentered);

    n = size(XevalCentered, 3);
    X_mat = reshape(XevalCentered, [], n);
    residVec = yevalCentered - X_mat' * dataTrain.init(:);
    resid = reshape(residVec, [1, 1, n]);
    E_mean = mean(XevalCentered .* resid, 3);

    M_pro = P_sigma_mat(dataTrain.init, dataTrain.u1, dataTrain.u2, dataTrain.S1, ...
        dataTrain.S2 / dataTrain.avefro);
    Phi = phi_sigma_mat(T, dataTrain.u1, dataTrain.u2, dataTrain.S1, ...
        dataTrain.S2 / dataTrain.avefro);
    theta = sum(M_pro(:) .* T(:)) + sum(Phi(:) .* E_mean(:));
end

function X = normalizeDesignArray(X, nObs, p, q)
    X = toNumericArray(X, 'X');
    if ndims(X) ~= 3
        error('matProj:InvalidX', ...
            'X must be a three-dimensional array or tensor.');
    end

    sz = size(X);
    if isequal(sz, [p, q, nObs])
        return;
    end

    if isequal(sz, [nObs, p, q])
        X = permute(X, [2, 3, 1]);
        return;
    end

    error('matProj:InvalidXSize', ...
        'X must have size p-by-q-by-N or N-by-p-by-q, with N = numel(y).');
end

function A = toNumericMatrix(A, name)
    A = toNumericArray(A, name);
    if ~ismatrix(A)
        error('matProj:InvalidMatrix', ...
            '%s must be a matrix.', name);
    end
end

function A = toNumericArray(A, name)
    if isa(A, 'tensor') || isa(A, 'sptensor')
        A = double(A);
    end

    if ~isnumeric(A)
        error('matProj:InvalidInput', ...
            '%s must be numeric or a Tensor Toolbox tensor.', name);
    end
end
