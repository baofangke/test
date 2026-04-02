function [thetahat, sdest] = tsrProj(X, y, T, r)
%TSRPROJ Cross-fitted projection estimator for tensor regression.
%   [THETAHAT, SDEST] = TSRPROJ(X, Y, T, R) computes the cross-fitted
%   estimator THETAHAT for the target tensor T and its estimated standard
%   deviation SDEST in a third-order tensor regression model.
%
%   X may be p1-by-p2-by-p3-by-N or N-by-p1-by-p2-by-p3. T must be a
%   third-order tensor with size p1-by-p2-by-p3. R may be either a scalar
%   Tucker rank or a 1-by-3 / 3-by-1 rank vector.

    narginchk(4, 4);

    y = y(:);
    [T, dimsT] = toNumericTensor(T, 'T');

    X = normalizeDesignArray(X, numel(y), dimsT);
    N = size(X, 4);
    if N < 2
        error('tsrProj:TooFewSamples', ...
            'At least two samples are required.');
    end

    ranks = normalizeRanks(r, dimsT);

    n1 = floor(N / 2);
    n2 = N - n1;
    if n1 == 0 || n2 == 0
        error('tsrProj:InvalidSplit', ...
            'The sample split must leave observations in both folds.');
    end

    index1 = sort(randperm(N, n1));
    index2 = setdiff(1:N, index1, 'stable');

    data1 = buildFold(X(:, :, :, index1), y(index1), ranks);
    data2 = buildFold(X(:, :, :, index2), y(index2), ranks);

    [data1.theta, data1.Phi, data1.evalS1, data1.evalS2, data1.evalS3, data1.evalAvefro, resid1] = foldStatistic(data1, data2, T);
    [data2.theta, data2.Phi, data2.evalS1, data2.evalS2, data2.evalS3, data2.evalAvefro, resid2] = foldStatistic(data2, data1, T);
    thetahat = (data1.theta + data2.theta) / 2;

    sigma2 = (sum(resid1 .^ 2) + sum(resid2 .^ 2)) / N;

    Phi = (data1.Phi(:)' * kron(data1.evalS3, kron(data1.evalS2, data1.evalS1 / data1.evalAvefro^2)) * data1.Phi(:) + ...
        data2.Phi(:)' * kron(data2.evalS3, kron(data2.evalS2, data2.evalS1 / data2.evalAvefro^2)) * data2.Phi(:)) / 2;
    sdest = sqrt(max(sigma2 * Phi, 0));
end

function data = buildFold(Xfold, yfold, ranks)
    [Xcentered, ycentered, Xmean, ymean] = centerFold(Xfold, yfold);
    [Smode1, Smode2, Smode3, avgFroSq] = secondMomentMatrices(Xcentered);
    [init, g, u1, u2, u3] = fitInitialEstimator(Xcentered, ycentered, ranks);

    data = struct( ...
        'X', Xfold, ...
        'y', yfold(:), ...
        'Xmean', Xmean, ...
        'ymean', ymean, ...
        'S1', Smode1, ...
        'S2', Smode2, ...
        'S3', Smode3, ...
        'avefro', avgFroSq, ...
        'init', init, ...
        'g', g, ...
        'u1', u1, ...
        'u2', u2, ...
        'u3', u3, ...
        'theta', [], ...
        'Phi', [], ...
        'evalS1', [], ...
        'evalS2', [], ...
        'evalS3', [], ...
        'evalAvefro', []);
end

function [Xcentered, ycentered, Xmean, ymean] = centerFold(Xfold, yfold, XrefMean, yrefMean)
    if nargin < 3
        Xmean = mean(Xfold, 4);
        ymean = mean(yfold(:));
    else
        Xmean = XrefMean;
        ymean = yrefMean;
    end

    Xcentered = bsxfun(@minus, Xfold, Xmean);
    ycentered = yfold(:) - ymean;
end

function [Smode1, Smode2, Smode3, avgFroSq] = secondMomentMatrices(Xfold)
    dims = size(Xfold);
    p1 = dims(1);
    p2 = dims(2);
    p3 = dims(3);
    n = dims(4);

    Xmode1 = reshape(Xfold, p1, []);
    Smode1 = Xmode1 * Xmode1' / n;

    Xmode2 = reshape(permute(Xfold, [2, 1, 3, 4]), p2, []);
    Smode2 = Xmode2 * Xmode2' / n;

    Xmode3 = reshape(permute(Xfold, [3, 1, 2, 4]), p3, []);
    Smode3 = Xmode3 * Xmode3' / n;

    avgFroSq = mean(reshape(sum(sum(sum(Xfold .^ 2, 1), 2), 3), [], 1));
end

function [init, g, u1, u2, u3] = fitInitialEstimator(Xcentered, ycentered, ranks)

    [model, ~] = fit_tucker_tensor_regression(Xcentered, ycentered, ranks, ...
        'MaxIter', 4000, 'Tol', 1e-7);

    init = model.B;
    g = tensor(model.core, [size(model.U1, 2), size(model.U2, 2), size(model.U3, 2)]);
    u1 = model.U1;
    u2 = model.U2;
    u3 = model.U3;
end

function [theta, Phi, evalS1, evalS2, evalS3, evalAvefro, residVec] = foldStatistic(dataEval, dataTrain, T)
    [XevalCentered, yevalCentered] = centerFold(dataEval.X, dataEval.y, dataTrain.Xmean, dataTrain.ymean);
    [evalS1, evalS2, evalS3, evalAvefro] = secondMomentMatrices(XevalCentered);

    n = size(XevalCentered, 4);
    Xmat = reshape(XevalCentered, [], n);
    residVec = yevalCentered - Xmat' * dataTrain.init(:);
    resid = reshape(residVec, [1, 1, 1, n]);
    Emean = mean(XevalCentered .* resid, 4);

    Mpro = P_sigma(dataTrain.init, dataTrain.g, dataTrain.u1, dataTrain.u2, dataTrain.u3, ...
        dataTrain.S1, dataTrain.S2, dataTrain.S3 / dataTrain.avefro^2);
    Phi = phi_sigma(T, dataTrain.g, dataTrain.u1, dataTrain.u2, dataTrain.u3, ...
        dataTrain.S1, dataTrain.S2, dataTrain.S3 / dataTrain.avefro^2);
    theta = sum(Mpro(:) .* T(:)) + sum(Phi(:) .* Emean(:));
end

function X = normalizeDesignArray(X, nObs, dimsT)
    X = toNumericArray(X, 'X');
    sz = size(X);
    if numel(sz) < 4
        sz(end+1:4) = 1;
    end

    if isequal(sz, [dimsT, nObs])
        return;
    end

    if isequal(sz, [nObs, dimsT])
        X = permute(X, [2, 3, 4, 1]);
        return;
    end

    error('tsrProj:InvalidXSize', ...
        'X must have size p1-by-p2-by-p3-by-N or N-by-p1-by-p2-by-p3, with N = numel(y).');
end

function ranks = normalizeRanks(r, dimsT)
    validateattributes(r, {'numeric'}, {'vector', 'nonempty', 'integer', 'positive'}, ...
        mfilename, 'r');

    if isscalar(r)
        ranks = repmat(double(r), 1, 3);
    else
        if numel(r) ~= 3
            error('tsrProj:InvalidRank', ...
                'r must be a scalar or a length-3 rank vector.');
        end
        ranks = reshape(double(r), 1, 3);
    end

    ranks = min(ranks, dimsT);
end

function [A, dims] = toNumericTensor(A, name)
    A = toNumericArray(A, name);
    sz = size(A);
    if numel(sz) < 3
        sz(end+1:3) = 1;
    end
    if numel(sz) > 3
        error('tsrProj:InvalidTensor', ...
            '%s must be a third-order tensor.', name);
    end
    dims = sz(1:3);
    A = reshape(A, dims(1), dims(2), dims(3));
end

function A = toNumericArray(A, name)
    if isa(A, 'tensor') || isa(A, 'sptensor')
        A = double(A);
    end

    if ~isnumeric(A)
        error('tsrProj:InvalidInput', ...
            '%s must be numeric or a Tensor Toolbox tensor.', name);
    end
end
