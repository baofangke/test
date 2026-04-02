function [B, info] = fit_low_rank_trace_regression(Xtrain, ytrain, rank, varargin)
%FIT_LOW_RANK_TRACE_REGRESSION Projected-gradient solver for fixed-rank trace regression.
%
%   B = FIT_LOW_RANK_TRACE_REGRESSION(XTRAIN, YTRAIN, RANK) estimates a
%   p-by-q coefficient matrix B under the hard constraint rank(B) <= RANK.
%   XTRAIN may be p-by-q-by-n or n-by-p-by-q.
%
%   [B, INFO] = FIT_LOW_RANK_TRACE_REGRESSION(...) also returns solver
%   diagnostics in INFO.
%
%   Optional name/value pairs:
%       'MaxIter'  Maximum iterations, default 1000
%       'Tol'      Relative stopping tolerance, default 1e-7

    narginchk(3, 7);

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'MaxIter', 1000, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1);
    addParameter(parser, 'Tol', 1e-7, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    parse(parser, varargin{:});

    ytrain = ytrain(:);
    Xtrain = normalize_training_array(Xtrain, numel(ytrain));
    [p, q, n] = size(Xtrain);

    validateattributes(rank, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'rank');
    rank = min(rank, min(p, q));

    Z = reshape(Xtrain, [], n)';
    if isempty(Z)
        error('fit_low_rank_trace_regression:EmptyDesign', ...
            'Xtrain must contain at least one observation.');
    end

    s = svd(Z, 'econ');
    if isempty(s)
        smax = 0;
    else
        smax = s(1);
    end
    lipschitz = max((smax * smax) / max(n, 1), 1e-8);
    step = 1.0 / lipschitz;

    B = zeros(p, q);
    prevObj = inf;
    nIter = 0;

    for iteration = 1:parser.Results.MaxIter
        residual = Z * B(:) - ytrain;
        grad = reshape((Z' * residual) / n, p, q);
        candidate = project_rank_matrix(B - step * grad, rank);
        residualNew = Z * candidate(:) - ytrain;
        obj = 0.5 * mean(residualNew .^ 2);
        relChange = norm(candidate - B, 'fro') / max(norm(B, 'fro'), 1e-12);

        B = candidate;
        nIter = iteration;

        if ~isfinite(prevObj)
            objChange = inf;
        else
            objChange = abs(prevObj - obj) / max(abs(prevObj), 1.0);
        end

        if relChange < parser.Results.Tol || objChange < parser.Results.Tol
            prevObj = obj;
            break;
        end
        prevObj = obj;
    end

    singularValues = svd(B, 'econ');
    info = struct( ...
        'step_size', step, ...
        'lipschitz', lipschitz, ...
        'n_iter', nIter, ...
        'objective', prevObj, ...
        'effective_rank', sum(singularValues > 1e-8));
end

function X = normalize_training_array(X, nObs)
    if ndims(X) ~= 3
        error('fit_low_rank_trace_regression:InvalidX', ...
            'Xtrain must be a three-dimensional array.');
    end

    sz = size(X);
    if sz(3) == nObs
        return;
    end
    if sz(1) == nObs
        X = permute(X, [2, 3, 1]);
        return;
    end

    error('fit_low_rank_trace_regression:InvalidXSize', ...
        'Xtrain must have size p-by-q-by-n or n-by-p-by-q, with n = numel(ytrain).');
end

function Bproj = project_rank_matrix(B, rank)
    [U, S, V] = svd(B, 'econ');
    keep = min(rank, size(S, 1));
    if keep <= 0
        Bproj = zeros(size(B), 'like', B);
        return;
    end
    Bproj = U(:, 1:keep) * S(1:keep, 1:keep) * V(:, 1:keep)';
end
