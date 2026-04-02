function yhat = predict_tucker_tensor_regression(model, X)
%PREDICT_TUCKER_TENSOR_REGRESSION Predict responses for tensor covariates.

    [X, dims] = normalize_prediction_array(X, model.dims);
    Xcentered = bsxfun(@minus, X, reshape(model.Xmean, [1, dims]));
    Z = reshape(Xcentered, size(Xcentered, 1), []);
    yhat = model.ymean + Z * model.B(:);
end

function [X, dims] = normalize_prediction_array(X, expectedDims)
    if ndims(X) ~= 4
        error('predict_tucker_tensor_regression:InvalidX', ...
            'X must be a four-dimensional array.');
    end

    sz = size(X);
    if isequal(sz(2:4), expectedDims)
        dims = expectedDims;
        return;
    end

    if isequal(sz(1:3), expectedDims)
        X = permute(X, [4, 1, 2, 3]);
        dims = expectedDims;
        return;
    end

    error('predict_tucker_tensor_regression:InvalidXSize', ...
        'X dimensions do not match the fitted model.');
end
