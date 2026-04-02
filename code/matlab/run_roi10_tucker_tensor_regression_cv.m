function outputs = run_roi10_tucker_tensor_regression_cv(varargin)
%RUN_ROI10_TUCKER_TENSOR_REGRESSION_CV Select Tucker rank for ROI10 age prediction.
%
%   OUTPUTS = RUN_ROI10_TUCKER_TENSOR_REGRESSION_CV() constructs a
%   10-by-8-by-2 tensor covariate per subject from log10(raw EC) and
%   log10(raw EO), performs 5-fold cross-validation over the Tucker rank
%   grid {1:10} x {1:5} x {1:2}, and refits the selected model on all
%   subjects.

    opts = parse_options(varargin{:});

    if opts.setup_paths
        paths = setup_hbn_matlab_path(opts.code_root, opts.tensor_toolbox_root, true);
        opts.tensor_toolbox_root = paths.tensor_toolbox_root;
    end

    data = load_roi10_tensor_data(opts.mat_file);
    foldId = make_cv_folds(numel(data.age), opts.n_folds, opts.split_seed);
    rankGrid = make_rank_grid(opts.rank1_list, opts.rank2_list, opts.rank3_list);

    outputDir = opts.output_dir;
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    nCandidates = size(rankGrid, 1);
    detailRows = cell(nCandidates * opts.n_folds, 1);
    summaryRows = cell(nCandidates, 1);
    rowIdx = 0;

    for candidateIdx = 1:nCandidates
        ranks = rankGrid(candidateIdx, :);
        foldMSE = zeros(opts.n_folds, 1);
        foldR2 = zeros(opts.n_folds, 1);

        fprintf('Rank [%d %d %d] (%d/%d)\n', ...
            ranks(1), ranks(2), ranks(3), candidateIdx, nCandidates);

        for fold = 1:opts.n_folds
            trainMask = foldId ~= fold;
            valMask = foldId == fold;

            [model, info] = fit_tucker_tensor_regression( ...
                data.X(trainMask, :, :, :), data.age(trainMask), ranks, ...
                'MaxIter', opts.max_iter, 'Tol', opts.tol);
            yhat = predict_tucker_tensor_regression(model, data.X(valMask, :, :, :));
            yval = data.age(valMask);

            foldMSE(fold) = mean((yval - yhat) .^ 2);
            denom = sum((yval - mean(yval)) .^ 2);
            if denom > 0
                foldR2(fold) = 1 - sum((yval - yhat) .^ 2) / denom;
            else
                foldR2(fold) = NaN;
            end

            rowIdx = rowIdx + 1;
            detailRows{rowIdx} = table( ...
                fold, ranks(1), ranks(2), ranks(3), ...
                sum(trainMask), sum(valMask), foldMSE(fold), foldR2(fold), ...
                info.n_iter, info.train_mse, ...
                'VariableNames', { ...
                    'fold', 'rank1', 'rank2', 'rank3', ...
                    'n_train', 'n_val', 'val_mse', 'val_r2', ...
                    'fit_iterations', 'train_mse'});
        end

        summaryRows{candidateIdx} = table( ...
            ranks(1), ranks(2), ranks(3), sum(ranks), ...
            mean(foldMSE), std(foldMSE), mean(foldR2, 'omitnan'), std(foldR2, 'omitnan'), ...
            'VariableNames', { ...
                'rank1', 'rank2', 'rank3', 'rank_sum', ...
                'mean_val_mse', 'std_val_mse', 'mean_val_r2', 'std_val_r2'});
    end

    cvDetails = vertcat(detailRows{1:rowIdx});
    cvSummary = vertcat(summaryRows{:});
    cvSummary = sortrows(cvSummary, {'mean_val_mse', 'rank_sum', 'rank1', 'rank2', 'rank3'});
    bestRank = [cvSummary.rank1(1), cvSummary.rank2(1), cvSummary.rank3(1)];

    [finalModel, finalInfo] = fit_tucker_tensor_regression( ...
        data.X, data.age, bestRank, 'MaxIter', opts.max_iter, 'Tol', opts.tol);
    yhatFull = predict_tucker_tensor_regression(finalModel, data.X);
    fullMSE = mean((data.age - yhatFull) .^ 2);
    fullDenom = sum((data.age - mean(data.age)) .^ 2);
    if fullDenom > 0
        fullR2 = 1 - sum((data.age - yhatFull) .^ 2) / fullDenom;
    else
        fullR2 = NaN;
    end

    selectedCv = cvDetails(cvDetails.rank1 == bestRank(1) & ...
        cvDetails.rank2 == bestRank(2) & cvDetails.rank3 == bestRank(3), :);

    detailsCsv = fullfile(outputDir, 'tucker_tensor_regression_cv_details.csv');
    summaryCsv = fullfile(outputDir, 'tucker_tensor_regression_cv_summary.csv');
    coeffCsv = fullfile(outputDir, 'tucker_tensor_regression_best_model_coefficients.csv');
    resultsMat = fullfile(outputDir, 'tucker_tensor_regression_best_model.mat');
    summaryTxt = fullfile(outputDir, 'tucker_tensor_regression_summary.txt');
    cvPlot = fullfile(outputDir, 'tucker_tensor_regression_cv_mean_mse.png');
    coefPlot = fullfile(outputDir, 'tucker_tensor_regression_best_model_coefficients.png');

    writetable(cvDetails, detailsCsv);
    writetable(cvSummary, summaryCsv);
    coeffTable = coefficient_table(finalModel.B, data.roi_names, data.band_names, data.condition_names);
    writetable(coeffTable, coeffCsv);

    save(resultsMat, ...
        'finalModel', 'finalInfo', 'bestRank', 'cvSummary', 'cvDetails', 'selectedCv', ...
        'yhatFull', 'fullMSE', 'fullR2', 'foldId', 'data', 'opts');

    write_summary_text(summaryTxt, opts, bestRank, cvSummary(1, :), selectedCv, fullMSE, fullR2);
    plot_cv_grid(cvSummary, opts.rank1_list, opts.rank2_list, opts.rank3_list, cvPlot);
    plot_coefficient_slices(finalModel.B, data.roi_names, data.band_names, data.condition_names, coefPlot);

    outputs = struct();
    outputs.best_rank = bestRank;
    outputs.cv_summary = cvSummary;
    outputs.cv_details = cvDetails;
    outputs.selected_cv = selectedCv;
    outputs.final_model = finalModel;
    outputs.final_info = finalInfo;
    outputs.full_mse = fullMSE;
    outputs.full_r2 = fullR2;
    outputs.output_dir = outputDir;
end

function opts = parse_options(varargin)
    paths = resolve_release_paths();
    defaults = struct( ...
        'code_root', paths.code_root, ...
        'tensor_toolbox_root', '', ...
        'mat_file', paths.roi10_mat_file, ...
        'output_dir', paths.tensor_cv_dir, ...
        'n_folds', 5, ...
        'split_seed', 20260321, ...
        'rank1_list', 1:10, ...
        'rank2_list', 1:5, ...
        'rank3_list', 1:2, ...
        'max_iter', 4000, ...
        'tol', 1e-6, ...
        'setup_paths', true);

    if nargin == 0
        opts = defaults;
        return;
    end

    parser = inputParser;
    parser.FunctionName = mfilename;
    names = fieldnames(defaults);
    for idx = 1:numel(names)
        addParameter(parser, names{idx}, defaults.(names{idx}));
    end
    parse(parser, varargin{:});
    opts = parser.Results;
end

function data = load_roi10_tensor_data(matFile)
    mat = load(matFile);
    rawEO = double(mat.EO_roi_power_uV2);
    rawEC = double(mat.EC_roi_power_uV2);

    positivesRaw = [rawEO(rawEO > 0); rawEC(rawEC > 0)];
    if isempty(positivesRaw)
        error('run_roi10_tucker_tensor_regression_cv:NoPositiveRawValues', ...
            'Raw EO/EC arrays have no positive entries.');
    end
    rawFloor = max(min(positivesRaw) * 0.5, 1e-12);

    eoLog = log10(max(rawEO, rawFloor));
    ecLog = log10(max(rawEC, rawFloor));

    data = struct();
    data.X = cat(4, ecLog, eoLog);
    data.age = double(mat.age(:));
    data.roi_names = cellstr_from_mat(mat.roi_names);
    data.band_names = cellstr_from_mat(mat.band_names);
    data.condition_names = {'EC', 'EO'};
end

function foldId = make_cv_folds(nObs, nFolds, seed)
    rng(seed, 'twister');
    order = randperm(nObs);
    foldId = zeros(nObs, 1);
    foldId(order) = mod(0:nObs-1, nFolds) + 1;
end

function rankGrid = make_rank_grid(rank1List, rank2List, rank3List)
    [g1, g2, g3] = ndgrid(rank1List, rank2List, rank3List);
    rankGrid = [g1(:), g2(:), g3(:)];
end

function coeffTable = coefficient_table(B, roiNames, bandNames, conditionNames)
    [p1, p2, p3] = size(B);
    nRows = p1 * p2 * p3;
    rows = repmat(struct( ...
        'roi_index', NaN, ...
        'band_index', NaN, ...
        'condition_index', NaN, ...
        'roi_name', "", ...
        'band_name', "", ...
        'condition_name', "", ...
        'coefficient', NaN), nRows, 1);

    idx = 0;
    for i = 1:p1
        for j = 1:p2
            for k = 1:p3
                idx = idx + 1;
                rows(idx).roi_index = i;
                rows(idx).band_index = j;
                rows(idx).condition_index = k;
                rows(idx).roi_name = string(roiNames{i});
                rows(idx).band_name = string(bandNames{j});
                rows(idx).condition_name = string(conditionNames{k});
                rows(idx).coefficient = B(i, j, k);
            end
        end
    end

    coeffTable = struct2table(rows);
end

function write_summary_text(outputFile, opts, bestRank, bestRow, selectedCv, fullMSE, fullR2)
    fid = fopen(outputFile, 'w');
    if fid < 0
        error('run_roi10_tucker_tensor_regression_cv:SummaryWriteFailed', ...
            'Could not write summary text: %s', outputFile);
    end

    fprintf(fid, 'ROI10 Tucker tensor regression CV summary\n');
    fprintf(fid, 'Tensor shape: 10 x 8 x 2 (ROI x Band x Condition)\n');
    fprintf(fid, 'Transform: log10(raw EC) and log10(raw EO)\n');
    fprintf(fid, 'Response: age\n');
    fprintf(fid, 'Number of folds: %d\n', opts.n_folds);
    fprintf(fid, 'Split seed: %d\n', opts.split_seed);
    fprintf(fid, 'Candidate ranks: rank1=%s rank2=%s rank3=%s\n', ...
        mat2str(opts.rank1_list), mat2str(opts.rank2_list), mat2str(opts.rank3_list));
    fprintf(fid, 'Selected rank: [%d %d %d]\n', bestRank(1), bestRank(2), bestRank(3));
    fprintf(fid, 'Mean CV MSE: %.6f\n', bestRow.mean_val_mse);
    fprintf(fid, 'Mean CV R2: %.6f\n', bestRow.mean_val_r2);
    fprintf(fid, 'Selected-rank fold MSEs: %s\n', mat2str(selectedCv.val_mse', 6));
    fprintf(fid, 'Selected-rank fold R2s: %s\n', mat2str(selectedCv.val_r2', 6));
    fprintf(fid, 'Full-sample MSE: %.6f\n', fullMSE);
    fprintf(fid, 'Full-sample R2: %.6f\n', fullR2);
    fclose(fid);
end

function plot_cv_grid(cvSummary, rank1List, rank2List, rank3List, outputFile)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 500]);
    tiledlayout(fig, 1, numel(rank3List), 'TileSpacing', 'compact', 'Padding', 'compact');

    for idx = 1:numel(rank3List)
        ax = nexttile;
        slice = cvSummary(cvSummary.rank3 == rank3List(idx), :);
        mseMat = nan(numel(rank1List), numel(rank2List));
        for row = 1:height(slice)
            i = find(rank1List == slice.rank1(row), 1);
            j = find(rank2List == slice.rank2(row), 1);
            mseMat(i, j) = slice.mean_val_mse(row);
        end
        imagesc(ax, rank2List, rank1List, mseMat);
        axis(ax, 'tight');
        colorbar(ax);
        xlabel(ax, 'rank2');
        ylabel(ax, 'rank1');
        title(ax, sprintf('rank3 = %d', rank3List(idx)));
    end

    exportgraphics(fig, outputFile, 'Resolution', 200);
    close(fig);
end

function plot_coefficient_slices(B, roiNames, bandNames, conditionNames, outputFile)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 500]);
    tiledlayout(fig, 1, numel(conditionNames), 'TileSpacing', 'compact', 'Padding', 'compact');
    maxAbs = max(abs(B(:)));

    for idx = 1:numel(conditionNames)
        ax = nexttile;
        imagesc(ax, B(:, :, idx));
        axis(ax, 'tight');
        colorbar(ax);
        if maxAbs > 0
            caxis(ax, [-maxAbs, maxAbs]);
        end
        title(ax, sprintf('Coefficient Slice: %s', conditionNames{idx}), 'Interpreter', 'none');
        xticks(ax, 1:numel(bandNames));
        xticklabels(ax, bandNames);
        yticks(ax, 1:numel(roiNames));
        yticklabels(ax, roiNames);
        xtickangle(ax, 45);
        xlabel(ax, 'Band');
        ylabel(ax, 'ROI');
    end

    exportgraphics(fig, outputFile, 'Resolution', 200);
    close(fig);
end

function out = cellstr_from_mat(values)
    flat = values(:);
    out = cell(numel(flat), 1);
    for idx = 1:numel(flat)
        item = flat(idx);
        if iscell(item)
            item = item{1};
        end
        if isstring(item)
            out{idx} = char(item);
        elseif ischar(item)
            out{idx} = item;
        else
            out{idx} = char(string(item));
        end
    end
end
