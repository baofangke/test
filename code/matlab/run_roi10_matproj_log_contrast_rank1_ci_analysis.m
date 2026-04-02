function results = run_roi10_matproj_log_contrast_rank1_ci_analysis(varargin)
%RUN_ROI10_MATPROJ_LOG_CONTRAST_RANK1_CI_ANALYSIS
% Evaluate Frobenius-normalized loadings with matProj for
% X = log10(EC) - log10(EO) at rank 1.

    opts = parse_options(varargin{:});

    if opts.setup_paths
        paths = setup_hbn_matlab_path(opts.code_root, opts.tensor_toolbox_root, false);
        opts.tensor_toolbox_root = paths.tensor_toolbox_root;
    end

    data = load_roi10_data(opts.mat_file);
    [X, rawFloor] = build_log_contrast(data.EO, data.EC);
    cvInfo = read_cv_summary(opts.cv_summary_csv);

    outputDir = opts.output_dir;
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    fprintf('Running log10(EC)-log10(EO) matProj inference with rank %d on %d subjects\n', ...
        opts.rank, numel(data.age));

    results = evaluate_all_loadings(X, data.age, opts.rank, ...
        data.roi_names, data.band_names, opts.split_seed);
    nRows = height(results);
    results.analysis_name = repmat("log EC - log EO", nRows, 1);
    results.condition = repmat("EC_minus_EO", nRows, 1);
    results.transform = repmat("log10", nRows, 1);
    results.rank = repmat(opts.rank, nRows, 1);
    results.cv_mean_val_mse = repmat(cvInfo.mean_val_mse, nRows, 1);
    results.cv_mean_val_r2 = repmat(cvInfo.mean_val_r2, nRows, 1);

    results = sortrows(results, {'loading_type', 'roi_index', 'band_index', 'loading_label'});

    resultsCsv = fullfile(outputDir, 'matproj_log_contrast_rank1_all_results.csv');
    summaryCsv = fullfile(outputDir, 'matproj_log_contrast_rank1_summary.csv');
    configJson = fullfile(outputDir, 'matproj_log_contrast_rank1_config.json');
    summaryTxt = fullfile(outputDir, 'matproj_log_contrast_rank1_summary.txt');
    heatmapFile = fullfile(outputDir, 'log_contrast_rank1_single_element_heatmap.png');
    structuredFile = fullfile(outputDir, 'log_contrast_rank1_structured_loadings.png');

    writetable(results, resultsCsv);

    summaryTable = table( ...
        "log EC - log EO", "EC_minus_EO", "log10", opts.rank, ...
        cvInfo.mean_val_mse, cvInfo.mean_val_r2, ...
        sum(results.loading_type == "single_element" & results.excludes_zero), ...
        sum(results.loading_type == "row" & results.excludes_zero), ...
        sum(results.loading_type == "column" & results.excludes_zero), ...
        sum(results.loading_type == "full_matrix" & results.excludes_zero), ...
        'VariableNames', { ...
            'analysis_name', 'condition', 'transform', 'rank', ...
            'cv_mean_val_mse', 'cv_mean_val_r2', ...
            'single_element_excludes_zero', 'row_excludes_zero', ...
            'column_excludes_zero', 'full_matrix_excludes_zero'});
    writetable(summaryTable, summaryCsv);

    plot_single_element_heatmap(results, data.roi_names, data.band_names, ...
        'log EC - log EO Single-Element Loadings', heatmapFile);
    plot_structured_loadings(results, ...
        'log EC - log EO Structured Loadings', structuredFile);

    config = struct();
    config.mat_file = opts.mat_file;
    config.output_dir = outputDir;
    config.code_root = opts.code_root;
    config.tensor_toolbox_root = opts.tensor_toolbox_root;
    config.cv_summary_csv = opts.cv_summary_csv;
    config.rank = opts.rank;
    config.cv_mean_val_mse = cvInfo.mean_val_mse;
    config.cv_mean_val_r2 = cvInfo.mean_val_r2;
    config.raw_floor_uV2 = rawFloor;
    config.split_seed = opts.split_seed;
    config.loading_normalization = 'frobenius';
    config.heatmap_file = heatmapFile;
    config.structured_file = structuredFile;
    fid = fopen(configJson, 'w');
    if fid < 0
        error('run_roi10_matproj_log_contrast_rank1_ci_analysis:ConfigWriteFailed', ...
            'Could not write config JSON: %s', configJson);
    end
    fprintf(fid, '%s', jsonencode(config));
    fclose(fid);

    fid = fopen(summaryTxt, 'w');
    if fid < 0
        error('run_roi10_matproj_log_contrast_rank1_ci_analysis:SummaryWriteFailed', ...
            'Could not write summary text: %s', summaryTxt);
    end
    fprintf(fid, 'ROI10 matProj loading analysis for log10(EC) - log10(EO)\n');
    fprintf(fid, 'MAT file: %s\n', opts.mat_file);
    fprintf(fid, 'CV summary CSV: %s\n', opts.cv_summary_csv);
    fprintf(fid, 'Rank fixed at: %d\n', opts.rank);
    fprintf(fid, 'CV mean validation MSE: %.6f\n', cvInfo.mean_val_mse);
    fprintf(fid, 'CV mean validation R2: %.6f\n', cvInfo.mean_val_r2);
    fprintf(fid, 'Raw log floor uV2: %.12g\n', rawFloor);
    fprintf(fid, 'Split seed reused across all loadings: %d\n', opts.split_seed);
    fprintf(fid, 'All loadings normalized to unit Frobenius norm before estimation.\n');
    fprintf(fid, 'Results CSV: %s\n', resultsCsv);
    fprintf(fid, 'Summary CSV: %s\n', summaryCsv);
    fprintf(fid, 'Heatmap: %s\n', heatmapFile);
    fprintf(fid, 'Structured plot: %s\n', structuredFile);
    fclose(fid);

    fprintf('Saved results CSV: %s\n', resultsCsv);
    fprintf('Saved summary CSV: %s\n', summaryCsv);
end

function opts = parse_options(varargin)
    paths = resolve_release_paths();
    defaults = struct( ...
        'code_root', paths.code_root, ...
        'tensor_toolbox_root', '', ...
        'mat_file', paths.roi10_mat_file, ...
        'cv_summary_csv', fullfile(paths.matrix_cv_dir, 'cv5_rank_summary.csv'), ...
        'output_dir', paths.matrix_inference_dir, ...
        'rank', 1, ...
        'split_seed', 20260321, ...
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

function data = load_roi10_data(matFile)
    mat = load(matFile);
    data = struct();
    data.EO = double(mat.EO_roi_power_uV2);
    data.EC = double(mat.EC_roi_power_uV2);
    data.age = double(mat.age(:));
    data.roi_names = cellstr_from_mat(mat.roi_names);
    data.band_names = cellstr_from_mat(mat.band_names);
end

function [X, rawFloor] = build_log_contrast(rawEO, rawEC)
    positivesRaw = [rawEO(rawEO > 0); rawEC(rawEC > 0)];
    if isempty(positivesRaw)
        error('run_roi10_matproj_log_contrast_rank1_ci_analysis:NoPositiveRawValues', ...
            'Raw EO/EC arrays have no positive entries.');
    end
    rawFloor = max(min(positivesRaw) * 0.5, 1e-12);
    eoLog = log10(max(rawEO, rawFloor));
    ecLog = log10(max(rawEC, rawFloor));
    X = ecLog - eoLog;
end

function cvInfo = read_cv_summary(csvFile)
    cvSummary = readtable(csvFile, 'TextType', 'string');
    row = cvSummary(cvSummary.rank == 1, :);
    if isempty(row)
        error('run_roi10_matproj_log_contrast_rank1_ci_analysis:MissingCvRow', ...
            'Could not find rank-1 row in %s.', csvFile);
    end
    row = row(1, :);
    cvInfo = struct();
    cvInfo.mean_val_mse = double(row.mean_val_mse);
    cvInfo.mean_val_r2 = double(row.mean_val_r2);
end

function resultTable = evaluate_all_loadings(X, y, rank, roiNames, bandNames, splitSeed)
    y = y(:);
    N = numel(y);
    [p, q] = infer_matrix_shape(X, N);
    n1 = floor(N / 2);
    n2 = N - n1;
    zcrit = 1.959963984540054;

    loadings = generate_loadings(p, q, roiNames, bandNames);
    rows = repmat(struct( ...
        'loading_type', "", ...
        'loading_label', "", ...
        'roi_name', "", ...
        'band_name', "", ...
        'roi_index', NaN, ...
        'band_index', NaN, ...
        'fro_norm', NaN, ...
        'delta_hat', NaN, ...
        'sdest', NaN, ...
        'ci_low', NaN, ...
        'ci_high', NaN, ...
        'excludes_zero', false, ...
        'N', N, ...
        'fold1_n', n1, ...
        'fold2_n', n2), numel(loadings), 1);

    for idx = 1:numel(loadings)
        T = loadings(idx).matrix;
        froNorm = norm(T, 'fro');
        if froNorm <= 0
            error('run_roi10_matproj_log_contrast_rank1_ci_analysis:InvalidLoading', ...
                'Encountered a loading with zero Frobenius norm.');
        end
        T = T / froNorm;

        rng(splitSeed, 'twister');
        [deltaHat, sdest] = matProj(X, y, T, rank);

        ciHalf = zcrit * sdest / sqrt(N);
        ciLow = deltaHat - ciHalf;
        ciHigh = deltaHat + ciHalf;

        rows(idx).loading_type = string(loadings(idx).type);
        rows(idx).loading_label = string(loadings(idx).label);
        rows(idx).roi_name = string(loadings(idx).roi_name);
        rows(idx).band_name = string(loadings(idx).band_name);
        rows(idx).roi_index = loadings(idx).roi_index;
        rows(idx).band_index = loadings(idx).band_index;
        rows(idx).fro_norm = froNorm;
        rows(idx).delta_hat = deltaHat;
        rows(idx).sdest = sdest;
        rows(idx).ci_low = ciLow;
        rows(idx).ci_high = ciHigh;
        rows(idx).excludes_zero = (ciLow > 0) || (ciHigh < 0);
    end

    resultTable = struct2table(rows);
end

function [p, q] = infer_matrix_shape(X, nObs)
    if ndims(X) ~= 3
        error('run_roi10_matproj_log_contrast_rank1_ci_analysis:InvalidX', ...
            'Design array must be three-dimensional.');
    end

    sz = size(X);
    if sz(1) == nObs
        p = sz(2);
        q = sz(3);
        return;
    end
    if sz(3) == nObs
        p = sz(1);
        q = sz(2);
        return;
    end

    error('run_roi10_matproj_log_contrast_rank1_ci_analysis:InvalidXSize', ...
        'Could not match X dimensions to numel(y).');
end

function loadings = generate_loadings(p, q, roiNames, bandNames)
    nSingle = p * q;
    nRow = p;
    nCol = q;
    total = nSingle + nRow + nCol + 1;
    loadings(total) = struct('type', '', 'label', '', 'matrix', [], ...
        'roi_name', '', 'band_name', '', 'roi_index', NaN, 'band_index', NaN);

    idx = 0;
    for i = 1:p
        for j = 1:q
            idx = idx + 1;
            T = zeros(p, q);
            T(i, j) = 1;
            loadings(idx) = struct( ...
                'type', 'single_element', ...
                'label', sprintf('%s | %s', roiNames{i}, bandNames{j}), ...
                'matrix', T, ...
                'roi_name', roiNames{i}, ...
                'band_name', bandNames{j}, ...
                'roi_index', i, ...
                'band_index', j);
        end
    end

    for i = 1:p
        idx = idx + 1;
        T = zeros(p, q);
        T(i, :) = 1;
        loadings(idx) = struct( ...
            'type', 'row', ...
            'label', format_display_label(roiNames{i}), ...
            'matrix', T, ...
            'roi_name', roiNames{i}, ...
            'band_name', '', ...
            'roi_index', i, ...
            'band_index', NaN);
    end

    for j = 1:q
        idx = idx + 1;
        T = zeros(p, q);
        T(:, j) = 1;
        loadings(idx) = struct( ...
            'type', 'column', ...
            'label', format_display_label(bandNames{j}), ...
            'matrix', T, ...
            'roi_name', '', ...
            'band_name', bandNames{j}, ...
            'roi_index', NaN, ...
            'band_index', j);
    end

    idx = idx + 1;
    loadings(idx) = struct( ...
        'type', 'full_matrix', ...
        'label', 'global', ...
        'matrix', ones(p, q), ...
        'roi_name', '', ...
        'band_name', '', ...
        'roi_index', NaN, ...
        'band_index', NaN);
end

function plot_single_element_heatmap(resultTable, roiNames, bandNames, figTitle, outputFile)
    single = resultTable(resultTable.loading_type == "single_element", :);
    p = numel(roiNames);
    q = numel(bandNames);
    deltaMat = nan(p, q);
    sigMat = false(p, q);

    for idx = 1:height(single)
        i = single.roi_index(idx);
        j = single.band_index(idx);
        deltaMat(i, j) = single.delta_hat(idx);
        sigMat(i, j) = single.excludes_zero(idx);
    end

    fig = figure('Visible', 'off', 'Position', [100, 100, 1400, 650]);
    tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    imagesc(ax1, deltaMat);
    axis(ax1, 'tight');
    colorbar(ax1);
    title(ax1, sprintf('%s: delta', figTitle), 'Interpreter', 'none');
    xticks(ax1, 1:q);
    xticklabels(ax1, bandNames);
    yticks(ax1, 1:p);
    yticklabels(ax1, roiNames);
    xtickangle(ax1, 45);
    xlabel(ax1, 'Band');
    ylabel(ax1, 'ROI');
    hold(ax1, 'on');
    [sigRows, sigCols] = find(sigMat);
    scatter(ax1, sigCols, sigRows, 60, 'k', 'filled', 'MarkerFaceAlpha', 0.75);
    hold(ax1, 'off');

    ax2 = nexttile;
    imagesc(ax2, double(sigMat));
    axis(ax2, 'tight');
    colormap(ax2, [1 1 1; 0.10 0.55 0.20]);
    c = colorbar(ax2);
    c.Ticks = [0, 1];
    c.TickLabels = {'CI includes 0', 'CI excludes 0'};
    title(ax2, sprintf('%s: CI excludes 0', figTitle), 'Interpreter', 'none');
    xticks(ax2, 1:q);
    xticklabels(ax2, bandNames);
    yticks(ax2, 1:p);
    yticklabels(ax2, roiNames);
    xtickangle(ax2, 45);
    xlabel(ax2, 'Band');
    ylabel(ax2, 'ROI');

    exportgraphics(fig, outputFile, 'Resolution', 200);
    close(fig);
end

function plot_structured_loadings(resultTable, figTitle, outputFile)
    %#ok<INUSD>
    rows = resultTable(resultTable.loading_type == "row", :);
    cols = resultTable(resultTable.loading_type == "column", :);
    fullRow = resultTable(resultTable.loading_type == "full_matrix", :);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1500, 500]);
    tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    plot_errorbars(ax1, rows.loading_label, rows.delta_hat, rows.ci_low, rows.ci_high, rows.excludes_zero);
    title(ax1, 'ROI-slice loadings', 'Interpreter', 'none');

    ax2 = nexttile;
    plot_errorbars(ax2, cols.loading_label, cols.delta_hat, cols.ci_low, cols.ci_high, cols.excludes_zero);
    title(ax2, 'band-slice loadings', 'Interpreter', 'none');

    ax3 = nexttile;
    plot_errorbars(ax3, fullRow.loading_label, fullRow.delta_hat, fullRow.ci_low, fullRow.ci_high, fullRow.excludes_zero);
    title(ax3, 'the global loading', 'Interpreter', 'none');

    exportgraphics(fig, outputFile, 'Resolution', 200);
    close(fig);
end

function plot_errorbars(ax, labels, delta, ciLow, ciHigh, excludesZero)
    hold(ax, 'on');
    x = 1:numel(delta);
    for idx = 1:numel(delta)
        if excludesZero(idx)
            color = [0.85, 0.20, 0.20];
        else
            color = [0.35, 0.35, 0.35];
        end
        errorbar(ax, x(idx), delta(idx), delta(idx) - ciLow(idx), ciHigh(idx) - delta(idx), ...
            'o', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.2, 'CapSize', 8);
    end
    yline(ax, 0, '--k', 'LineWidth', 1.0);
    xlim(ax, [0.5, numel(delta) + 0.5]);
    xticks(ax, x);
    xticklabels(ax, labels);
    ax.TickLabelInterpreter = 'none';
    xtickangle(ax, 45);
    ylabel(ax, 'delta estimate with 95% CI');
    grid(ax, 'on');
    hold(ax, 'off');
end

function out = format_display_label(label)
    out = strrep(char(string(label)), '_', '-');
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
