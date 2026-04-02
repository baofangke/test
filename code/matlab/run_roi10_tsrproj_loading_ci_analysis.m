function results = run_roi10_tsrproj_loading_ci_analysis(varargin)
%RUN_ROI10_TSRPROJ_LOADING_CI_ANALYSIS Evaluate loadings with tsrProj.
%
%   RESULTS = RUN_ROI10_TSRPROJ_LOADING_CI_ANALYSIS() loads the ROI10
%   10x8x2 log10(raw EC/EO) tensor covariates and computes delta estimates
%   plus 95% CIs for a set of Frobenius-normalized tensor loadings using
%   TSRPROJ. In the public-release package, the default rank is fixed at
%   [2, 5, 2] to match the manuscript results, but this can be overridden.

    opts = parse_options(varargin{:});

    if opts.setup_paths
        paths = setup_hbn_matlab_path(opts.code_root, opts.tensor_toolbox_root, true);
        opts.tensor_toolbox_root = paths.tensor_toolbox_root;
    end

    data = load_roi10_tensor_data(opts.mat_file);
    bestRow = read_rank_row(opts.cv_summary_csv, opts.rank_override);
    bestRank = [bestRow.rank1, bestRow.rank2, bestRow.rank3];
    loadings = generate_loadings(data.roi_names, data.band_names, data.condition_names);
    nAllLoadings = numel(loadings);
    loadingStart = max(1, floor(double(opts.loading_start)));
    loadingEnd = min(nAllLoadings, floor(double(opts.loading_end)));
    if loadingEnd < loadingStart
        error('run_roi10_tsrproj_loading_ci_analysis:InvalidLoadingRange', ...
            'loading_end must be >= loading_start.');
    end
    loadings = loadings(loadingStart:loadingEnd);
    if isfinite(opts.max_loadings)
        keep = min(numel(loadings), max(1, floor(opts.max_loadings)));
        loadings = loadings(1:keep);
    end

    outputDir = opts.output_dir;
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    zcrit = 1.959963984540054;
    nLoadings = numel(loadings);
    N = numel(data.age);
    rows = repmat(struct( ...
        'loading_type', "", ...
        'loading_label', "", ...
        'roi_name', "", ...
        'band_name', "", ...
        'condition_name', "", ...
        'roi_index', NaN, ...
        'band_index', NaN, ...
        'condition_index', NaN, ...
        'fro_norm', NaN, ...
        'delta_hat', NaN, ...
        'sdest', NaN, ...
        'ci_low', NaN, ...
        'ci_high', NaN, ...
        'excludes_zero', false, ...
        'N', N), nLoadings, 1);

    warnState1 = warning('off', 'MATLAB:nearlySingularMatrix');
    warnState2 = warning('off', 'MATLAB:singularMatrix');
    warnState3 = warning('off', 'MATLAB:rankDeficientMatrix');
    warningCleanup = onCleanup(@() restore_warning_states(warnState1, warnState2, warnState3)); %#ok<NASGU>

    for idx = 1:nLoadings
        T = loadings(idx).tensor;
        froNorm = norm(T(:));
        if froNorm <= 0
            error('run_roi10_tsrproj_loading_ci_analysis:InvalidLoading', ...
                'Encountered a loading with zero Frobenius norm.');
        end
        T = T / froNorm;

        fprintf('Loading %d/%d (global %d/%d): %s\n', ...
            idx, nLoadings, loadingStart + idx - 1, nAllLoadings, loadings(idx).label);
        rng(opts.split_seed, 'twister');
        [deltaHat, sdest] = tsrProj(data.X, data.age, T, bestRank);

        ciHalf = zcrit * sdest / sqrt(N);
        ciLow = deltaHat - ciHalf;
        ciHigh = deltaHat + ciHalf;

        rows(idx).loading_type = string(loadings(idx).type);
        rows(idx).loading_label = string(loadings(idx).label);
        rows(idx).roi_name = string(loadings(idx).roi_name);
        rows(idx).band_name = string(loadings(idx).band_name);
        rows(idx).condition_name = string(loadings(idx).condition_name);
        rows(idx).roi_index = loadings(idx).roi_index;
        rows(idx).band_index = loadings(idx).band_index;
        rows(idx).condition_index = loadings(idx).condition_index;
        rows(idx).fro_norm = froNorm;
        rows(idx).delta_hat = deltaHat;
        rows(idx).sdest = sdest;
        rows(idx).ci_low = ciLow;
        rows(idx).ci_high = ciHigh;
        rows(idx).excludes_zero = (ciLow > 0) || (ciHigh < 0);
    end

    results = struct2table(rows);
    nRows = height(results);
    results.rank1 = repmat(bestRank(1), nRows, 1);
    results.rank2 = repmat(bestRank(2), nRows, 1);
    results.rank3 = repmat(bestRank(3), nRows, 1);
    results.cv_mean_val_mse = repmat(bestRow.mean_val_mse, nRows, 1);
    results.cv_mean_val_r2 = repmat(bestRow.mean_val_r2, nRows, 1);
    results.loading_normalization = repmat("frobenius", nRows, 1);
    results = sortrows(results, ...
        {'loading_type', 'condition_index', 'roi_index', 'band_index', 'loading_label'});

    resultsCsv = fullfile(outputDir, 'tsrproj_loading_ci_all_results.csv');
    writetable(results, resultsCsv);

    summaryTable = build_summary_table(results, bestRow);
    summaryCsv = fullfile(outputDir, 'tsrproj_loading_ci_summary.csv');
    writetable(summaryTable, summaryCsv);

    singlePlot = "";
    structuredPlot = "";
    sharedContrastPlot = "";
    if opts.make_plots
        singlePlot = fullfile(outputDir, 'tsrproj_single_element_heatmaps.png');
        structuredPlot = fullfile(outputDir, 'tsrproj_structured_loadings.png');
        sharedContrastPlot = fullfile(outputDir, 'tsrproj_shared_contrast_loadings.png');
        plot_single_element_heatmaps(results, data.roi_names, data.band_names, data.condition_names, singlePlot);
        plot_structured_loadings(results, structuredPlot);
        plot_shared_contrast_loadings(results, sharedContrastPlot);
    end

    summaryTxt = fullfile(outputDir, 'tsrproj_loading_ci_summary.txt');
    write_summary_text(summaryTxt, opts, bestRow, results, resultsCsv, summaryCsv, char(singlePlot), char(structuredPlot), char(sharedContrastPlot));

    config = struct();
    config.mat_file = opts.mat_file;
    config.cv_summary_csv = opts.cv_summary_csv;
    config.output_dir = outputDir;
    config.code_root = opts.code_root;
    config.tensor_toolbox_root = opts.tensor_toolbox_root;
    config.split_seed = opts.split_seed;
    config.best_rank = bestRank;
    config.loading_normalization = 'frobenius';
    config.loading_count = nLoadings;
    config.loading_start = loadingStart;
    config.loading_end = loadingEnd;
    config.total_available_loadings = nAllLoadings;
    config.major_roi_groups = major_roi_groups(data.roi_names);
    config.plot_files = {singlePlot, structuredPlot, sharedContrastPlot};
    configJson = fullfile(outputDir, 'tsrproj_loading_ci_config.json');
    fid = fopen(configJson, 'w');
    if fid < 0
        error('run_roi10_tsrproj_loading_ci_analysis:ConfigWriteFailed', ...
            'Could not write config JSON: %s', configJson);
    end
    fprintf(fid, '%s', jsonencode(config));
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
        'cv_summary_csv', fullfile(paths.tensor_cv_dir, 'tucker_tensor_regression_cv_summary.csv'), ...
        'output_dir', paths.tensor_inference_rank252_dir, ...
        'split_seed', 20260322, ...
        'rank_override', [2 5 2], ...
        'loading_start', 1, ...
        'loading_end', inf, ...
        'max_loadings', inf, ...
        'make_plots', true, ...
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
        error('run_roi10_tsrproj_loading_ci_analysis:NoPositiveRawValues', ...
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

function bestRow = read_rank_row(summaryCsv, rankOverride)
    tbl = readtable(summaryCsv, 'TextType', 'string');
    tbl = sortrows(tbl, {'mean_val_mse', 'rank_sum', 'rank1', 'rank2', 'rank3'});
    if isempty(rankOverride)
        bestRow = tbl(1, :);
        return;
    end

    rankOverride = reshape(double(rankOverride), 1, []);
    if numel(rankOverride) ~= 3
        error('run_roi10_tsrproj_loading_ci_analysis:InvalidRankOverride', ...
            'rank_override must be a length-3 rank vector.');
    end

    mask = tbl.rank1 == rankOverride(1) & ...
        tbl.rank2 == rankOverride(2) & ...
        tbl.rank3 == rankOverride(3);
    hit = tbl(mask, :);
    if height(hit) ~= 1
        error('run_roi10_tsrproj_loading_ci_analysis:MissingRankOverride', ...
            'Could not find rank [%d %d %d] in CV summary.', ...
            rankOverride(1), rankOverride(2), rankOverride(3));
    end
    bestRow = hit(1, :);
end

function loadings = generate_loadings(roiNames, bandNames, conditionNames)
    p1 = numel(roiNames);
    p2 = numel(bandNames);
    p3 = numel(conditionNames);
    groups = major_roi_groups(roiNames);
    nGroups = numel(groups);
    nSingle = p1 * p2 * p3;
    total = nSingle + p1 + p2 + p3 + 1 + 1 + p2 + nGroups + 1 + p2 + nGroups;

    loadings(total) = struct( ...
        'type', '', ...
        'label', '', ...
        'tensor', [], ...
        'roi_name', '', ...
        'band_name', '', ...
        'condition_name', '', ...
        'roi_index', NaN, ...
        'band_index', NaN, ...
        'condition_index', NaN);

    idx = 0;
    for i = 1:p1
        for j = 1:p2
            for k = 1:p3
                idx = idx + 1;
                T = zeros(p1, p2, p3);
                T(i, j, k) = 1;
                loadings(idx) = struct( ...
                    'type', 'single_element', ...
                    'label', sprintf('%s | %s | %s', roiNames{i}, bandNames{j}, conditionNames{k}), ...
                    'tensor', T, ...
                    'roi_name', roiNames{i}, ...
                    'band_name', bandNames{j}, ...
                    'condition_name', conditionNames{k}, ...
                    'roi_index', i, ...
                    'band_index', j, ...
                    'condition_index', k);
            end
        end
    end

    for i = 1:p1
        idx = idx + 1;
        T = zeros(p1, p2, p3);
        T(i, :, :) = 1;
        loadings(idx) = struct( ...
            'type', 'roi_slice', ...
            'label', sprintf('ROI slice: %s', roiNames{i}), ...
            'tensor', T, ...
            'roi_name', roiNames{i}, ...
            'band_name', '', ...
            'condition_name', '', ...
            'roi_index', i, ...
            'band_index', NaN, ...
            'condition_index', NaN);
    end

    for j = 1:p2
        idx = idx + 1;
        T = zeros(p1, p2, p3);
        T(:, j, :) = 1;
        loadings(idx) = struct( ...
            'type', 'band_slice', ...
            'label', sprintf('Band slice: %s', bandNames{j}), ...
            'tensor', T, ...
            'roi_name', '', ...
            'band_name', bandNames{j}, ...
            'condition_name', '', ...
            'roi_index', NaN, ...
            'band_index', j, ...
            'condition_index', NaN);
    end

    for k = 1:p3
        idx = idx + 1;
        T = zeros(p1, p2, p3);
        T(:, :, k) = 1;
        loadings(idx) = struct( ...
            'type', 'condition_slice', ...
            'label', sprintf('Condition slice: %s', conditionNames{k}), ...
            'tensor', T, ...
            'roi_name', '', ...
            'band_name', '', ...
            'condition_name', conditionNames{k}, ...
            'roi_index', NaN, ...
            'band_index', NaN, ...
            'condition_index', k);
    end

    idx = idx + 1;
    loadings(idx) = struct( ...
        'type', 'full_tensor', ...
        'label', 'all_elements', ...
        'tensor', ones(p1, p2, p3), ...
        'roi_name', '', ...
        'band_name', '', ...
        'condition_name', '', ...
        'roi_index', NaN, ...
        'band_index', NaN, ...
        'condition_index', NaN);

    sharedVec = [1, 1] / sqrt(2);
    contrastVec = [1, -1] / sqrt(2);

    idx = idx + 1;
    loadings(idx) = struct( ...
        'type', 'shared_global', ...
        'label', 'shared: global', ...
        'tensor', build_mode3_loading(ones(p1, p2), sharedVec), ...
        'roi_name', '', ...
        'band_name', '', ...
        'condition_name', 'shared', ...
        'roi_index', NaN, ...
        'band_index', NaN, ...
        'condition_index', NaN);

    for j = 1:p2
        idx = idx + 1;
        spatial = zeros(p1, p2);
        spatial(:, j) = 1;
        loadings(idx) = struct( ...
            'type', 'shared_band', ...
            'label', sprintf('shared: %s', bandNames{j}), ...
            'tensor', build_mode3_loading(spatial, sharedVec), ...
            'roi_name', '', ...
            'band_name', bandNames{j}, ...
            'condition_name', 'shared', ...
            'roi_index', NaN, ...
            'band_index', j, ...
            'condition_index', NaN);
    end

    for g = 1:nGroups
        idx = idx + 1;
        spatial = zeros(p1, p2);
        spatial(groups(g).indices, :) = 1;
        loadings(idx) = struct( ...
            'type', 'shared_roi_group', ...
            'label', sprintf('shared: %s', groups(g).name), ...
            'tensor', build_mode3_loading(spatial, sharedVec), ...
            'roi_name', groups(g).name, ...
            'band_name', '', ...
            'condition_name', 'shared', ...
            'roi_index', NaN, ...
            'band_index', NaN, ...
            'condition_index', NaN);
    end

    idx = idx + 1;
    loadings(idx) = struct( ...
        'type', 'contrast_global', ...
        'label', 'contrast: global', ...
        'tensor', build_mode3_loading(ones(p1, p2), contrastVec), ...
        'roi_name', '', ...
        'band_name', '', ...
        'condition_name', 'contrast', ...
        'roi_index', NaN, ...
        'band_index', NaN, ...
        'condition_index', NaN);

    for j = 1:p2
        idx = idx + 1;
        spatial = zeros(p1, p2);
        spatial(:, j) = 1;
        loadings(idx) = struct( ...
            'type', 'contrast_band', ...
            'label', sprintf('contrast: %s', bandNames{j}), ...
            'tensor', build_mode3_loading(spatial, contrastVec), ...
            'roi_name', '', ...
            'band_name', bandNames{j}, ...
            'condition_name', 'contrast', ...
            'roi_index', NaN, ...
            'band_index', j, ...
            'condition_index', NaN);
    end

    for g = 1:nGroups
        idx = idx + 1;
        spatial = zeros(p1, p2);
        spatial(groups(g).indices, :) = 1;
        loadings(idx) = struct( ...
            'type', 'contrast_roi_group', ...
            'label', sprintf('contrast: %s', groups(g).name), ...
            'tensor', build_mode3_loading(spatial, contrastVec), ...
            'roi_name', groups(g).name, ...
            'band_name', '', ...
            'condition_name', 'contrast', ...
            'roi_index', NaN, ...
            'band_index', NaN, ...
            'condition_index', NaN);
    end
end

function summaryTable = build_summary_table(results, bestRow)
    summaryTable = table( ...
        [bestRow.rank1, bestRow.rank2, bestRow.rank3], ...
        bestRow.mean_val_mse, ...
        bestRow.mean_val_r2, ...
        sum(results.loading_type == "single_element" & results.excludes_zero), ...
        sum(results.loading_type == "roi_slice" & results.excludes_zero), ...
        sum(results.loading_type == "band_slice" & results.excludes_zero), ...
        sum(results.loading_type == "condition_slice" & results.excludes_zero), ...
        sum(results.loading_type == "full_tensor" & results.excludes_zero), ...
        sum(results.loading_type == "shared_global" & results.excludes_zero), ...
        sum(results.loading_type == "shared_band" & results.excludes_zero), ...
        sum(results.loading_type == "shared_roi_group" & results.excludes_zero), ...
        sum(results.loading_type == "contrast_global" & results.excludes_zero), ...
        sum(results.loading_type == "contrast_band" & results.excludes_zero), ...
        sum(results.loading_type == "contrast_roi_group" & results.excludes_zero), ...
        'VariableNames', { ...
            'selected_rank', 'cv_mean_val_mse', 'cv_mean_val_r2', ...
            'single_element_excludes_zero', 'roi_slice_excludes_zero', ...
            'band_slice_excludes_zero', 'condition_slice_excludes_zero', ...
            'full_tensor_excludes_zero', ...
            'shared_global_excludes_zero', 'shared_band_excludes_zero', ...
            'shared_roi_group_excludes_zero', 'contrast_global_excludes_zero', ...
            'contrast_band_excludes_zero', 'contrast_roi_group_excludes_zero'});
end

function plot_single_element_heatmaps(results, roiNames, bandNames, conditionNames, outputFile)
    single = results(results.loading_type == "single_element", :);
    p1 = numel(roiNames);
    p2 = numel(bandNames);
    p3 = numel(conditionNames);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1500, 900]);
    tiledlayout(fig, p3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for k = 1:p3
        deltaMat = nan(p1, p2);
        sigMat = false(p1, p2);
        sub = single(single.condition_index == k, :);
        for idx = 1:height(sub)
            i = sub.roi_index(idx);
            j = sub.band_index(idx);
            deltaMat(i, j) = sub.delta_hat(idx);
            sigMat(i, j) = sub.excludes_zero(idx);
        end

        ax1 = nexttile;
        imagesc(ax1, deltaMat);
        axis(ax1, 'tight');
        colorbar(ax1);
        title(ax1, sprintf('%s: delta', conditionNames{k}), 'Interpreter', 'none');
        xticks(ax1, 1:p2);
        xticklabels(ax1, bandNames);
        yticks(ax1, 1:p1);
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
        title(ax2, sprintf('%s: CI excludes 0', conditionNames{k}), 'Interpreter', 'none');
        xticks(ax2, 1:p2);
        xticklabels(ax2, bandNames);
        yticks(ax2, 1:p1);
        yticklabels(ax2, roiNames);
        xtickangle(ax2, 45);
        xlabel(ax2, 'Band');
        ylabel(ax2, 'ROI');
    end

    exportgraphics(fig, outputFile, 'Resolution', 200);
    close(fig);
end

function plot_structured_loadings(results, outputFile)
    roiRows = results(results.loading_type == "roi_slice", :);
    bandRows = results(results.loading_type == "band_slice", :);
    condRows = results(results.loading_type == "condition_slice", :);
    fullRow = results(results.loading_type == "full_tensor", :);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1700, 500]);
    tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    plot_errorbars(ax1, roiRows.loading_label, roiRows.delta_hat, roiRows.ci_low, roiRows.ci_high, roiRows.excludes_zero);
    title(ax1, 'ROI slices', 'Interpreter', 'none');

    ax2 = nexttile;
    plot_errorbars(ax2, bandRows.loading_label, bandRows.delta_hat, bandRows.ci_low, bandRows.ci_high, bandRows.excludes_zero);
    title(ax2, 'Band slices', 'Interpreter', 'none');

    ax3 = nexttile;
    plot_errorbars(ax3, condRows.loading_label, condRows.delta_hat, condRows.ci_low, condRows.ci_high, condRows.excludes_zero);
    title(ax3, 'Condition slices', 'Interpreter', 'none');

    ax4 = nexttile;
    plot_errorbars(ax4, fullRow.loading_label, fullRow.delta_hat, fullRow.ci_low, fullRow.ci_high, fullRow.excludes_zero);
    title(ax4, 'Full tensor', 'Interpreter', 'none');

    exportgraphics(fig, outputFile, 'Resolution', 200);
    close(fig);
end

function plot_shared_contrast_loadings(results, outputFile)
    sharedGlobal = results(results.loading_type == "shared_global", :);
    sharedBand = results(results.loading_type == "shared_band", :);
    sharedGroup = results(results.loading_type == "shared_roi_group", :);
    contrastGlobal = results(results.loading_type == "contrast_global", :);
    contrastBand = results(results.loading_type == "contrast_band", :);
    contrastGroup = results(results.loading_type == "contrast_roi_group", :);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 800]);
    tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    plot_errorbars(ax1, sharedGlobal.loading_label, sharedGlobal.delta_hat, sharedGlobal.ci_low, sharedGlobal.ci_high, sharedGlobal.excludes_zero);
    title(ax1, 'Shared Global', 'Interpreter', 'none');

    ax2 = nexttile;
    plot_errorbars(ax2, sharedBand.loading_label, sharedBand.delta_hat, sharedBand.ci_low, sharedBand.ci_high, sharedBand.excludes_zero);
    title(ax2, 'Shared By Band', 'Interpreter', 'none');

    ax3 = nexttile;
    plot_errorbars(ax3, sharedGroup.loading_label, sharedGroup.delta_hat, sharedGroup.ci_low, sharedGroup.ci_high, sharedGroup.excludes_zero);
    title(ax3, 'Shared By ROI Group', 'Interpreter', 'none');

    ax4 = nexttile;
    plot_errorbars(ax4, contrastGlobal.loading_label, contrastGlobal.delta_hat, contrastGlobal.ci_low, contrastGlobal.ci_high, contrastGlobal.excludes_zero);
    title(ax4, 'Contrast Global', 'Interpreter', 'none');

    ax5 = nexttile;
    plot_errorbars(ax5, contrastBand.loading_label, contrastBand.delta_hat, contrastBand.ci_low, contrastBand.ci_high, contrastBand.excludes_zero);
    title(ax5, 'Contrast By Band', 'Interpreter', 'none');

    ax6 = nexttile;
    plot_errorbars(ax6, contrastGroup.loading_label, contrastGroup.delta_hat, contrastGroup.ci_low, contrastGroup.ci_high, contrastGroup.excludes_zero);
    title(ax6, 'Contrast By ROI Group', 'Interpreter', 'none');

    exportgraphics(fig, outputFile, 'Resolution', 200);
    close(fig);
end

function plot_errorbars(ax, labels, delta, ciLow, ciHigh, excludesZero)
    if isempty(delta)
        axis(ax, 'off');
        text(ax, 0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        return;
    end

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
    xtickangle(ax, 45);
    ylabel(ax, 'delta estimate with 95% CI');
    grid(ax, 'on');
    hold(ax, 'off');
end

function restore_warning_states(varargin)
    for idx = 1:nargin
        warning(varargin{idx});
    end
end

function T = build_mode3_loading(spatial, mode3vec)
    T = spatial .* reshape(mode3vec, 1, 1, []);
end

function groups = major_roi_groups(~)
    groups = struct( ...
        'name', {'anterior', 'central', 'posterior'}, ...
        'indices', {[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]});
end

function write_summary_text(outputFile, opts, bestRow, results, resultsCsv, summaryCsv, singlePlot, structuredPlot, sharedContrastPlot)
    fid = fopen(outputFile, 'w');
    if fid < 0
        error('run_roi10_tsrproj_loading_ci_analysis:SummaryWriteFailed', ...
            'Could not write summary text: %s', outputFile);
    end

    fprintf(fid, 'ROI10 tsrProj loading analysis\n');
    fprintf(fid, 'Tensor shape: 10 x 8 x 2 (ROI x Band x Condition)\n');
    fprintf(fid, 'Transform: log10(raw EC) and log10(raw EO)\n');
    fprintf(fid, 'Selected Tucker rank: [%d %d %d]\n', bestRow.rank1, bestRow.rank2, bestRow.rank3);
    fprintf(fid, 'CV mean MSE: %.6f\n', bestRow.mean_val_mse);
    fprintf(fid, 'CV mean R2: %.6f\n', bestRow.mean_val_r2);
    fprintf(fid, 'Split seed reused across all loadings: %d\n', opts.split_seed);
    fprintf(fid, 'All loadings normalized to unit Frobenius norm before estimation.\n');
    fprintf(fid, 'Single elements excluding zero: %d\n', sum(results.loading_type == "single_element" & results.excludes_zero));
    fprintf(fid, 'ROI slices excluding zero: %d\n', sum(results.loading_type == "roi_slice" & results.excludes_zero));
    fprintf(fid, 'Band slices excluding zero: %d\n', sum(results.loading_type == "band_slice" & results.excludes_zero));
    fprintf(fid, 'Condition slices excluding zero: %d\n', sum(results.loading_type == "condition_slice" & results.excludes_zero));
    fprintf(fid, 'Full tensor excluding zero: %d\n', sum(results.loading_type == "full_tensor" & results.excludes_zero));
    fprintf(fid, 'Shared global excluding zero: %d\n', sum(results.loading_type == "shared_global" & results.excludes_zero));
    fprintf(fid, 'Shared band excluding zero: %d\n', sum(results.loading_type == "shared_band" & results.excludes_zero));
    fprintf(fid, 'Shared ROI-group excluding zero: %d\n', sum(results.loading_type == "shared_roi_group" & results.excludes_zero));
    fprintf(fid, 'Contrast global excluding zero: %d\n', sum(results.loading_type == "contrast_global" & results.excludes_zero));
    fprintf(fid, 'Contrast band excluding zero: %d\n', sum(results.loading_type == "contrast_band" & results.excludes_zero));
    fprintf(fid, 'Contrast ROI-group excluding zero: %d\n', sum(results.loading_type == "contrast_roi_group" & results.excludes_zero));
    fprintf(fid, 'Major ROI groups used: anterior={1,2,3}, central={4,5,6,7}, posterior={8,9,10}\n');
    fprintf(fid, 'Results CSV: %s\n', resultsCsv);
    fprintf(fid, 'Summary CSV: %s\n', summaryCsv);
    fprintf(fid, 'Single-element heatmaps: %s\n', singlePlot);
    fprintf(fid, 'Structured loading plot: %s\n', structuredPlot);
    fprintf(fid, 'Shared/contrast loading plot: %s\n', sharedContrastPlot);
    fclose(fid);
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
