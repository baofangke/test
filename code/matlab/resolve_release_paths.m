function paths = resolve_release_paths()
%RESOLVE_RELEASE_PATHS Resolve repository-relative paths for the public release.

    thisFile = mfilename('fullpath');
    matlabDir = fileparts(thisFile);
    codeDir = fileparts(matlabDir);
    repoRoot = fileparts(codeDir);

    paths = struct();
    paths.repo_root = repoRoot;
    paths.code_root = matlabDir;
    paths.data_root = fullfile(repoRoot, 'data');
    paths.results_root = fullfile(repoRoot, 'results');
    paths.external_root = fullfile(repoRoot, 'external');

    paths.roi10_mat_file = fullfile(paths.data_root, 'hbn_bandpower_8band_roi10.mat');

    paths.matrix_results_root = fullfile(paths.results_root, 'matrix');
    paths.matrix_cv_dir = fullfile(paths.matrix_results_root, 'rank_selection_log_contrast_cv5');
    paths.matrix_inference_dir = fullfile(paths.matrix_results_root, 'inference_log_contrast_rank1');

    paths.tensor_results_root = fullfile(paths.results_root, 'tensor');
    paths.tensor_cv_dir = fullfile(paths.tensor_results_root, 'rank_selection');
    paths.tensor_inference_rank252_dir = fullfile(paths.tensor_results_root, 'inference_rank252_foldcentered');

    paths.tensor_toolbox_candidates = {
        getenv('TENSOR_TOOLBOX_ROOT')
        fullfile(paths.external_root, 'tensor_toolbox')
        fullfile(paths.external_root, 'tensor_toolbox-v3.6')
    };
end
