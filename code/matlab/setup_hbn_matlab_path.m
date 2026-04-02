function paths = setup_hbn_matlab_path(codeRoot, tensorToolboxRoot, requireTensorToolbox)
%SETUP_HBN_MATLAB_PATH Add repository MATLAB code and optional Tensor Toolbox.

    defaults = resolve_release_paths();

    if nargin < 1 || isempty(codeRoot)
        codeRoot = defaults.code_root;
    end
    if nargin < 3 || isempty(requireTensorToolbox)
        requireTensorToolbox = false;
    end

    if nargin < 2 || isempty(tensorToolboxRoot)
        tensorToolboxRoot = '';
        tensorCandidates = defaults.tensor_toolbox_candidates;
        for idx = 1:numel(tensorCandidates)
            candidate = tensorCandidates{idx};
            if ~isempty(candidate) && exist(candidate, 'dir')
                tensorToolboxRoot = candidate;
                break;
            end
        end
    end

    if ~exist(codeRoot, 'dir')
        error('setup_hbn_matlab_path:MissingCodeRoot', ...
            'Code root does not exist: %s', codeRoot);
    end

    addpath(codeRoot);
    rehash;

    if isempty(tensorToolboxRoot)
        if requireTensorToolbox
            error('setup_hbn_matlab_path:MissingTensorToolbox', ...
                ['Tensor Toolbox was not found. Set TENSOR_TOOLBOX_ROOT, ' ...
                 'place the toolbox under external/, or pass tensor_toolbox_root explicitly.']);
        end
    elseif exist(tensorToolboxRoot, 'dir')
        addpath(genpath(tensorToolboxRoot));
        fprintf('Added Tensor Toolbox path: %s\n', tensorToolboxRoot);
    else
        if requireTensorToolbox
            error('setup_hbn_matlab_path:MissingTensorToolbox', ...
                'Tensor Toolbox root does not exist: %s', tensorToolboxRoot);
        else
            warning('setup_hbn_matlab_path:SkippingTensorToolbox', ...
                'Tensor Toolbox root does not exist and will be skipped: %s', tensorToolboxRoot);
            tensorToolboxRoot = '';
        end
    end

    paths = struct( ...
        'code_root', codeRoot, ...
        'tensor_toolbox_root', tensorToolboxRoot);

    fprintf('Added code path: %s\n', codeRoot);
end
