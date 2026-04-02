# External Dependencies

This directory can be used for third-party tools that are needed for local reruns but should not be vendored directly into the repository.

For MATLAB tensor analyses, one convenient option is to place Tensor Toolbox here, for example as:

- `external/tensor_toolbox`
- or `external/tensor_toolbox-v3.6`

Alternatively, set the `TENSOR_TOOLBOX_ROOT` environment variable before running the MATLAB scripts.
