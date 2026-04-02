# MATLAB Dependencies

The analysis scripts were originally run with a local MATLAB installation and, for some components, additional toolbox support.

## Core requirements

- MATLAB
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox

## Tensor toolbox

Tensor-regression scripts rely on a Tensor Toolbox installation. In the original workstation, this was supplied through a locally available Tensor Toolbox directory added to the MATLAB path.

## Local helper files

The MATLAB functions included in [`code/matlab`](../code/matlab) are the curated subset used by the reported real-data analyses. They include:

- `matProj.m`, `tsrProj.m`
- covariance/projection helpers such as `P_sigma.m`, `phi_sigma.m`, `compute_W_star.m`
- matrix/tensor fitting helpers

## Practical advice

Before a clean rerun on a different machine:

1. confirm MATLAB version compatibility,
2. install Tensor Toolbox,
3. verify that all hard-coded paths are replaced by project-relative paths,
4. test the MATLAB path setup before launching long CV or inference jobs.
