# Paths and Portability

This release was assembled from a working analysis environment that used local drive-based paths such as `H:\\HBN-EEG\\...` and `C:\\Users\\...`.

## Important consequence

Several copied scripts still contain machine-specific absolute paths. They are suitable for inspection and provenance tracking, but they are **not yet fully portable** as a one-command public rerun package.

## Recommended cleanup before a public reproducibility release

- Replace absolute input/output paths with a project configuration file.
- Use relative paths anchored at the repository root.
- Separate raw-data paths from derived-output paths.
- Add a single top-level setup script that defines:
  - data root,
  - output root,
  - figure/table output locations,
  - toolbox locations.

## Why the current release is still useful

Even before path cleanup, the repository already provides:

- the exact manuscript source used for drafting,
- the main scripts that generated the reported real-data results,
- lightweight outputs and algorithm notes that allow the reported claims to be inspected.
