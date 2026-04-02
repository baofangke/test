# Data Access

The real-data application uses resting-state EEG from the Healthy Brain Network (HBN; releases R1--R4).

This repository does **not** redistribute:

- raw HBN EEG recordings,
- participant-level metadata,
- subject-level processed power arrays stored in large `.mat` files.

## What is included here

Only lightweight analysis outputs are included, such as:

- cross-validation summaries,
- inference summaries,
- manuscript figures,
- algorithm notes describing how the results were produced.

## What a user must obtain separately

To reproduce the full workflow from raw data, the user must separately obtain access to the HBN EEG data and then regenerate the processed derivatives locally.

## Recommended wording for a GitHub release

If this repository is made public, the README should clearly state that:

1. the repository contains code and lightweight derived summaries only,
2. HBN data are subject to their own access and governance rules,
3. full reruns require local regeneration of derivatives from authorized source data.
