# Dataset Engineering

This folder contains reusable scripts for creating and validating datasets.

## inspectors

Read dataset metadata, class names, annotations, counts, and quality information without modifying source data.

## importers

Import or convert data from external sources, ZIP archives, detection datasets, videos, or image folders.

## builders

Create final train, validation, and test dataset structures.

## cleaners

Clean filenames, filter images, remove unsuitable data, sort samples, and prepare curated inputs.

## inventory

Contains dataset reports and registries. Raw dataset ZIP files should not be stored here.

## Important rule

Large datasets may remain outside the repository. Set `UAV_DATASETS_ROOT` for the
shared layout or one of the dataset-specific variables documented in
`00_PROJECT_GUIDE/PORTABLE_PATHS.md`. Without an override, scripts use the repository's
`datasets/` layout.
