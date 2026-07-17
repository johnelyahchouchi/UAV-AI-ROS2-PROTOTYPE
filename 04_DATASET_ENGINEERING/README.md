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

Large datasets remain in the external dataset master directory. This project folder stores the scripts, mappings, registries, and documentation required to reproduce them.
