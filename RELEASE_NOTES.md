# Release notes

## Included workflows

- Training and evaluation of the four SimVP stages, with hidden dimension 32, two temporal modules and pooled global validation RMSE for model selection.
- Training-year climatology estimation in month and UTC-slot bins.
- QD and MLT generation with ApexPy 2.1.1 at 450 km and an annual July-1 epoch. Coordinate order, grid identity and geometry settings are recorded with the generated arrays.
- Streaming evaluation of pooled errors, spatial errors, daily statistics and EIA detection and morphology.
- Portable data, model and output paths, command-line examples and an evaluation configuration example.

## Validation

The release checks used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, CPU PyTorch 2.15.0+cpu, tqdm 4.66.5 and ApexPy 2.1.1. ApexPy was compiled with GNU Fortran 13.3.0.

- All nine Python modules passed syntax compilation and import checks. All four executable scripts passed `--help`.
- A short synthetic-data evaluation preflight passed. Sample counts, target UTC alignment and training defaults were checked.
- Synthetic CPU forwards matched the original model implementation for all four stage configurations with identical state dictionaries. These checks used freshly initialized parameters and reduced input dimensions.
- Synthetic data loading, normalization, residual construction and month/UTC climatology matched the original calculations. Changing future observations alone preserved the historical input and scaled climatology background. Pooled RMSE, R² and the strict MAPE threshold matched hand-calculated values.
- ApexPy calculations on two 4×4 coordinate grids with 2015/2016 July-1 epochs passed. The affine MLT expression matched direct calculations at six UTCs with maximum circular difference `3.55e-15` hours. Coordinate order and hashes were verified, and geometry metadata passed through preflight.

The checks used synthetic inputs and covered module imports, command-line entry points, geometry calculations, data loading and evaluation formulas. Reported annual metrics were cross-checked against the manuscript's numerical summaries.
