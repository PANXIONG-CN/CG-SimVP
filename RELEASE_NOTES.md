# Public-release preparation

## Changes

- Renamed `Dataloader_clim_aux.py` to `dataloader_clim_aux.py` and `SimVP.py` to `simvp.py`, with matching imports and manifest file names. Defaults use relative `data`, `models` and `results` locations.
- Aligned model and training defaults with the supplied run settings, including hidden dimension 32, two temporal modules and validation global RMSE for checkpoint selection. README commands specify the four reported stages.
- Added a climatology CLI and explicit grid/cache arguments. Training now derives and records geometry provenance from the selected assets. Evaluation validates the same information during preflight and configures the loader environment. Training metadata records normalization and an explicit global-RMSE history alongside the legacy batch-RMSE history.
- Added `prepare_geometry.py` from the documented ApexPy 2.1.1 / 450 km / annual July-1 protocol. It imports actual coordinate arrays, preserves their order, records provenance and file hashes, and checks the affine MLT expression against direct ApexPy calls. This is a public-release reconstruction. Historical cache identity requires comparison with original author assets.
- Added the English README, dependency list, evaluation example and repository ignore rules. The numerical architecture, alpha/residual formulas, loss and streaming metric formulas follow the supplied implementation.

## Actual validation

Checks ran with real installed dependencies on Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, CPU PyTorch 2.15.0+cpu, tqdm 4.66.5 and ApexPy 2.1.1. ApexPy was compiled with GNU Fortran 13.3.0. This is the release-check environment.

- Syntax compilation of all nine Python files and imports of all nine modules passed. All four executable scripts passed `--help`.
- A short synthetic-data `evaluate.py --preflight` run passed. Sample counts, target UTC alignment and published training defaults were checked.
- Small synthetic CPU forwards matched the original implementation exactly for all four stage configurations after loading identical state dictionaries. These checks used freshly initialized parameters and reduced synthetic input dimensions.
- Synthetic cross-file loading, normalization, residual construction and month/UTC climatology matched the original implementation. Changing future observations alone preserved the historical input and scaled climatology background. Pooled RMSE, R² and the strict MAPE threshold matched hand-calculated values.
- Real ApexPy calls on two 4×4 synthetic coordinate grids with 2015/2016 July-1 epochs passed. The affine MLT expression matched direct calls at six UTCs with maximum circular difference `3.55e-15` hours. Coordinate import preserved order and produced matching hashes. Geometry metadata passed through preflight, and a deliberately incorrect version was rejected.

Validation covered entry points, dependency APIs and local numerical consistency with synthetic inputs. Full training, original checkpoints, annual TEC inference and full annual geometry-cache generation were outside these checks. During manuscript preparation, the reported annual metrics were cross-checked against the supplied CSV sufficient statistics.

## Author release items

Before repository publication, choose a license and supply the final paper citation and a statement of weight availability. Shared historical weights should be paired with their preprocessing and configuration assets. The README identifies the official JPLD data source and the scope of packaged model support.
