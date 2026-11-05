# CG-SimVP research code

This repository provides the SimVP-style direct-field backbone, three successive extensions, and the streaming TEC evaluator. It supports the retrospective year split used in the manuscript.

The project repository hosts the implementation and release documentation. [CG-SimVP repository](https://github.com/PANXIONG-CN/CG-SimVP)

Corresponding authors are Pan Xiong (xiong.pan@gmail.com) and Xuhui Shen (shenxuhui@nssc.ac.cn).

| Split | Years |
| --- | --- |
| Training and climatology estimation | 2012, 2014, 2016, 2018, 2020 |
| Validation and checkpoint selection | 2013, 2017 |
| Test | 2015, 2019 |

The forecast uses 24 historical frames to predict 24 future frames at 15-minute cadence. JPLD supplies the reference TEC field. Geometry channels are deterministic functions of coordinates and UTC. The four stages share a SimVP-style architecture, with direct-field prediction as Stage I.

## Environment

Use Python 3.10 or newer. Run the commands below from this directory. Install a PyTorch build appropriate for the CPU or CUDA environment, then install the remaining requirements. ApexPy 2.1.1 may require a Fortran compiler when built from source.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency bounds describe the public package. The exact environment used for release checks is recorded in `RELEASE_NOTES.md`.

## Data and coordinates

Obtain the 2012–2020 JPLD daily NetCDF4 files from the official JPL research archive. Export TEC maps in TECU, spatial coordinates and UTC timestamps to the NPY layout below. Preserve the source variable definitions, units, axis order and chronology when preparing the NPY inputs. [Official JPLD archive](https://sideshow.jpl.nasa.gov/pub/iono_daily/gim_for_research/jpld/)

Place TEC chunks in `data/2012_npy/`, ..., `data/2020_npy/`. Each NPY file contains raw TECU with shape `[frames, 180, 360]`. Lexicographic filename order must be chronological within each year. Concatenated chunks must begin at January 1, 00:00 UTC and retain every 15-minute frame, including leap days. Keep missing-value processing consistent with the selected checkpoint. The evaluator requires finite maps and nonnegative reference TEC on a common support.

`data/global_max.txt` holds the training-year maximum used by the loader. Training creates this cache from the declared training years when it is absent. Evaluation requires the existing cache. The recorded manuscript value is `157.6999969482422`, with minimum zero. Retain the normalization associated with any existing checkpoint.

Export the actual TEC coordinate arrays to `data/latitude.npy` and `data/longitude.npy`. They must be one-dimensional and preserve the row and column order of the TEC arrays. `geo2qd` uses geodetic latitude in degrees north and longitude in degrees east. Record any upstream coordinate conversion in the source description.

Generate the grid and geometry cache from those arrays. Replace the source-description text with the actual source fields and preprocessing record.

```bash
python prepare_geometry.py \
  --latitude-file data/latitude.npy \
  --longitude-file data/longitude.npy \
  --grid-npz data/grid.npz \
  --grid-source "Source coordinate fields and preprocessing record for these TEC arrays" \
  --cache-root data/geometry \
  --years 2012 2013 2014 2015 2016 2017 2018 2019 2020
```

An existing `grid.npz` containing `lat` and `lon` can be supplied by omitting the two coordinate-file arguments. The generator records coordinate hashes and structural checks in `grid.provenance.json`. The source description records how these coordinates correspond to the TEC arrays.

The geometry defaults follow the documented settings. ApexPy 2.1.1 computes QD latitude/longitude at 450 km with a fixed July-1 epoch within each year. `mlon2mlt(0, UTC)` supplies an offset for every frame, using ApexPy's subsolar height of 318550 km. MLT is reconstructed as `(qd_lon / 15 + offset) % 24`. The generator cross-checks this expression against direct ApexPy calls and stores file hashes in each year's `metadata.json`.

`prepare_geometry.py` generates QD coordinates and time-dependent MLT offsets using the settings above. Existing caches can be used after the version, grid, cadence and hashes are validated. [ApexPy documentation](https://apexpy.readthedocs.io/en/stable/autoapi/generated/apexpy/apex/index.html)

## Climatology and training

```bash
python compute_climatology.py --data-root data --output data/climatology.npy
```

This averages training TEC within month and UTC-slot bins. The loader divides TEC and climatology by the same training maximum. For each sample, the historical-window ratio `alpha = (mean(history) + 1e-6) / (mean(history_climatology) + 1e-6)` is clipped to `[0.3, 3.0]`. The same alpha scales historical and future climatology. The model predicts the residual, and evaluation restores the background before conversion to TECU. Auxiliary channels use the historical window.

| Stage | Climatology residual | Attention | Auxiliary channels |
| --- | --- | --- | --- |
| I, direct-field backbone | Off | None | None |
| II | On | None | None |
| III | On | Per-frame CBAM | None |
| IV, CG-SimVP | On | Per-frame CBAM | cos(SZA), sin(QD latitude), sin(MLT phase), cos(MLT phase) |

```bash
python train_cbam_clim.py --model-suffix stage_i
python train_cbam_clim.py --model-suffix stage_ii --use-climatology 1
python train_cbam_clim.py --model-suffix stage_iii --use-climatology 1 --attn-name cbam
python train_cbam_clim.py --model-suffix cg_simvp --use-climatology 1 --attn-name cbam --use-aux-sza 1 --use-aux-qd 1
```

Defaults are hidden dimension 32, two temporal Inception modules, batch size 4, learning rate `6e-4`, up to 300 epochs, patience 50, and stride 24 for training and validation. Training uses AdamW with weight decay `1e-4`, SmoothL1Loss, gradient norm clipping at 5, and ReduceLROnPlateau with factor 0.5 and patience 10. The minimum learning rate is the initial rate multiplied by `1e-4`. The selected checkpoint minimizes validation global RMSE, computed from accumulated squared error and pixel count. Final predictions retain signed model outputs.

The random seed is 42. cuDNN benchmarking and TF32 are enabled, and CUDA training uses bfloat16 autocast.

Use `--data-root`, `--model-save-dir`, `--grid-npz` and `--geometry-cache` for alternate locations. Training automatically validates the selected geometry and saves its provenance in `train_metrics_SimVP_SUFFIX.json`. `--geometry-provenance` optionally cross-checks an existing geometry summary. Preserve the metrics JSON with its matching `best_SimVP_SUFFIX.pth`. Existing checkpoint paths are protected against overwrite. All command-line options are available through `--help`.

## Evaluation

Copy `evaluation_config.example.json` to `evaluation_config.json` and set its paths to the selected checkpoint and associated metadata. Paths in JSON are relative to the JSON file. The example uses explicit UTC timestamps, stored as a one-dimensional NPY array of `datetime64` values or UTC strings. They must match concatenated data frames. If the full-year chronology has been independently verified, remove `timestamps` and set `confirmed_full_year_regular_utc` to `true`.

```bash
python evaluate.py --config evaluation_config.json --preflight
python evaluate.py --config evaluation_config.json
```

For 2019, change `year`, `timestamps` and `output_dir` together. Choose a separate output directory for every checkpoint/year/configuration. Preflight validates data layout, chronology, normalization and geometry metadata before model loading. Its scope is recorded in `preflight.json`. Checkpoint loading uses strict state-dict matching.

`metrics.csv` contains per-lead scene statistics. `pooled_metrics.csv` combines squared-error sufficient statistics across all leads. `spatial.npz` stores per-lead RMSE and bias maps. `daily.csv` supplies day/lead SSE and weight for subsequent paired block resampling. `frame_metrics.csv`, EIA tables, requested case arrays and `manifest.json` retain timestamps, support and provenance. Set `export_predictions` to a new NPY path to save `[sample, lead, latitude, longitude]` predictions in TECU together with target UTC arrays.

RMSE is the square root of pooled SSE divided by weight. R² uses the same pooled targets and their global mean. MAPE uses reference TEC strictly above 1 TECU by default. Pixel and cosine-latitude weighting are reported separately. A lead of 360 minutes is the 6 h endpoint. A pooled row combines all 24 leads. Baseline predictors available through `predictor` are `persistence`, `climatology_scaled`, and `climatology_unscaled`. Scaled climatology requires a residual-loader configuration. Unscaled climatology is read in raw TECU.

The default EIA protocol samples every fourth UTC frame and every fourth longitude column. At 1° resolution the latter corresponds to 4° spacing. Under the default window, stride and cadence, the sampled EIA leads are 15, 75, 135, 195, 255 and 315 minutes. Profiles are restricted to LMST 12–18 h, interpolated to 1° QD latitude between −30° and 30°, and searched for interior crests in −25° to −5° and 5° to 25°. Both crest-to-trough differences must reach 1 TECU. Detection recall uses all reference twin-crest profiles in its denominator. Position, amplitude and separation errors use matched detections. Trough error uses all reference twin-crest profiles. Read conditional errors together with `truth_twin` and `matched_twins` counts.

Optional `activity_csv` requires `start_utc,end_utc,kp,dst` with half-open UTC intervals. Optional `events_csv` requires `start_utc,end_utc,event_id,phase`. Activity labels remain `unknown` when activity data are absent. `case_times_utc` accepts explicit timezone-bearing ISO timestamps.

## Manuscript citation

Yuqi Lin, Jing Liu, Pan Xiong, Jianghe Chen, Cheng Long, Huiyu Zhou, Angelo De Santis, and Xuhui Shen. "CG-SimVP for Global TEC Forecasting With Adaptive Climatological Residuals and Solar-Geomagnetic Geometry." Unpublished manuscript, 2026.
