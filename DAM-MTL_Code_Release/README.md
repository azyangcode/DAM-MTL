# DAM-MTL code

This repository contains the regression implementation used for cross-session decoder calibration in the DAM-MTL study. It includes the processed Monkey I and Monkey L sessions, the leave-one-day-out (LODO) and forward-chain protocols, the three-stage training procedure, and the final evaluation metrics.

 The  classification settings are retained in `configs/classification.yaml` and `docs/classification.md`

## Requirements

- Python 3.9 or later
- PyTorch 1.10 or later
- NumPy, SciPy, pandas, PyYAML

Install the package in editable mode:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data

The source regression recordings are the public non-human primate reaching dataset by O'Doherty et al. (Zenodo DOI: `10.5281/zenodo.583331`). Raw source files are not redistributed here. The manifest lists the 34 Monkey I sessions and 10 Monkey L sessions used in the study.

Download the source data from Zenodo and generate the processed sessions with:

```bash
python -m dam_mtl.preprocess --raw-dir /path/to/sabes_mat --output-dir data/regression
```

The preprocessing command preserves the original processing chain: MUA threshold-crossing trains are merged by channel, units below 0.5 spikes/s are removed, spike counts are computed in 240 ms windows with a 120 ms step, and the two velocity components are used as labels. Each processed pickle contains a two-item tuple:

```python
X, y
```

- `X`: float array of shape `(n_samples, n_channels)` containing spike-count features.
- `y`: float array of shape `(n_samples, 2)` containing horizontal and vertical hand velocity.

The session order is listed in `data/manifest.csv`. Fixed fold definitions are in `data/splits/lodo.csv` and `data/splits/forward.csv`. Inputs are zero-padded to 95 channels at load time.

## Quick start

Run one Monkey L LODO fold with one seed:

```bash
dam-mtl-regression --monkey L --protocol lodo --fold 0 --seed-index 0
```

Run one forward-chain fold:

```bash
dam-mtl-regression --monkey I --protocol forward --fold 10 --seed-index 0
```

By default, the command runs all valid folds and all ten seeds. New results are written to `runs/{monkey}_{protocol}.csv`.

```bash
dam-mtl-regression --monkey I --protocol lodo
dam-mtl-regression --monkey L --protocol lodo
dam-mtl-regression --monkey I --protocol forward
dam-mtl-regression --monkey L --protocol forward
```

The full protocol is computationally expensive because every target day is trained with ten seeds. For a first test, specify one fold and one seed.

## Evaluation protocols

### Leave-one-day-out

Each recording day is used once as the target day. All other days form the source set.

### Forward chaining

For target day `t`, only days `0, ..., t-1` are used as sources. At least five source days are required.

### Target-day split

For each target day:

- the first 120 s form the calibration segment;
- the final 20% of this segment is used for validation and early stopping;
- all subsequent samples form the query set;
- the query set is used only for final evaluation.

## Training configuration

The final regression model uses a time-conditioned session descriptor and channel-wise modulation in three convolution blocks.

| Stage | Configuration |
|---|---|
| Stage 1 | SGD, learning rate `1e-3`, momentum `0.9`, batch size `36`, `10` epochs |
| Stage 2 | `50` meta-iterations, `3` inner-loop steps, inner learning rate `0.01`, AdamW outer optimizer with learning rate `1e-4`, weight decay `1e-4`, anchor coefficient `0.05`, gradient clipping `2.0` |
| Stage 3 | SGD, learning rate `1e-2`, momentum `0.9`, batch size `32`, at most `100` epochs, early-stopping patience `10` |

The target-session offset is initialized by the model's default initializer and is updated with the other target-adaptation parameters using only the calibration segment.

Training seeds are `1000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000`. For fold `f` and seed index `j`, the random seed is `seed_base[j] + f`.

## Output metrics

Each result row contains:

- Pearson correlation coefficient (`cc_x`, `cc_y`, `cc_avg`)
- velocity RMSE (`rmse_x`, `rmse_y`, `rmse_avg`)
- coefficient of determination (`r2_x`, `r2_y`, `r2_avg`)
- mean absolute error (`mae_x`, `mae_y`, `mae_avg`)
- support and query sample counts
- elapsed time

## Package layout

```text
configs/                 Regression and classification settings
data/regression/         Produced after preprocessing the public source data
data/splits/             Fixed LODO and forward-chain fold definitions
docs/classification.md   Classification configuration retained for reference
results/                 Final per-fold results and summary
src/dam_mtl/             Data loading, model, training, metrics, runner
tests/                   Unit and smoke tests
```

The files in `results` contain the ten-seed fold-level outputs and the per-monkey summaries for the reported regression correlations.

## Checks

Run the included tests:

```bash
pytest -q
```

## Citation

If this code is used, please cite the accompanying DAM-MTL manuscript and the original regression dataset referenced above.

## License

The source code is released under the MIT License. The data remain subject to the terms of the original dataset.
