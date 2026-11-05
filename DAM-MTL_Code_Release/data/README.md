# Regression data source

The Monkey I and Monkey L recordings are from the public non-human primate reaching dataset by O'Doherty et al.:

- Title: Nonhuman primate reaching with multichannel sensorimotor cortex electrophysiology
- Zenodo DOI: `10.5281/zenodo.583331`
- DOI link: `https://doi.org/10.5281/zenodo.583331`

This repository does not redistribute the source `.mat` files or the processed monkey pickles. Download the source files from Zenodo and place them in a local directory. `manifest.csv` lists the required source filenames and output filenames in recording-day order.

Generate the processed regression sessions with:

```bash
python -m dam_mtl.preprocess --raw-dir /path/to/sabes_mat --output-dir data/regression
```

The preprocessing procedure is:

1. Read the MATLAB v7.3/HDF5 source fields `cursor_pos`, `target_pos`, `t`, and `spikes`.
2. Merge all threshold-crossing spike trains on each electrode channel into one MUA train.
3. Remove channels with a mean firing rate at or below 0.5 spikes/s.
4. Compute hand velocity and acceleration from cursor position.
5. Count spikes in 240 ms windows advanced by 120 ms.
6. Save each session as a pickle `(X, y)`, where `X` is spike count by channel and `y` contains the horizontal and vertical velocity components.

At training time, arrays are zero-padded to 95 channels. Fixed evaluation folds are provided in `splits/lodo.csv` and `splits/forward.csv`.
