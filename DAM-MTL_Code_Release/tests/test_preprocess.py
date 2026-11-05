import pickle

import h5py
import numpy as np

from dam_mtl.preprocess import process_session


def test_preprocess_session(tmp_path):
    raw_path = tmp_path / "raw.mat"
    output_path = tmp_path / "processed.pickle"
    task_time = np.arange(0.0, 1.0, 0.004)
    cursor_position = np.stack([np.sin(task_time), np.cos(task_time)], axis=1)
    with h5py.File(raw_path, "w") as handle:
        handle["cursor_pos"] = cursor_position.T
        handle["target_pos"] = cursor_position.T
        handle["t"] = task_time
        references = np.empty((2, 2), dtype=h5py.ref_dtype)
        for channel in range(2):
            for unit in range(2):
                train = np.arange(0.05 + channel * 0.01 + unit * 0.02, 0.95, 0.08)
                dataset = handle.create_dataset("spike_%d_%d" % (channel, unit), data=train)
                references[channel, unit] = dataset.ref
        handle["spikes"] = references

    process_session(raw_path, output_path, window_ms=240, step_ms=120, number_of_channels=2)
    with output_path.open("rb") as handle:
        x, y = pickle.load(handle)
    assert x.ndim == 2 and x.shape[1] == 2
    assert y.shape == (x.shape[0], 2)
    assert x.shape[0] > 0
