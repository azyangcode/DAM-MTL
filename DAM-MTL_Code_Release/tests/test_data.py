import numpy as np

from dam_mtl.data import split_target, support_sample_count, zero_pad_channels


def test_calibration_sample_count():
    assert support_sample_count(120.0, 240, 120) == 1000


def test_chronological_target_split():
    x = np.arange(1200, dtype=np.float32)[:, None]
    y = np.arange(1200, dtype=np.float32)[:, None]
    y = np.repeat(y, 2, axis=1)
    result = split_target(x, y, 1000, 0.2)
    assert len(result[0]) == 800
    assert len(result[2]) == 200
    assert len(result[4]) == 200
    assert result[4][0, 0] == 1000


def test_zero_padding():
    x = np.ones((4, 3), dtype=np.float32)
    padded = zero_pad_channels(x, 5)
    assert padded.shape == (4, 5)
    assert np.all(padded[:, :3] == 1)
    assert np.all(padded[:, 3:] == 0)
