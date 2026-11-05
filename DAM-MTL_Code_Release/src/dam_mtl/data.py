from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import RegressionConfig


@dataclass(frozen=True)
class SessionRecord:
    monkey: str
    day_index: int
    source_file: str
    output_file: str


@dataclass(frozen=True)
class SessionData:
    monkey: str
    day_index: int
    filename: str
    x: np.ndarray
    y: np.ndarray


def read_manifest(path: Path) -> List[SessionRecord]:
    records: List[SessionRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                SessionRecord(
                    monkey=row["monkey"],
                    day_index=int(row["day_index"]),
                    source_file=row["source_file"],
                    output_file=row["output_file"],
                )
            )
    return records


def verify_manifest(config: RegressionConfig) -> None:
    records = read_manifest(config.data.manifest)
    expected = {"I": 34, "L": 10}
    counts: Dict[str, int] = {}
    for record in records:
        path = config.data.root / record.output_file
        if not path.is_file():
            raise FileNotFoundError(
                "%s is missing; prepare the source data with python -m dam_mtl.preprocess" % path
            )
        counts[record.monkey] = counts.get(record.monkey, 0) + 1
    if counts != expected:
        raise ValueError("Unexpected session counts: %s" % counts)


def _read_pickle(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        x, y = payload["X"], payload["y"]
    else:
        x, y = payload
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.shape != (x.shape[0], 2):
        raise ValueError("Unexpected array shapes in %s: %s %s" % (path, x.shape, y.shape))
    return x, y


def zero_pad_channels(x: np.ndarray, channels: int) -> np.ndarray:
    if x.shape[1] > channels:
        raise ValueError("Session has %d channels, exceeding the configured %d" % (x.shape[1], channels))
    if x.shape[1] == channels:
        return np.ascontiguousarray(x, dtype=np.float32)
    padded = np.zeros((x.shape[0], channels), dtype=np.float32)
    padded[:, : x.shape[1]] = x
    return padded


def load_monkey(config: RegressionConfig, monkey: str) -> List[SessionData]:
    records = [record for record in read_manifest(config.data.manifest) if record.monkey == monkey]
    records.sort(key=lambda record: record.day_index)
    sessions: List[SessionData] = []
    for record in records:
        x, y = _read_pickle(config.data.root / record.output_file)
        sessions.append(
            SessionData(
                monkey=record.monkey,
                day_index=record.day_index,
                filename=record.output_file,
                x=zero_pad_channels(x, config.data.channels),
                y=np.ascontiguousarray(y, dtype=np.float32),
            )
        )
    return sessions


def fold_indices(n_sessions: int, protocol: str, minimum_sources: int = 5) -> List[int]:
    if protocol == "lodo":
        return list(range(n_sessions))
    if protocol == "forward":
        return list(range(minimum_sources, n_sessions))
    raise ValueError("Unknown protocol: %s" % protocol)


def source_indices(target: int, n_sessions: int, protocol: str) -> List[int]:
    if protocol == "lodo":
        return [index for index in range(n_sessions) if index != target]
    if protocol == "forward":
        return list(range(target))
    raise ValueError("Unknown protocol: %s" % protocol)


def support_sample_count(seconds: float, window_ms: int, step_ms: int) -> int:
    # Window starts are sampled every ``step_ms``; the first window starts at 0.
    count = int((seconds * 1000.0 - step_ms) / step_ms) + 1
    if count <= 0:
        raise ValueError("Calibration segment is shorter than one window")
    return count


def split_target(
    x: np.ndarray,
    y: np.ndarray,
    support_samples: int,
    validation_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if support_samples >= len(y):
        raise ValueError("Calibration segment does not leave a query interval")
    validation_samples = int(support_samples * validation_fraction)
    training_samples = support_samples - validation_samples
    return (
        x[:training_samples],
        y[:training_samples],
        x[training_samples:support_samples],
        y[training_samples:support_samples],
        x[support_samples:],
        y[support_samples:],
    )


def split_source(x: np.ndarray, y: np.ndarray, fraction: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(fraction * len(y))
    if split <= 0 or split >= len(y):
        raise ValueError("Invalid source-session split")
    return x[:split], y[:split], x[split:], y[split:]
