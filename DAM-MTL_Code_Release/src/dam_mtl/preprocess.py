from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np


def _flatten_spike_trains(trains: Iterable[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(train, dtype=np.float64).reshape(-1) for train in trains if len(train) > 0]
    if not nonempty:
        return np.empty(0, dtype=np.float64)
    merged = np.concatenate(nonempty)
    merged.sort()
    return merged


def _read_spike_train(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[()]
    if getattr(values, "ndim", 0) == 2:
        return np.asarray(values).squeeze(axis=0)
    return np.asarray(values).reshape(-1)


def _extract_mua_trains(handle: h5py.File, task_time: np.ndarray, number_of_channels: int = 96) -> List[np.ndarray]:
    spike_refs = handle["spikes"][()].T
    number_of_units = spike_refs.shape[1]
    mua_trains: List[np.ndarray] = []
    minimum_firing_rate = 0.5
    minimum_spike_count = int(np.round(minimum_firing_rate * (task_time[-1] - task_time[0])))
    for channel_index in range(number_of_channels):
        channel_trains = []
        for unit_index in range(number_of_units):
            reference = spike_refs[channel_index, unit_index]
            train = _read_spike_train(handle[reference])
            if train.size:
                inside = (train >= task_time[0]) & (train <= task_time[-1])
                train = train[inside]
                if train.size:
                    channel_trains.append(train)
        merged = _flatten_spike_trains(channel_trains)
        if len(merged) > minimum_spike_count:
            mua_trains.append(merged)
    return mua_trains


def _task_kinematics(task_position: np.ndarray, task_time: np.ndarray) -> np.ndarray:
    delta_time = float(np.diff(task_time).mean())
    velocity = np.diff(task_position, axis=0) / delta_time
    acceleration = np.diff(velocity, axis=0) / delta_time
    velocity = np.concatenate((velocity, velocity[-1:]), axis=0)
    acceleration = np.concatenate((acceleration, acceleration[-2:]), axis=0)
    return np.concatenate((task_position, velocity, acceleration), axis=1)


def _bin_mua(spike_trains: List[np.ndarray], task_time: np.ndarray, task_data: np.ndarray, window_ms: int, step_ms: int):
    sample_interval_ms = float(np.diff(task_time).mean() * 1000.0)
    window_samples = int(round(window_ms / sample_interval_ms)) + 1
    step_samples = int(round(step_ms / sample_interval_ms))
    starts = [start for start in range(0, max(0, len(task_time) - window_samples), step_samples) if start + window_samples <= len(task_time) - 1]
    targets = np.asarray([start + window_samples for start in starts], dtype=np.int64)
    output = []
    for train in spike_trains:
        rates = []
        for start in starts:
            end = start + window_samples
            count = int(np.sum((train >= task_time[start]) & (train < task_time[end])))
            rates.append(count)
        output.append(np.asarray(rates, dtype=np.float32))
    x = np.stack(output, axis=1)
    y = task_data[targets, 2:4].astype(np.float32)
    return x, y


def process_session(
    input_file: Path,
    output_file: Path,
    window_ms: int = 240,
    step_ms: int = 120,
    number_of_channels: int = 96,
) -> None:
    with h5py.File(input_file, "r") as handle:
        task_position = np.asarray(handle["cursor_pos"][()], dtype=np.float64).T
        task_time = np.asarray(handle["t"][()], dtype=np.float64).squeeze()
        if task_position.shape[0] != task_time.shape[0]:
            raise ValueError("cursor_pos and task time have different lengths in %s" % input_file)
        task_data = _task_kinematics(task_position, task_time)
        mua_trains = _extract_mua_trains(handle, task_time, number_of_channels)
    x, y = _bin_mua(mua_trains, task_time, task_data, window_ms, step_ms)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as handle:
        pickle.dump((x, y), handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Sabes regression sessions used by DAM-MTL")
    parser.add_argument("--raw-dir", required=True, help="Directory containing the source MATLAB v7.3/HDF5 files")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed pickle files")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--window-ms", type=int, default=240)
    parser.add_argument("--step-ms", type=int, default=120)
    parser.add_argument("--channels", type=int, default=96, help="Number of electrode channels in the source file")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = Path(args.manifest).resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        input_file = raw_dir / row["source_file"]
        output_file = output_dir / row["output_file"]
        if not input_file.is_file():
            raise FileNotFoundError(input_file)
        process_session(input_file, output_file, args.window_ms, args.step_ms, args.channels)
        print("%s -> %s" % (input_file.name, output_file.name))


if __name__ == "__main__":
    main()
