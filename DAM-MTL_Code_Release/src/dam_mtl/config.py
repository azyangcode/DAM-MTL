from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    root: Path
    manifest: Path
    channels: int = 95
    window_ms: int = 240
    step_ms: int = 120


@dataclass(frozen=True)
class ProtocolConfig:
    calibration_seconds: float = 120.0
    validation_fraction: float = 0.2
    minimum_source_sessions: int = 5
    source_meta_fraction: float = 0.7


@dataclass(frozen=True)
class Stage1Config:
    epochs: int = 10
    batch_size: int = 36
    learning_rate: float = 1e-3
    momentum: float = 0.9


@dataclass(frozen=True)
class Stage2Config:
    iterations: int = 50
    inner_steps: int = 3
    inner_learning_rate: float = 0.01
    outer_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    anchor_coefficient: float = 0.05
    gradient_clip_norm: float = 2.0
    batch_size: int = 36


@dataclass(frozen=True)
class Stage3Config:
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-2
    momentum: float = 0.9


@dataclass(frozen=True)
class EvaluationConfig:
    query_batch_size: int = 512


@dataclass(frozen=True)
class RegressionConfig:
    data: DataConfig
    protocol: ProtocolConfig
    stage1: Stage1Config
    stage2: Stage2Config
    stage3: Stage3Config
    evaluation: EvaluationConfig
    seeds: List[int]


def _resolve(path_value: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError("Config section '%s' must be a mapping" % name)
    return value


def load_config(
    path: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
) -> RegressionConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    data_raw = _section(raw, "data")
    protocol_raw = _section(raw, "protocol")
    training_raw = _section(raw, "training")
    evaluation_raw = _section(raw, "evaluation")
    stage1_raw = _section(training_raw, "stage1")
    stage2_raw = _section(training_raw, "stage2")
    stage3_raw = _section(training_raw, "stage3")

    root = Path(data_dir).resolve() if data_dir is not None else _resolve(str(data_raw["root"]))
    manifest = _resolve(str(data_raw["manifest"]))
    seeds = [int(value) for value in training_raw["seeds"]]
    if len(seeds) != 10:
        raise ValueError("The regression configuration requires ten seed bases")

    return RegressionConfig(
        data=DataConfig(
            root=root,
            manifest=manifest,
            channels=int(data_raw["channels"]),
            window_ms=int(data_raw["window_ms"]),
            step_ms=int(data_raw["step_ms"]),
        ),
        protocol=ProtocolConfig(
            calibration_seconds=float(protocol_raw["calibration_seconds"]),
            validation_fraction=float(protocol_raw["validation_fraction"]),
            minimum_source_sessions=int(protocol_raw["minimum_source_sessions"]),
            source_meta_fraction=float(protocol_raw["source_meta_fraction"]),
        ),
        stage1=Stage1Config(**stage1_raw),
        stage2=Stage2Config(**stage2_raw),
        stage3=Stage3Config(**stage3_raw),
        evaluation=EvaluationConfig(**evaluation_raw),
        seeds=seeds,
    )
