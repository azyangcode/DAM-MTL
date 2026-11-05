from pathlib import Path

import yaml

from dam_mtl.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_regression_config():
    config = load_config(ROOT / "configs" / "regression.yaml")
    assert config.data.channels == 95
    assert config.protocol.calibration_seconds == 120
    assert config.stage1.epochs == 10
    assert config.stage2.iterations == 50
    assert config.stage3.patience == 10
    assert len(config.seeds) == 10


def test_classification_config_matches_shared_backbone():
    with (ROOT / "configs" / "classification.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["model"]["backbone"] == "shared_with_regression"
    assert config["model"]["output_dim"] == 3
    assert config["model"]["session_descriptor"]["dimension"] == 64
    assert config["training"]["stage1"]["epochs"] == 10
