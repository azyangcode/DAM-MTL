import numpy as np
import torch

from dam_mtl.config import DataConfig, EvaluationConfig, ProtocolConfig, RegressionConfig, Stage1Config, Stage2Config, Stage3Config
from dam_mtl.data import SessionData
from dam_mtl.training import train_and_evaluate


def test_training_and_evaluation_smoke(tmp_path):
    rng = np.random.RandomState(0)
    sessions = [
        SessionData(
            monkey="L",
            day_index=index,
            filename="day_%d.pickle" % index,
            x=rng.normal(size=(40, 95)).astype(np.float32),
            y=rng.normal(size=(40, 2)).astype(np.float32),
        )
        for index in range(6)
    ]
    config = RegressionConfig(
        data=DataConfig(root=tmp_path, manifest=tmp_path / "manifest.csv", channels=95, window_ms=240, step_ms=120),
        protocol=ProtocolConfig(calibration_seconds=2.4, validation_fraction=0.2, minimum_source_sessions=5, source_meta_fraction=0.7),
        stage1=Stage1Config(epochs=1, batch_size=36),
        stage2=Stage2Config(iterations=1, inner_steps=1, batch_size=36),
        stage3=Stage3Config(max_epochs=1, patience=1, batch_size=32),
        evaluation=EvaluationConfig(query_batch_size=32),
        seeds=[1000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000],
    )
    result = train_and_evaluate(sessions, config, "lodo", 5, 0, torch.device("cpu"))
    assert result.n_support == 16
    assert result.n_validation == 4
    assert result.n_query == 20
    assert np.isfinite(result.cc_avg)
