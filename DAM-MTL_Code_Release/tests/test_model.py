from pathlib import Path

import torch

from dam_mtl.data import read_manifest
from dam_mtl.model import DAMMTLRegressor


def test_model_output_shape():
    model = DAMMTLRegressor(num_domains=5, input_length=95)
    x = torch.randn(2, 1, 95)
    domain_ids = torch.tensor([0, 4], dtype=torch.long)
    output = model(x, domain_ids)
    assert output.shape == (2, 2)


def test_packaged_data_exists():
    root = Path(__file__).resolve().parents[1]
    records = read_manifest(root / "data" / "manifest.csv")
    assert len([record for record in records if record.monkey == "I"]) == 34
    assert len([record for record in records if record.monkey == "L"]) == 10
