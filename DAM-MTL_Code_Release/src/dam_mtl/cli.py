from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional

import torch

from .config import PROJECT_ROOT, load_config
from .data import fold_indices, load_monkey, verify_manifest
from .training import FoldResult, train_and_evaluate


OUTPUT_FIELDS = [
    "monkey",
    "protocol",
    "fold",
    "session",
    "seed",
    "cc_x",
    "cc_y",
    "cc_avg",
    "rmse_x",
    "rmse_y",
    "rmse_avg",
    "r2_x",
    "r2_y",
    "r2_avg",
    "mae_x",
    "mae_y",
    "mae_avg",
    "n_support",
    "n_validation",
    "n_query",
    "elapsed_seconds",
]


def parse_index_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not value.strip():
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def write_result(writer: csv.DictWriter, result: FoldResult) -> None:
    writer.writerow(result.as_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate DAM-MTL on the regression data")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "regression.yaml"))
    parser.add_argument("--monkey", choices=["I", "L"], required=True)
    parser.add_argument("--protocol", choices=["lodo", "forward"], required=True)
    parser.add_argument("--folds", help="Comma-separated fold indices; defaults to all valid folds")
    parser.add_argument("--seed-indices", help="Comma-separated indices into the configured seed list; defaults to all")
    parser.add_argument("--data-dir", help="Override the data directory from the configuration")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing result file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, args.data_dir)
    verify_manifest(config)
    sessions = load_monkey(config, args.monkey)

    available_folds = fold_indices(len(sessions), args.protocol, config.protocol.minimum_source_sessions)
    requested_folds = parse_index_list(args.folds)
    folds = available_folds if requested_folds is None else requested_folds
    invalid_folds = [fold for fold in folds if fold not in available_folds]
    if invalid_folds:
        raise ValueError("Invalid folds for this protocol: %s" % invalid_folds)

    requested_seed_indices = parse_index_list(args.seed_indices)
    seed_indices = list(range(len(config.seeds))) if requested_seed_indices is None else requested_seed_indices
    invalid_seed_indices = [index for index in seed_indices if index < 0 or index >= len(config.seeds)]
    if invalid_seed_indices:
        raise ValueError("Invalid seed indices: %s" % invalid_seed_indices)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("%s_%s.csv" % (args.monkey, args.protocol))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError("Result file exists; pass --overwrite to replace it: %s" % output_path)

    device = resolve_device(args.device)
    logging.info("monkey=%s protocol=%s folds=%s seeds=%s device=%s", args.monkey, args.protocol, folds, seed_indices, device)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for fold in folds:
            for seed_index in seed_indices:
                logging.info("starting fold=%d seed_index=%d", fold, seed_index)
                result = train_and_evaluate(sessions, config, args.protocol, fold, seed_index, device)
                write_result(writer, result)
                handle.flush()
                logging.info(
                    "finished fold=%d seed=%d cc_avg=%.4f rmse_avg=%.4f r2_avg=%.4f mae_avg=%.4f",
                    fold,
                    result.seed,
                    result.cc_avg,
                    result.rmse_avg,
                    result.r2_avg,
                    result.mae_avg,
                )
    logging.info("results written to %s", output_path)


if __name__ == "__main__":
    main()
