from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["cc_avg", "rmse_avg", "r2_avg", "mae_avg"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fold-level results across training seeds")
    parser.add_argument("results", nargs="+", help="One or more result CSV files")
    parser.add_argument("--output", default="runs/summary.csv")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.results]
    data = pd.concat(frames, ignore_index=True)
    per_fold = data.groupby(["monkey", "protocol", "fold"], as_index=False)[METRICS].mean()
    rows = []
    for (monkey, protocol), group in per_fold.groupby(["monkey", "protocol"]):
        row = {"monkey": monkey, "protocol": protocol, "n_folds": len(group)}
        for metric in METRICS:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
