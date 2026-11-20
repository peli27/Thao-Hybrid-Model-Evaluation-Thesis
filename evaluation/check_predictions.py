"""Check that predictions.csv.gz reproduces the RMSE values in seed_metrics.csv.

Usage: python evaluation/check_predictions.py --test 1 [--out FOLDER]
Prints the largest absolute RMSE difference; it should be below 1e-5 (the network outputs float32).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from multiseed_core import OUTPUTS, RESULTS, groups_for


def check(test, out=None):
    out = Path(out) if out is not None else RESULTS / f"test_{test}"
    pred = pd.read_csv(out / "predictions.csv.gz")
    stored = pd.read_csv(out / "seed_metrics.csv")
    rows = []
    for (model, seed, percentage, condition), cases in pred.groupby(["model", "seed", "percentage", "condition"]):
        groups = {condition: np.ones(len(cases), bool)} if test == 3 else groups_for(test, cases)
        for group, mask in groups.items():
            if not mask.any():
                continue
            for label in OUTPUTS:
                error = cases[f"pred_{label}"].to_numpy()[mask] - cases[f"true_{label}"].to_numpy()[mask]
                rows.append({"group": group, "model": model, "seed": seed, "percentage": percentage, "output": label, "rmse_check": float(np.sqrt(np.mean(error**2)))})
    merged = stored.merge(pd.DataFrame(rows), on=["group", "model", "seed", "percentage", "output"], how="outer", indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    worst = float(((merged.rmse - merged.rmse_check).abs() / merged.rmse.abs()).max())
    print(f"Test {test}: {len(pred)} prediction rows, {len(stored)} metric rows, unmatched {unmatched}, max relative RMSE difference {worst:.2e}")
    if unmatched or not worst < 1e-5:
        raise SystemExit("Mismatch between predictions.csv.gz and seed_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    check(args.test, args.out)
