"""Analyse the sensitivity of surrogate performance to nonlinear-response classification.

Inputs: Saved Test 5 predictions, nonlinear stiffness ratios and analytical
displacement estimates from the final surrogate evaluations.

Outputs: A threshold-sensitivity comparison table saved to
comparison/results/<results_folder>/analysis/step3/.

This script investigates whether the conclusions of Test 5B depend on the
threshold used to classify strongly nonlinear structural responses.

It evaluates nonlinear stiffness thresholds of 5%, 10%, 20% and 30%,
comparing the peak-displacement prediction accuracy of all five surrogate
models against the data-driven baseline.

The analysis also examines linear analytical estimate errors and calculates
paired bootstrap confidence intervals for improvements over the baseline.

Existing predictions are reused without retraining models or generating
additional structural simulations.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_utils as c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="comparison/results/10seedresults")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    results = Path(a.results).resolve()
    pred = c.load_predictions(results, 5)
    seeds = sorted(pred.seed.unique())
    one = pred[(pred.model == c.BASELINE) & (pred.seed == seeds[0])]
    rows = []
    for threshold in (0.05, 0.10, 0.20, 0.30):
        keep = set(one.row_index[one.nonlinear_stiffness_ratio_peak >= threshold])
        part = pred[pred.row_index.isin(keep)]
        est = one[one.row_index.isin(keep)]
        est_rmse = c.mm(np.sqrt(np.mean((est.linear_peak_estimate - est[f"true_{c.PEAK}"]) ** 2)))
        sq = {m: c.error_matrix(part[part.model == m], c.PEAK, seeds)[0] for m in c.MODEL_ORDER}
        for model in c.MODEL_ORDER:
            rm = np.sqrt(sq[model].mean(axis=1))
            row = {"threshold_pct": 100 * threshold, "n_cases": len(keep), "linear_estimate_peak_rmse_mm": est_rmse,
                   "model": c.SHORT[model], "peak_rmse_mm_mean": c.mm(rm.mean())}
            if model != c.BASELINE:
                _, imp = c.paired_bootstrap(sq[model], sq[c.BASELINE], n_boot=a.n_boot)
                lo, hi = np.percentile(imp, [2.5, 97.5])
                mean_imp, _, better, n, _, _ = c.seed_tests(rm, np.sqrt(sq[c.BASELINE].mean(axis=1)))
                row.update({"improvement_pct_mean": mean_imp, "ci_low": lo, "ci_high": hi, "seeds_better": f"{better}/{n}", "verdict": c.verdict(lo, hi)})
            rows.append(row)
    table = pd.DataFrame(rows)
    out = results / "analysis" / "threshold_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "3f_threshold_sensitivity.csv", index=False)
    for (t, n, e), g in table.groupby(["threshold_pct", "n_cases", "linear_estimate_peak_rmse_mm"]):
        m1 = g[g.model == "M1"].peak_rmse_mm_mean.iloc[0]
        parts = "  ".join(f"{r.model} {r.improvement_pct_mean:+.1f}% [{r.ci_low:+.1f}, {r.ci_high:+.1f}] {r.verdict}" for _, r in g[g.model != "M1"].iterrows())
        print(f"η ≥ {t:g}% (n={n}, linear-estimate RMSE {e:.2f} mm, M1 {m1:.3f} mm): {parts}")


if __name__ == "__main__":
    main()
