"""Evaluate an alternative baseline and the proposed model-selection framework.

Inputs: Saved datasets, fixed data partitions, trained surrogate models,
existing experimental predictions and documented framework predictions.

Outputs: Alternative baseline comparisons, physical-plausibility results
and framework evaluation tables.

This script conducts two supplementary investigations.

First, it trains a baseline model using log-transformed displacement targets
and compares its accuracy against the original baseline and physics-integrated
models. This assesses whether the observed benefits of physics integration
depend on the baseline target formulation.

Second, it evaluates the proposed model-selection framework by comparing
model performance under reduced training data, unseen geometries and
known or hidden parameter uncertainty.

Framework predictions must be documented before the evaluation is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis.analysis_utils as c
from analysis.analyse_performance_and_cost import blocks

LOG = c.mc.NAMES["baseline_log"]
KINDS = ("baseline", "physics_guided", "physics_estimate", "residual", "physics_regularised")


def evaluate_log_baseline(results, seeds):
    original, split, _ = c.mc.fixed_data()
    subsets = c.training_subsets(original, split)
    for pct, idx in subsets.items():
        expected = json.loads((c.mc.folder("baseline", seeds[0], pct) / "complete.json").read_text())["n_train"]
        if len(idx) != expected:
            raise ValueError(f"Training subset size for {pct}% differs from the saved baseline runs")
        for s in seeds:
            c.mc.train(original, split, "baseline_log", s, idx, pct)
    frames = {}
    for test in range(1, 6):
        data = c.mc.test_data(test, original, split)
        rows = []
        for pct in (c.mc.PERCENTAGES if test == 4 else (100,)):
            for s in seeds:
                m = c.Loaded("baseline_log", s, pct)
                if test == 3:
                    for level in (2.5, 5., 10.):
                        cases = data.loc[data.uncertainty_level_percent == level].sort_values("nominal_case_id")
                        truth = cases[c.mc.OUTPUTS].to_numpy(float)
                        for case in ("3A_known", "3B_hidden"):
                            inputs = cases
                            if case == "3B_hidden":
                                inputs = cases[[f"nominal_{col}" for col in c.mc.INPUTS]].copy()
                                inputs.columns = c.mc.INPUTS
                            rows.append(c.mc.prediction_rows(cases, inputs, truth, m.predict(inputs), test, f"{case}_{level:g}pct", "baseline_log", s, pct))
                else:
                    truth = data[c.mc.OUTPUTS].to_numpy(float)
                    rows.append(c.mc.prediction_rows(data, data, truth, m.predict(data), test, "all", "baseline_log", s, pct))
        frames[test] = pd.concat(rows, ignore_index=True)
    return frames


def compare(results, log_frames, n_boot):
    """Peak RMSE for every model, M1-log vs M1, and every physics model vs M1-log."""
    rows = []
    for test in range(1, 6):
        pred = pd.concat([c.load_predictions(results, test), log_frames[test]], ignore_index=True)
        seeds = sorted(pred.seed.unique())
        for group, pct, part in blocks(pred, test):
            sq = {m: c.error_matrix(part[part.model == m], c.PEAK, seeds)[0] for m in c.MODEL_ORDER + [LOG]}
            row = {"test": test, "group": group, "percentage": pct}
            for m, e in sq.items():
                row[f"{c.SHORT[m]}_peak_rmse_mm"] = c.mm(np.sqrt(e.mean(axis=1)).mean())
            pairs = [(LOG, c.BASELINE)] + [(m, LOG) for m in c.MODEL_ORDER[1:]]
            for model, ref in pairs:
                _, imp = c.paired_bootstrap(sq[model], sq[ref], n_boot=n_boot)
                lo, hi = np.percentile(imp, [2.5, 97.5])
                mean_imp, _, better, n, _, _ = c.seed_tests(np.sqrt(sq[model].mean(axis=1)), np.sqrt(sq[ref].mean(axis=1)))
                tag = f"{c.SHORT[model]}_vs_{c.SHORT[ref]}"
                row.update({f"{tag}_pct": mean_imp, f"{tag}_ci": f"[{lo:+.1f}, {hi:+.1f}]", f"{tag}_seeds": f"{better}/{n}", f"{tag}_verdict": c.verdict(lo, hi)})
            rows.append(row)
    return pd.DataFrame(rows)


def plausibility(results, log_frames):
    pred = pd.concat([c.load_predictions(results, 1), log_frames[1]], ignore_index=True)
    rows = []
    for (model, seed), b in pred.groupby(["model", "seed"]):
        bad = c.implausible(b)
        rows.append({"model": model, "seed": seed, "violation_pct": 100 * bad.mean(),
                     "negative_peak_pct": 100 * (b[f"pred_{c.PEAK}"] < 0).mean(),
                     "rms_above_peak_pct": 100 * (b[f"pred_{c.RMS}"] > b[f"pred_{c.PEAK}"]).mean()})
    per_seed = pd.DataFrame(rows)
    out = per_seed.groupby("model", as_index=False).agg(violation_pct_mean=("violation_pct", "mean"), violation_pct_sd=("violation_pct", "std"),
                                                        negative_peak_pct_mean=("negative_peak_pct", "mean"), rms_above_peak_pct_mean=("rms_above_peak_pct", "mean"), n_seeds=("seed", "nunique"))
    out["order"] = out.model.map({m: i for i, m in enumerate(c.MODEL_ORDER + [LOG])})
    return out.sort_values("order").drop(columns="order")


def framework_test(results, seeds, n_boot):
    original, split, _ = c.mc.fixed_data()
    t3 = c.mc.test_data(3, original, split)
    t3 = t3[t3.uncertainty_level_percent == 10.].sort_values("nominal_case_id")
    t2 = c.mc.test_data(2, original, split)
    hidden = t3[[f"nominal_{col}" for col in c.mc.INPUTS]].copy()
    hidden.columns = c.mc.INPUTS
    conditions = {"Test 3B hidden ±10%": (t3, hidden), "Test 3A known ±10%": (t3, t3), "Test 2 unseen geometry": (t2, t2)}
    frames = []
    for pct in (100, 10):
        for kind in KINDS:
            for s in seeds:
                m = c.Loaded(kind, s, pct)
                for cond, (cases, inputs) in conditions.items():
                    truth = cases[c.mc.OUTPUTS].to_numpy(float)
                    frames.append(c.mc.prediction_rows(cases, inputs, truth, m.predict(inputs), 0, cond, kind, s, pct))
    pred = pd.concat(frames, ignore_index=True)
    # Standard test set at 10% / 100% data, from the saved Test 4 predictions.
    t4 = c.load_predictions(results, 4)
    t4 = t4[t4.percentage.isin([10, 100])].assign(condition="Standard test set (Test 4)")
    pred = pd.concat([pred, t4], ignore_index=True)
    rows = []
    for (cond, pct), part in pred.groupby(["condition", "percentage"]):
        sq = {m: c.error_matrix(part[part.model == m], c.PEAK, seeds)[0] for m in c.MODEL_ORDER}
        for model in c.MODEL_ORDER:
            rm = np.sqrt(sq[model].mean(axis=1))
            row = {"condition": cond, "percentage": pct, "model": c.SHORT[model], "peak_rmse_mm_mean": c.mm(rm.mean()), "peak_rmse_mm_sd": c.mm(rm.std(ddof=1))}
            if model != c.BASELINE:
                _, imp = c.paired_bootstrap(sq[model], sq[c.BASELINE], n_boot=n_boot)
                lo, hi = np.percentile(imp, [2.5, 97.5])
                mean_imp, _, better, n, _, _ = c.seed_tests(rm, np.sqrt(sq[c.BASELINE].mean(axis=1)))
                row.update({"improvement_pct_mean": mean_imp, "ci_low": lo, "ci_high": hi, "seeds_better": f"{better}/{n}", "verdict": c.verdict(lo, hi)})
            rows.append(row)
    return pd.DataFrame(rows)


def check_predictions(ft):
    def get(cond, pct, model):
        return ft[(ft.condition == cond) & (ft.percentage == pct) & (ft.model == model)].iloc[0]
    std10 = get("Standard test set (Test 4)", 10, "M2").improvement_pct_mean
    hid10 = get("Test 3B hidden ±10%", 10, "M2").improvement_pct_mean
    p1 = hid10 < std10 / 2
    hid = ft[(ft.condition == "Test 3B hidden ±10%") & (ft.percentage == 10) & (ft.model != "M1")]
    p2 = not (hid.ci_low > 5).any()
    worst = 0.0
    for model in ("M1", "M2", "M3", "M4B", "M5"):
        a, b = get("Test 3B hidden ±10%", 10, model).peak_rmse_mm_mean, get("Test 3B hidden ±10%", 100, model).peak_rmse_mm_mean
        worst = max(worst, abs(a / b - 1))
    p3 = worst <= .20
    return [
        f"P1 {'HELD' if p1 else 'DID NOT HOLD'}: M2 improvement at 10% data is {std10:+.1f}% on the standard test set and {hid10:+.1f}% under hidden ±10% uncertainty.",
        f"P2 {'HELD' if p2 else 'DID NOT HOLD'}: largest lower 95% bound among physics models under hidden ±10% at 10% data is {hid.ci_low.max():+.1f}%.",
        f"P3 {'HELD' if p3 else 'DID NOT HOLD'}: largest change in Test 3B error between 100% and 10% data across models is {100 * worst:.1f}%.",
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="comparison/results/10seedresults")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    results = Path(a.results).resolve()
    out = results / "analysis" / "eval_baseline_and_framework"
    out.mkdir(parents=True, exist_ok=True)
    seeds = c.seeds_in(results)

    prediction_file = Path(__file__).resolve().parent / "framework_prediction.txt"
    if not prediction_file.is_file():
        raise SystemExit("Write analysis/framework_prediction.txt before running the framework test.")

    log_frames = evaluate_log_baseline(results, seeds)
    pd.concat(log_frames.values(), ignore_index=True).to_csv(out / "5a_log_baseline_predictions.csv.gz", index=False)
    comp = compare(results, log_frames, a.n_boot)
    comp.to_csv(out / "5a_log_baseline_comparison.csv", index=False)
    plaus = plausibility(results, log_frames)
    plaus.to_csv(out / "5a_physical_plausibility_test1.csv", index=False)

    ft = framework_test(results, seeds, a.n_boot)
    ft.to_csv(out / "5b_framework_test.csv", index=False)
    checks = check_predictions(ft)

    L = [f"Seeds: {len(seeds)}", "", "A. Fairness check: peak RMSE (mm) and M1-log vs M1"]
    show = comp[comp.group.isin(["all", "3B_hidden_10pct", "3A_known_10pct", "5A_near_resonance", "5B_strong_nonlinearity"])]
    for _, r in show.iterrows():
        tag = f"Test {r.test} {r.group}" + (f" {r.percentage}%" if r.test == 4 else "")
        L.append(f"  {tag:<32} M1 {r.M1_peak_rmse_mm:.3f}  M1-log {r['M1-log_peak_rmse_mm']:.3f}  M2 {r.M2_peak_rmse_mm:.3f}  M3 {r.M3_peak_rmse_mm:.3f}  M4B {r.M4B_peak_rmse_mm:.3f}  M5 {r.M5_peak_rmse_mm:.3f}"
                 f"  | M1-log vs M1 {r['M1-log_vs_M1_pct']:+.1f}% {r['M1-log_vs_M1_ci']} {r['M1-log_vs_M1_verdict']}")
    L.append("\n  Physics models vs M1-log (peak, mean improvement and verdict):")
    for _, r in show.iterrows():
        tag = f"Test {r.test} {r.group}" + (f" {r.percentage}%" if r.test == 4 else "")
        L.append(f"  {tag:<32} " + "  ".join(f"{m} {r[f'{m}_vs_M1-log_pct']:+.1f}% ({r[f'{m}_vs_M1-log_verdict']})" for m in ("M2", "M3", "M4B", "M5")))
    L.append("\n  Physically implausible predictions on Test 1 (% of cases, mean over seeds):")
    for _, r in plaus.iterrows():
        L.append(f"  {c.SHORT[r.model]:<7} {r.violation_pct_mean:.2f}%  (negative peak {r.negative_peak_pct_mean:.2f}%, RMS above peak {r.rms_above_peak_pct_mean:.2f}%)")
    L.append("\nB. Framework test (peak RMSE improvement vs M1)")
    for (cond, pct), g in ft.groupby(["condition", "percentage"], sort=False):
        L.append(f"  {cond}, {pct}% data: M1 {g[g.model == 'M1'].peak_rmse_mm_mean.iloc[0]:.3f} mm; " +
                 "  ".join(f"{r.model} {r.improvement_pct_mean:+.1f}% [{r.ci_low:+.1f}, {r.ci_high:+.1f}]" for _, r in g[g.model != "M1"].iterrows()))
    written = next((x for x in prediction_file.read_text().splitlines() if x.startswith("Written:")), "Written: unknown")
    L.append(f"\nPredictions in analysis/framework_prediction.txt ({written}):")
    L += ["  " + x for x in checks]
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nStep 5 outputs: {out}")


if __name__ == "__main__":
    main()
