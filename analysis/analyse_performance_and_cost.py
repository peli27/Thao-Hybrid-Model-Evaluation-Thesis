"""Analyse surrogate accuracy, physical-model reliability and computational cost.

Inputs: Saved model predictions from Tests 1-5, numerical reference responses,
and existing simulation and training timing records.

Outputs: Supplementary tables, figures and a summary report under
<results>/analysis/performance_and_cost_analysis/.

This script provides additional evidence for interpreting the thesis results.
It evaluates displacement prediction errors relative to response magnitudes,
calculates paired bootstrap ranges for improvements over the baseline, and
examines the effects of hidden parameter uncertainty and nonlinear response
on prediction accuracy.

It also compares computational costs using recorded model training times,
measured inference latency and estimated simulation times.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis.analysis_utils as c  # noqa: E402

OUTS = (c.PEAK, c.RMS)


def blocks(pred, test):
    """Yield (group, percentage, frame restricted to that group) for every subgroup in a test."""
    for (pct, cond), part in pred.groupby(["percentage", "condition"]):
        one = part[(part.model == c.BASELINE) & (part.seed == part.seed.min())].sort_values("row_index")
        for group, mask in c.group_masks(test, one).items():
            keep = set(one.row_index.to_numpy()[mask])
            yield group, pct, part[part.row_index.isin(keep)]


def error_scale_and_confidence(results, n_boot):
    scale_rows, conf_rows = [], []
    for test in range(1, 6):
        pred = c.load_predictions(results, test)
        seeds = sorted(pred.seed.unique())
        for group, pct, part in blocks(pred, test):
            truth_case = part[(part.model == c.BASELINE) & (part.seed == seeds[0])]
            sq = {}
            for model in c.MODEL_ORDER:
                block = part[part.model == model]
                for out in OUTS:
                    e, _ = c.error_matrix(block, out, seeds)
                    sq[(model, out)] = e
                    rmse = np.sqrt(e.mean(axis=1))
                    y = truth_case[f"true_{out}"].to_numpy()
                    rel = []
                    for s in seeds:
                        b = block[block.seed == s]
                        rel.append(np.median(np.abs(b[f"pred_{out}"] - b[f"true_{out}"]) / b[f"true_{out}"]))
                    scale_rows.append({
                        "test": test, "group": group, "percentage": pct, "model": model, "output": out,
                        "n_cases": e.shape[1], "n_seeds": len(seeds),
                        "rmse_mm_mean": c.mm(rmse.mean()), "rmse_mm_sd": c.mm(rmse.std(ddof=1)),
                        "nrmse_mean": float((rmse / y.std()).mean()),
                        "median_relative_error_pct": 100 * float(np.mean(rel)),
                        "target_mean_mm": c.mm(y.mean()), "target_sd_mm": c.mm(y.std()),
                        "target_min_mm": c.mm(y.min()), "target_max_mm": c.mm(y.max()),
                    })
            for model in c.MODEL_ORDER[1:]:
                for out in OUTS:
                    em, eb = sq[(model, out)], sq[(c.BASELINE, out)]
                    rm, rb = np.sqrt(em.mean(axis=1)), np.sqrt(eb.mean(axis=1))
                    d, imp = c.paired_bootstrap(em, eb, n_boot=n_boot)
                    mean_imp, sd_imp, better, n, sign_p, wil_p = c.seed_tests(rm, rb)
                    lo, hi = np.percentile(imp, [2.5, 97.5])
                    conf_rows.append({
                        "test": test, "group": group, "percentage": pct, "model": model, "output": out,
                        "baseline_rmse_mm": c.mm(rb.mean()), "model_rmse_mm": c.mm(rm.mean()),
                        "delta_rmse_mm": c.mm((rm - rb).mean()),
                        "delta_rmse_mm_ci_low": c.mm(np.percentile(d, 2.5)), "delta_rmse_mm_ci_high": c.mm(np.percentile(d, 97.5)),
                        "improvement_pct_mean": mean_imp, "improvement_pct_sd": sd_imp,
                        "improvement_pct_ci_low": lo, "improvement_pct_ci_high": hi,
                        "seeds_better": better, "n_seeds": n, "sign_test_p": sign_p, "wilcoxon_p": wil_p,
                        "verdict": c.verdict(lo, hi),
                    })
    return pd.DataFrame(scale_rows), pd.DataFrame(conf_rows)


def test3_reference(results, scale, out_dir):
    pred = c.load_predictions(results, 3)
    one = pred[(pred.model == c.BASELINE) & (pred.seed == pred.seed.min()) & pred.condition.str.startswith("3B")]
    rows = []
    for level, b in one.groupby("uncertainty_level_percent"):
        shift = b.frequency_ratio - b.nominal_frequency_ratio
        row = {"uncertainty_pct": level, "n_cases": len(b)}
        for out in OUTS:
            err = b[f"nominal_{out}"] - b[f"true_{out}"]
            row[f"reference_rmse_mm_{out}"] = c.mm(np.sqrt(np.mean(err**2)))
        row.update({
            "ratio_shift_mean_abs": float(shift.abs().mean()), "ratio_shift_p95_abs": float(shift.abs().quantile(.95)),
            "ratio_shift_max_abs": float(shift.abs().max()),
            "pct_cases_shift_over_0.05": 100 * float((shift.abs() > .05).mean()),
        })
        for model in c.MODEL_ORDER:
            for case in ("3A_known", "3B_hidden"):
                s = scale[(scale.test == 3) & (scale.group == f"{case}_{level:g}pct") & (scale.model == model) & (scale.output == c.PEAK)]
                row[f"{c.SHORT[model]}_{case}_peak_rmse_mm"] = float(s.rmse_mm_mean.iloc[0])
        row["M1_3B_peak_over_reference"] = row["M1_3B_hidden_peak_rmse_mm"] / row[f"reference_rmse_mm_{c.PEAK}"]
        rows.append(row)
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    levels = table.uncertainty_pct.to_numpy()
    for i, case in enumerate(("3A_known", "3B_hidden")):
        for model in c.MODEL_ORDER:
            s = scale[(scale.test == 3) & (scale.model == model) & (scale.output == c.PEAK)]
            s = s.set_index("group").loc[[f"{case}_{lv:g}pct" for lv in levels]]
            ax[i].errorbar(levels, s.rmse_mm_mean, yerr=s.rmse_mm_sd, marker="o", capsize=3, label=c.SHORT[model])
        ax[i].set_xticks(levels, [f"±{lv:g}%" for lv in levels])
        ax[i].set_ylabel("Peak displacement RMSE (mm)")
        ax[i].set_title("Known properties (3A)" if i == 0 else "Hidden property changes (3B)")
        ax[i].grid(alpha=.3)
    ax[1].plot(levels, table[f"reference_rmse_mm_{c.PEAK}"], "k--", lw=1.5, label="Perfect nominal model")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    b10 = one[one.uncertainty_level_percent == 10]
    ax[2].hist(b10.frequency_ratio - b10.nominal_frequency_ratio, bins=40, color="0.5")
    ax[2].set_xlabel("Actual minus nominal frequency ratio (±10% case)")
    ax[2].set_ylabel("Number of cases")
    ax[2].set_title("Frequency-ratio shift from hidden changes")
    fig.tight_layout()
    fig.savefig(out_dir / "3c_test3_figure.png", dpi=200)
    plt.close(fig)
    return table


def test5_linear_estimate(results, scale):
    pred = c.load_predictions(results, 5)
    one = pred[(pred.model == c.BASELINE) & (pred.seed == pred.seed.min())]
    strong = one.nonlinear_stiffness_ratio_peak.to_numpy() >= .10
    rows = []
    for label, mask in (("5A all near-resonance", np.ones(len(one), bool)), ("5B strongly nonlinear", strong), ("5A outside 5B", ~strong)):
        b = one[mask]
        err = b.linear_peak_estimate - b[f"true_{c.PEAK}"]
        row = {"group": label, "n_cases": int(mask.sum()),
               "linear_estimate_peak_rmse_mm": c.mm(np.sqrt(np.mean(err**2))),
               "linear_estimate_median_ratio_to_reference": float(np.median(b.linear_peak_estimate / b[f"true_{c.PEAK}"])),
               "linear_estimate_p95_abs_error_mm": c.mm(np.percentile(np.abs(err), 95)),
               "reference_peak_mean_mm": c.mm(b[f"true_{c.PEAK}"].mean())}
        # Model errors on the same cases (seed mean).
        for model in c.MODEL_ORDER:
            m = pred[pred.model == model]
            per_seed = []
            for s, ms in m.groupby("seed"):
                ms = ms.set_index("row_index").loc[b.row_index]
                per_seed.append(np.sqrt(np.mean((ms[f"pred_{c.PEAK}"].to_numpy() - ms[f"true_{c.PEAK}"].to_numpy()) ** 2)))
            row[f"{c.SHORT[model]}_peak_rmse_mm"] = c.mm(np.mean(per_seed))
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(results):
    trained = c.mc.RESULTS / "trained"
    timing = pd.read_csv(c.ROOT / "diagnostics/one_mode_validation/results/timing_summary.csv").set_index("method")
    seven = float(timing.loc["seven_mode", "median_seconds"])
    one_mode = float(timing.loc["one_mode", "median_seconds"])
    linear = float(timing.loc["linear", "median_seconds"])
    original, split, _ = c.mc.fixed_data()
    row = original.iloc[split["test_indices"][:1]].copy()
    rows = []
    for kind in ("baseline", "physics_guided", "physics_estimate", "residual", "physics_regularised"):
        train_s = [json.loads(p.read_text())["training_seconds"] for p in (trained / kind / "data_100").glob("seed_*/complete.json")]
        latency = c.median_single_query_seconds(c.Loaded(kind, 42, 100), row)
        rows.append({"item": c.mc.NAMES[kind], "single_query_ms": 1000 * latency, "speedup_vs_7mode_simulation": seven / latency,
                     "training_seconds_mean_100pct": float(np.mean(train_s)), "n_training_runs": len(train_s)})
    rows += [
        {"item": "Seven-mode reference simulation (median, 30 cases)", "single_query_ms": 1000 * seven},
        {"item": "Nonlinear one-mode simulation (median, 30 cases)", "single_query_ms": 1000 * one_mode},
        {"item": "Linear analytical estimate (amortised, vectorised)", "single_query_ms": 1000 * linear},
        {"item": "Main dataset generation estimate (5,000 x seven-mode median), hours", "single_query_ms": np.nan,
         "training_seconds_mean_100pct": np.nan, "estimate_hours": 5000 * seven / 3600},
        {"item": "All datasets incl. Tests 2, 3, 5 (7,500 simulations), hours", "estimate_hours": 7500 * seven / 3600},
    ]
    return pd.DataFrame(rows)


def write_summary(out_dir, scale, conf, t3, t5, cost):
    L = []
    t1 = scale[(scale.test == 1) & (scale.group == "all") & (scale.output == c.PEAK) & (scale.model == c.BASELINE)].iloc[0]
    L.append(f"Peak displacement in the Test 1 set: mean {t1.target_mean_mm:.3f} mm, range {t1.target_min_mm:.3f} to {t1.target_max_mm:.3f} mm.")
    L.append(f"Seeds used: {int(t1.n_seeds)}.\n")
    L.append("Peak-displacement improvement vs Model 1 (mean over seeds, 95% bootstrap range, seeds better):")
    main = conf[(conf.output == c.PEAK) & conf.group.isin(["all", "5A_near_resonance", "5B_strong_nonlinearity", "3A_known_10pct", "3B_hidden_10pct"])]
    for _, r in main.iterrows():
        tag = f"Test {r.test} {r.group}" + (f" {r.percentage}% data" if r.test == 4 else "")
        L.append(f"  {tag:<38} {c.SHORT[r.model]:<4} {r.improvement_pct_mean:+6.1f}%  [{r.improvement_pct_ci_low:+6.1f}, {r.improvement_pct_ci_high:+6.1f}]  {r.seeds_better}/{r.n_seeds}  {r.verdict}")
    L.append("\nTest 3B: perfect nominal model vs Model 1 (peak RMSE, mm):")
    for _, r in t3.iterrows():
        L.append(f"  ±{r.uncertainty_pct:g}%: perfect nominal model {r[f'reference_rmse_mm_{c.PEAK}']:.3f}, M1 {r.M1_3B_hidden_peak_rmse_mm:.3f}, M4B {r.M4B_3B_hidden_peak_rmse_mm:.3f}; mean |frequency-ratio shift| {r.ratio_shift_mean_abs:.3f}")
    L.append("\nTest 5 linear-estimate peak RMSE (mm) vs model RMSE:")
    for _, r in t5.iterrows():
        L.append(f"  {r.group:<24} estimate {r.linear_estimate_peak_rmse_mm:.3f}  M1 {r.M1_peak_rmse_mm:.3f}  M4B {r.M4B_peak_rmse_mm:.3f}  (n={r.n_cases})")
    sim = cost[cost.item.str.startswith("Seven-mode")].single_query_ms.iloc[0]
    L.append(f"\nCost: seven-mode simulation {sim/1000:.3f} s per case; surrogates:")
    for _, r in cost.dropna(subset=["speedup_vs_7mode_simulation"]).iterrows():
        L.append(f"  {c.SHORT[r['item']]:<4} {r.single_query_ms:.3f} ms per query, {r.speedup_vs_7mode_simulation:,.0f}x faster, training {r.training_seconds_mean_100pct:.1f} s")
    hrs = cost.dropna(subset=["estimate_hours"])
    L.append(f"  Dataset generation estimate: {hrs.estimate_hours.iloc[0]:.1f} h (main), {hrs.estimate_hours.iloc[1]:.1f} h (all datasets), single CPU process")
    (out_dir / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="comparison/results/10seedresults")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    results = Path(a.results).resolve()
    out_dir = results / "analysis" / "performance_and_cost_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    scale, conf = error_scale_and_confidence(results, a.n_boot)
    scale.to_csv(out_dir / "3a_error_scale.csv", index=False)
    conf.to_csv(out_dir / "3b_confidence_vs_baseline.csv", index=False)
    t3 = test3_reference(results, scale, out_dir)
    t3.to_csv(out_dir / "3c_test3_reference.csv", index=False)
    t5 = test5_linear_estimate(results, scale)
    t5.to_csv(out_dir / "3d_test5_linear_estimate.csv", index=False)
    cost = cost_table(results)
    cost.to_csv(out_dir / "3e_cost.csv", index=False)
    write_summary(out_dir, scale, conf, t3, t5, cost)
    print(f"\nStep 3 outputs: {out_dir}")


if __name__ == "__main__":
    main()
