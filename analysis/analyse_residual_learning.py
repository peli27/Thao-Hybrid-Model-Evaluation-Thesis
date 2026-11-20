"""Investigate the performance of residual-learning surrogate models.

Inputs: Main reference dataset, fixed data partitions, analytical displacement
estimates and trained neural-network models.

Outputs: Diagnostic figures, target statistics and regional model comparisons

This script investigates why the original Residual-Learning Hybrid (Model 4A)
performed less accurately than the revised implementation (Model 4B).

It examines errors in the linear analytical displacement estimate, compares
the learning targets of the baseline and residual models, and evaluates
Model 4A and Model 4B across different resonance regions.

Missing model checkpoints are trained using the common experimental settings.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import analysis.analysis_utils as c  # noqa: E402


def regions(r):
    return {"all": np.ones(len(r), bool), "below_resonance": r < .85, "near_resonance": (r >= .85) & (r <= 1.20), "above_resonance": r > 1.20}


def roughness(x, y):
    """Mean change in the target between each case and its nearest neighbour in standardised input space."""
    z = StandardScaler().fit_transform(x)
    _, idx = NearestNeighbors(n_neighbors=2).fit(z).kneighbors(z)
    return float(np.mean(np.abs(y - y[idx[:, 1]])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="comparison/results/10seedresults")
    a = p.parse_args()
    results = Path(a.results).resolve()
    out = results / "analysis" / "residual_learning_analysis"
    out.mkdir(parents=True, exist_ok=True)
    seeds = c.seeds_in(results)

    original, split, _ = c.mc.fixed_data()
    _, estimate = c.mc.physics_features(original)
    ref = original[c.PEAK].to_numpy()
    lin = estimate[:, 0]
    resid = ref - lin
    r = original.frequency_ratio.to_numpy()
    zeta = original.zeta.to_numpy()
    tr = split["train_indices"]

    # 4a: how wrong the linear estimate is across the frequency ratio.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sc = ax.scatter(r, lin / ref, c=zeta, s=6, cmap="viridis")
    ax.axhline(1, color="k", lw=.8)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency ratio r = f_exc / f1")
    ax.set_ylabel("Linear estimate / reference peak displacement")
    fig.colorbar(sc, label="Damping ratio ζ")
    fig.tight_layout(); fig.savefig(out / "4a_linear_estimate_vs_ratio.png", dpi=200); plt.close(fig)

    # 4b: the two learning targets on the same scale.
    lim = 1.05 * c.mm(max(np.abs(ref).max(), np.abs(resid).max()))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ax[0].scatter(r, c.mm(ref), s=5, color="tab:blue")
    ax[0].set_title("Model 1 target: reference peak displacement")
    ax[1].scatter(r, c.mm(resid), s=5, color="tab:red")
    ax[1].set_title("Model 4A target: reference minus linear estimate")
    for axis in ax:
        axis.axhline(0, color="k", lw=.6); axis.set_xlabel("Frequency ratio r"); axis.set_ylim(-lim, lim); axis.grid(alpha=.3)
    ax[0].set_ylabel("Displacement (mm)")
    fig.tight_layout(); fig.savefig(out / "4b_targets_vs_ratio.png", dpi=200); plt.close(fig)

    # 4c: size and roughness of each target on the training set.
    x7 = original[c.mc.INPUTS].to_numpy(float)
    rows = []
    for name, y in (("Reference peak (Model 1 target)", ref), ("Residual peak (Model 4A target)", resid)):
        for band, mask in (("all training cases", np.ones(len(tr), bool)), ("0.95 <= r <= 1.05", (r[tr] >= .95) & (r[tr] <= 1.05))):
            idx = tr[mask]
            yy = y[idx]
            rows.append({"target": name, "cases": band, "n": len(idx), "sd_mm": c.mm(yy.std()), "p99_abs_mm": c.mm(np.percentile(np.abs(yy), 99)),
                         "max_abs_mm": c.mm(np.abs(yy).max()), "nearest_neighbour_change_mm": c.mm(roughness(x7[idx], yy))})
    diff = pd.DataFrame(rows)
    diff.to_csv(out / "4c_target_difficulty.csv", index=False)

    # 4d: train 4A with the shared training code and compare with 4B by region.
    test_df = original.iloc[split["test_indices"]]
    val_df = original.iloc[split["validation_indices"]]
    rows = []
    for kind in ("residual_4a", "residual"):
        for s in seeds:
            c.mc.train(original, split, kind, s, split["train_indices"], 100)
            m = c.Loaded(kind, s, 100)
            for part, df in (("validation", val_df), ("test (Test 1)", test_df)):
                pred = m.predict(df)
                for reg, mask in regions(df.frequency_ratio.to_numpy()).items():
                    row = {"model": c.SHORT[c.mc.NAMES[kind]], "seed": s, "split": part, "region": reg, "n": int(mask.sum())}
                    for j, out_name in ((4, "peak"), (5, "rms")):
                        e = pred[mask, j] - df[c.mc.OUTPUTS[j]].to_numpy()[mask]
                        row[f"{out_name}_rmse_mm"] = c.mm(np.sqrt(np.mean(e**2)))
                    row["negative_peak_predictions"] = int((pred[mask, 4] < 0).sum())
                    rows.append(row)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "4d_4A_vs_4B_per_seed.csv", index=False)
    summary = per_seed.groupby(["split", "region", "model"], as_index=False).agg(
        n=("n", "first"), peak_rmse_mm_mean=("peak_rmse_mm", "mean"), peak_rmse_mm_sd=("peak_rmse_mm", "std"),
        rms_rmse_mm_mean=("rms_rmse_mm", "mean"), rms_rmse_mm_sd=("rms_rmse_mm", "std"), n_seeds=("seed", "nunique"))
    summary.to_csv(out / "4d_4A_vs_4B_by_region.csv", index=False)

    L = [f"Seeds: {len(seeds)}", "Linear estimate / reference (peak), all 5,000 cases:"]
    for lo, hi in ((0.5, 0.85), (0.85, 0.95), (0.95, 1.05), (1.05, 1.2), (1.2, 1.5)):
        m = (r >= lo) & (r < hi)
        q = lin[m] / ref[m]
        L.append(f"  {lo:.2f} <= r < {hi:.2f}: median {np.median(q):.2f}, 95th percentile {np.percentile(q, 95):.1f}, max {q.max():.1f}  (n={m.sum()})")
    L.append("\nTarget difficulty (training set):")
    for _, d in diff.iterrows():
        L.append(f"  {d.target:<34} {d.cases:<20} SD {d.sd_mm:.3f} mm, 99th pct |y| {d.p99_abs_mm:.3f} mm, max |y| {d.max_abs_mm:.3f} mm, nearest-neighbour change {d.nearest_neighbour_change_mm:.3f} mm")
    L.append("\nPeak RMSE (mm), mean ± SD over seeds:")
    for (part, reg), g in summary.groupby(["split", "region"], sort=False):
        vals = "  ".join(f"{r_.model} {r_.peak_rmse_mm_mean:.3f} ± {r_.peak_rmse_mm_sd:.3f}" for _, r_ in g.iterrows())
        L.append(f"  {part:<14} {reg:<16} {vals}")
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nStep 4 outputs: {out}")


if __name__ == "__main__":
    main()
