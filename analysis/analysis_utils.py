"""Shared utilities for supplementary surrogate-model analyses.

Inputs: Saved datasets, trained model checkpoints and experimental results.

Outputs: Reusable functions for loading models, generating predictions,
calculating error statistics and evaluating computational performance.

This module extends the common evaluation framework in
evaluation/multiseed_core.py to support additional diagnostic experiments,
including the original seven-input residual model (Model 4A) and a baseline
with log-transformed displacement targets.

The functions ensure that supplementary analyses use consistent datasets,
training settings and evaluation procedures.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import train_test_split

# Match multiseed_core.evaluate: small networks run with one CPU thread.
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import multiseed_core as mc  # noqa: E402

PEAK, RMS = "peak_displacement", "rms_displacement"
BASELINE = mc.NAMES["baseline"]
MODEL_ORDER = [mc.NAMES[k] for k in ("baseline", "physics_guided", "physics_estimate", "residual", "physics_regularised")]
SHORT = {mc.NAMES["baseline"]: "M1", mc.NAMES["physics_guided"]: "M2", mc.NAMES["physics_estimate"]: "M3", mc.NAMES["residual"]: "M4B", mc.NAMES["physics_regularised"]: "M5"}

# Extra model kinds used only by the analysis scripts (not part of the main five-model loop).
EXTRA = {
    "residual_4a": ("Model 4A Residual (7 inputs)", 7),
    "baseline_log": ("Model 1 Baseline (log displacement)", 7),
}
for _kind, (_name, _size) in EXTRA.items():
    mc.NAMES.setdefault(_kind, _name)
    mc.SIZES.setdefault(_kind, _size)
    SHORT[_name] = {"residual_4a": "M4A", "baseline_log": "M1-log"}[_kind]

_original_arrays = mc.arrays


def _arrays(df, kind, *, include_targets=True):
    if kind == "residual_4a":
        # 4A: seven raw inputs; the network learns reference minus linear estimate.
        raw, target, estimate = _original_arrays(df, "baseline", include_targets=include_targets)
        if target is not None:
            target[:, 4:6] -= estimate
        return raw, target, estimate
    if kind == "baseline_log":
        # Baseline with log-transformed peak and RMS displacement targets.
        raw, target, estimate = _original_arrays(df, "baseline", include_targets=include_targets)
        if target is not None:
            if np.any(target[:, 4:6] <= 0):
                raise ValueError("Non-positive displacement cannot be log-transformed")
            target[:, 4:6] = np.log(target[:, 4:6])
        return raw, target, estimate
    return _original_arrays(df, kind, include_targets=include_targets)


mc.arrays = _arrays  # train() looks up arrays at call time, so it now understands the extra kinds.


class Loaded:
    """A trained checkpoint held in memory, so repeated predictions exclude file I/O."""

    def __init__(self, kind, seed, percentage):
        f = mc.folder(kind, seed, percentage)
        ck = torch.load(f / "model.pt", map_location="cpu", weights_only=False)
        if (ck["kind"], ck["seed"], ck["percentage"]) != (kind, seed, percentage):
            raise ValueError(f"Checkpoint metadata mismatch in {f}")
        self.kind = kind
        self.xs = joblib.load(f / "x_scaler.pkl")
        self.ys = joblib.load(f / "y_scaler.pkl")
        self.model = mc.network(kind)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def predict(self, df):
        x, _, physics = mc.arrays(df, self.kind, include_targets=False)
        with torch.no_grad():
            scaled = self.model(torch.tensor(self.xs.transform(x), dtype=torch.float32)).numpy()
        pred = self.ys.inverse_transform(scaled).astype(float)
        if self.kind in ("residual", "residual_4a"):
            pred[:, 4:6] += physics
        if self.kind == "baseline_log":
            pred[:, 4:6] = np.exp(pred[:, 4:6])
        return pred


def training_subsets(original, split):
    """Training indices per data budget, identical to multiseed_core.evaluate (Test 4)."""
    pool = split["train_indices"]
    subsets = {100: pool}
    current = pool
    for pct in mc.PERCENTAGES[1:]:
        count = int(np.floor(len(pool) * pct / 100))
        ratio = original.iloc[current].frequency_ratio.to_numpy(float)
        labels = np.where(ratio < .85, 0, np.where(ratio <= 1.2, 1, 2))
        current, _ = train_test_split(current, train_size=count, stratify=labels, random_state=2026 + pct)
        subsets[pct] = np.sort(current)
    return subsets


def load_predictions(results, test):
    path = Path(results) / f"test_{test}" / "predictions.csv.gz"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run Step 1/2 first.")
    return pd.read_csv(path)


def seeds_in(results):
    return sorted(load_predictions(results, 1).seed.unique().tolist())


def group_masks(test, cases):
    """Subgroups for one model/seed/condition block, matching the thesis tables."""
    if test == 3:
        return {cases.condition.iloc[0]: np.ones(len(cases), bool)}
    return {g: np.asarray(m) for g, m in mc.groups_for(test, cases).items() if np.asarray(m).any()}


def error_matrix(block, output, seeds):
    """Squared errors as an array [seed, case] with cases in a fixed order."""
    rows, order = [], None
    for s in seeds:
        b = block[block.seed == s].sort_values("row_index")
        if order is None:
            order = b.row_index.to_numpy()
        elif not np.array_equal(order, b.row_index.to_numpy()):
            raise ValueError("Case order differs between seeds or models")
        rows.append((b[f"pred_{output}"].to_numpy() - b[f"true_{output}"].to_numpy()) ** 2)
    return np.vstack(rows), order


def paired_bootstrap(sq_model, sq_base, n_boot=2000, seed=2026, chunk=200):
    """Bootstrap over test cases; the same resampled cases are used for every seed and both models.

    Returns arrays of (model RMSE - baseline RMSE) and paired % improvement, each averaged over seeds.
    """
    rng = np.random.default_rng(seed)
    n = sq_model.shape[1]
    deltas, improvements = [], []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        rm = np.sqrt(sq_model[:, idx].mean(axis=2))
        rb = np.sqrt(sq_base[:, idx].mean(axis=2))
        deltas.append((rm - rb).mean(axis=0))
        improvements.append((100 * (rb - rm) / rb).mean(axis=0))
    return np.concatenate(deltas), np.concatenate(improvements)


def seed_tests(model_rmse, base_rmse):
    """Paired seed statistics: mean improvement, seeds better, sign-test and Wilcoxon p-values."""
    imp = 100 * (base_rmse - model_rmse) / base_rmse
    better = int((imp > 0).sum())
    n = len(imp)
    sign_p = float(stats.binomtest(better, n, 0.5).pvalue) if n else np.nan
    try:
        wil_p = float(stats.wilcoxon(model_rmse - base_rmse).pvalue) if n >= 5 else np.nan
    except ValueError:
        wil_p = np.nan
    return float(imp.mean()), float(imp.std(ddof=1)) if n > 1 else np.nan, better, n, sign_p, wil_p


def verdict(lo, hi):
    if lo > 0:
        return "clearly better"
    if hi < 0:
        return "clearly worse"
    return "no clear difference"


def implausible(pred_frame):
    """True where a prediction breaks any basic physical condition used in the thesis."""
    f = pred_frame[[f"pred_f{i}" for i in range(1, 5)]].to_numpy()
    peak, rms = pred_frame[f"pred_{PEAK}"].to_numpy(), pred_frame[f"pred_{RMS}"].to_numpy()
    bad = (f <= 0).any(axis=1) | (np.diff(f, axis=1) <= 0).any(axis=1)
    bad |= (peak < 0) | (rms < 0) | (rms > peak)
    return bad


def median_single_query_seconds(loaded, row, repeats=2000, warmup=200):
    torch.set_num_threads(1)
    for _ in range(warmup):
        loaded.predict(row)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        loaded.predict(row)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def mm(x):
    return 1000.0 * np.asarray(x)
