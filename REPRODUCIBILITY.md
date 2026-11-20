# Reproducibility Guide

This guide explains how to check the numerical beam model and reproduce the five final model evaluations. The repository includes the datasets, fixed split, saved model checkpoints and original results.

## 1. Setup

Use **Python 3.10**. From the repository root, create an environment and install the pinned dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the following commands from the repository root. The evaluation code uses PyTorch on the CPU.

## 2. Data and saved results

- Main dataset: `data/main/nonlinear_beam_dataset.csv` (5,000 cases)
- Fixed split: `data/splits/main_split_indices.npz` (3,499 training, 750 validation, 751 test)
- Additional test datasets: `data/test_datasets/`
- Final results: `comparison/results/10seedresults/test_1/` through `test_5/`
- Trained checkpoints and preprocessing scalers: `comparison/results/3seedresults/trained/`

The saved datasets and split should be used to reproduce the reported comparisons. Dataset-generation scripts are available in `dataset_generation/`, but regenerating the data is not required for the evaluation steps below and can overwrite supplied files.

## 3. Verify the beam simulator

```bash
python beam_model/verification/Verify_beam.py --output reproduced/verification_results.json
```

The original report is saved at `beam_model/verification/verification_results.json`.

## 4. Repeat the five model evaluations

The final thesis comparisons use **ten matched training seeds**: `1 2 3 4 5 6 7 42 123 456`.

To reproduce all five tests without replacing the supplied result tables:

```bash
for test in 1 2 3 4 5; do
    python evaluation/multiseed_core.py --test "$test" \
        --seeds 1 2 3 4 5 6 7 42 123 456 \
        --out "comparison/results/reproduced/test_$test"

    python evaluation/check_predictions.py --test "$test" \
        --out "comparison/results/reproduced/test_$test"
done
```

The tests cover interpolation (1), unseen geometry (2), parameter uncertainty (3), reduced training data (4) and nonlinear response (5). Existing checkpoints are reused. Missing checkpoints are trained automatically and saved under `comparison/results/3seedresults/trained/`.

Each output folder contains per-seed metrics, comparison summaries, case-level predictions and run metadata. Compare these files with the supplied results in `comparison/results/10seedresults/`.

## 5. Supplementary analyses

Supporting diagnostic scripts and results are in `diagnostics/`. Additional scripts are in `analysis/`, with saved outputs in `comparison/results/10seedresults/analysis/`.

Results may vary slightly across environments, especially computation times. Full independent reproduction on another computer has not been established.
