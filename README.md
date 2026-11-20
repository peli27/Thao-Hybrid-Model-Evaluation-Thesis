# Physics-Integrated Neural Networks for Nonlinear Structural Dynamics

This repository accompanies the thesis **Evaluating Physics Integration Strategies in Neural-Network Surrogates for Nonlinear Structural Dynamics**.

The study uses a nonlinear beam simulator to compare a baseline neural network with four approaches to incorporating physical information:

1. **Model 1:** Baseline MLP
2. **Model 2:** Physics-guided features
3. **Model 3:** Physics-estimate-assisted learning
4. **Model 4B:** Revised residual learning
5. **Model 5:** Physics-regularised learning (weight 0.01)

The models are compared across five tests: interpolation, unseen geometry, parameter uncertainty, reduced training data and difficult nonlinear responses.

## Repository structure

| Folder | Contents |
| --- | --- |
| `beam_model/` | Nonlinear beam simulator and numerical verification |
| `dataset_generation/` | Scripts for generating the datasets |
| `data/` | Main dataset, evaluation datasets and fixed data split |
| `models/` | Model implementations and training scripts |
| `evaluation/` | Scripts for Tests 1–5 and prediction checks |
| `comparison/` | Saved models and evaluation results |
| `diagnostics/` | Supporting investigations |
| `analysis/` | Additional analysis of the final results |

## Data and results

The main dataset contains **5,000 simulated cases**, split into **3,499 training**, **750 validation** and **751 test** cases.

- **Final ten-seed results:** `comparison/results/10seedresults/`
- **Saved multi-seed model checkpoints:** `comparison/results/3seedresults/trained/`
- **Numerical verification results:** `beam_model/verification/verification_results.json`

The final comparisons use seeds **1–7, 42, 123 and 456**. Earlier three-seed and preliminary results are also retained for reference.

See REPRODUCIBILITY.md for setup instructions and commands to repeat the verification and evaluations.
