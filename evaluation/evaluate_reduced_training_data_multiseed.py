"""Run the final five-model comparison for Test 4: reduced training data.

Inputs: The saved datasets and split, using matched seeds 42, 123 and 456.
Outputs: Results in comparison/results/3seedresults/test_4/.

Uses multiseed_core.py and nine-input Model 4B; missing runs train automatically.
Training budgets are 100%, 50%, 25% and 10% of the original training pool.
"""
from multiseed_core import evaluate

if __name__ == "__main__":
    evaluate(4)