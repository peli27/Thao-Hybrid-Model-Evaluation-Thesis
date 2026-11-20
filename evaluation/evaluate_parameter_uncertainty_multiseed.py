"""Run the final five-model comparison for Test 3: known and hidden parameter uncertainty.

Inputs: The saved datasets and split, using matched seeds 42, 123 and 456.
Outputs: Results in comparison/results/3seedresults/test_3/.

Uses multiseed_core.py and nine-input Model 4B; missing runs train automatically.
"""
from multiseed_core import evaluate

if __name__ == "__main__":
    evaluate(3)
