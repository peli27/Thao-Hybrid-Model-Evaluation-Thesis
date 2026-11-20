"""Run the final five-model comparison for Test 5: difficult nonlinear response.

Inputs: The saved datasets and split, using matched seeds 42, 123 and 456.
Outputs: Results in comparison/results/3seedresults/test_5/.

Uses multiseed_core.py and nine-input Model 4B; missing runs train automatically.
"""
from multiseed_core import evaluate

if __name__ == "__main__":
    evaluate(5)
