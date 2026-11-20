#!/bin/bash
# Run from any directory using Python from PATH or an explicit ANALYSIS_PYTHON.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-$(command -v python3 || true)}"

if [[ ! -x "$ANALYSIS_PYTHON" ]]; then
    echo "Cannot find a Python executable at: $ANALYSIS_PYTHON" >&2
    echo "Set ANALYSIS_PYTHON to the Python executable with requirements.txt installed." >&2
    exit 1
fi

cd "$PROJECT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.mplconfig}"
"$ANALYSIS_PYTHON" -c 'import matplotlib, numpy, pandas, scipy, torch, sklearn, joblib'

for script in analyse_performance_and_cost.py analyse_residual_learning.py evaluate_baseline_and_framework.py; do
    echo "Running $script with $ANALYSIS_PYTHON"
    "$ANALYSIS_PYTHON" -u "analysis/$script" "$@"
done
