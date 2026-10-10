#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHON=".venv/bin/python"
"$PYTHON" validation_wam_final_robustness_20261008/run_robustness.py \
  > validation_wam_final_robustness_20261008/logs/full_run.log 2>&1
"$PYTHON" validation_wam_final_robustness_20261008/finalize.py \
  > validation_wam_final_robustness_20261008/logs/finalize.log 2>&1
"$PYTHON" validation_wam_final_robustness_20261008/verify_artifacts.py \
  > validation_wam_final_robustness_20261008/logs/verification.log 2>&1
