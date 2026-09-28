#!/bin/bash

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/VRvsPC_analysis
SUMMARY_CSV="$PROJECT_ROOT/out/cv5/results/supervised_cv5_summary.csv"
PER_FOLD_CSV="$PROJECT_ROOT/out/cv5/results/supervised_cv5_per_fold.csv"
OUTPUT_DIR="$PROJECT_ROOT/figures_cv5"

cd "$PROJECT_ROOT"

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/vr_pc
set -u

if [ ! -f "$SUMMARY_CSV" ]; then
  echo "ERROR: Missing CV5 summary: $SUMMARY_CSV" >&2
  exit 1
fi

if [ ! -f "$PER_FOLD_CSV" ]; then
  echo "ERROR: Missing CV5 per-fold results: $PER_FOLD_CSV" >&2
  exit 1
fi

export MPLBACKEND=Agg

python plot_cv5_far_frr.py \
  --summary "$SUMMARY_CSV" \
  --per_fold "$PER_FOLD_CSV" \
  --out_dir "$OUTPUT_DIR" \
  --dpi 300

