#!/bin/bash

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/VRvsPC_analysis
CV5_NPZ="$PROJECT_ROOT/out/cv5/cv5_verification_dataset_lphp10_50.npz"
CACHE_DIR="$PROJECT_ROOT/out/cv5/cache/reve"
OUTPUT_DIR="$PROJECT_ROOT/reve_embedding_analysis"

cd "$PROJECT_ROOT"

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/vr_pc
set -u

if [ ! -f "$CV5_NPZ" ]; then
  echo "ERROR: Missing CV5 dataset: $CV5_NPZ" >&2
  exit 1
fi

if [ ! -d "$CACHE_DIR" ]; then
  echo "ERROR: Missing supervised REVE cache directory: $CACHE_DIR" >&2
  exit 1
fi

export MPLBACKEND=Agg

python analyze_reve_embeddings.py \
  --cv5_npz "$CV5_NPZ" \
  --cache_dir "$CACHE_DIR" \
  --out_dir "$OUTPUT_DIR" \
  --seed 42 \
  --tsne_perplexity 30 \
  --tsne_max_iter 1000 \
  --bootstrap_repeats 1000 \
  --bootstrap_ci_repeats 10000 \
  --dpi 300

