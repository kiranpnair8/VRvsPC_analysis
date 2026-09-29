#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu006
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/rizk_lab/shared/kiran/VRvsPC_analysis/jobs/logs/reve_centroid_figure_paper_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/VRvsPC_analysis/jobs/logs/reve_centroid_figure_paper_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/VRvsPC_analysis
RESULT_DIR="$PROJECT_ROOT/reve_embedding_analysis"
INPUT_CSV="$RESULT_DIR/subject_level_distances.csv"

mkdir -p "$PROJECT_ROOT/jobs/logs"
cd "$PROJECT_ROOT"

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/vr_pc
set -u

if [ ! -f "$INPUT_CSV" ]; then
  echo "ERROR: Missing subject-level distance table: $INPUT_CSV" >&2
  exit 1
fi

export MPLBACKEND=Agg

python plot_reve_centroid_distances_paper.py \
  --input_csv "$INPUT_CSV" \
  --out_dir "$RESULT_DIR" \
  --dpi 400

