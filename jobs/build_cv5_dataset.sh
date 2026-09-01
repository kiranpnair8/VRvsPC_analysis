#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/VRvsPC_analysis/jobs/logs/build_cv5_dataset_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/VRvsPC_analysis/jobs/logs/build_cv5_dataset_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/VRvsPC_analysis
DATA_DIR=/REPLACE/WITH/RAW/CSV/PATH

mkdir -p "$PROJECT_ROOT/jobs/logs"
mkdir -p "$PROJECT_ROOT/out/cv5/results"

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR does not exist: $DATA_DIR"
  echo "Edit jobs/build_cv5_dataset.sh and set DATA_DIR to the raw CSV folder."
  exit 1
fi

cd "$PROJECT_ROOT"

module purge
module load cuda/12.3

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/vr_pc
set -u

echo "hostname: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"
echo "date: $(date)"
nvidia-smi || true

python - <<'PY'
import sys
import torch
import transformers
import braindecode

print("Python version:", sys.version)
print("Executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("torch CUDA version:", torch.version.cuda)
print("CUDA availability:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("transformers version:", transformers.__version__)
print("braindecode version:", getattr(braindecode, "__version__", "unknown"))
PY

python cv_build_verification_dataset.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$PROJECT_ROOT/out/cv5" \
  --seed 42 \
  --n_imposters 3
