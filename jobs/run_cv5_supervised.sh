#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/VRvsPC_analysis/jobs/logs/run_cv5_supervised_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/VRvsPC_analysis/jobs/logs/run_cv5_supervised_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/VRvsPC_analysis

mkdir -p "$PROJECT_ROOT/jobs/logs"
mkdir -p "$PROJECT_ROOT/out/cv5/results"
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
nvidia-smi

python - <<'PY'
import sys
from pathlib import Path

import torch
import transformers
import braindecode
from transformers import AutoModel

project_root = Path("/home/rizk_lab/shared/kiran/VRvsPC_analysis")
model_dir = project_root / "models"

print("Python version:", sys.version)
print("Executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("torch CUDA version:", torch.version.cuda)
print("CUDA availability:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("transformers version:", transformers.__version__)
print("braindecode version:", getattr(braindecode, "__version__", "unknown"))

AutoModel.from_pretrained(
    str(model_dir / "reve-base"),
    trust_remote_code=True,
    torch_dtype="auto",
    local_files_only=True,
)
AutoModel.from_pretrained(
    str(model_dir / "reve-positions"),
    trust_remote_code=True,
    torch_dtype="auto",
    local_files_only=True,
)
print("Local REVE loading: OK")
PY

python run_cv5_supervised.py \
  --npz "$PROJECT_ROOT/out/cv5/cv5_verification_dataset_lphp10_50.npz" \
  --model all \
  --scenario all \
  --use_claimed_id \
  --model_dir "$PROJECT_ROOT/models" \
  --out_dir "$PROJECT_ROOT/out/cv5/results"
