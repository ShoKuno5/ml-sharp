#!/bin/bash
# SLURM submission template for iruka2 (A100). The user owns submission.
#
#   FILL IN the partition / account / gres for your cluster, then:
#     sbatch training/sharp_train/slurm/train_iruka2.sbatch
#
# Defaults to the single-GPU overfit sanity check. For multi-GPU training (Step 4) switch the
# launch line to the train entrypoint (added with the Step 3 dataset) and bump --gres.
#
#SBATCH --job-name=sharp-distort
#SBATCH --output=/data/uni0/users/kuno/3dfm_distortion/results/slurm-%j.out
#SBATCH --gres=gpu:1                 # overfit: 1 GPU; pilot/full: 4-8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
## #SBATCH --partition=<FILL_ME>
## #SBATCH --account=<FILL_ME>

set -euo pipefail

REPO=/home/mil/kuno/models/ml-sharp
ENV_PY=/data/uni0/users/kuno/envs/sharp/bin/python

export PYTHONPATH="${REPO}/src:${REPO}/training"
export HF_HOME=/data/uni0/users/kuno/3dfm_distortion/cache/huggingface
export TORCH_CUDA_ARCH_LIST=9.0          # H100/A100 (Hopper/Ampere); avoids building all archs
export TORCH_HOME=/data/uni0/users/kuno/3dfm_distortion/cache/torch
export TMPDIR=/tmp

echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- Overfit sanity (single GPU): the Step 1b acceptance ---
srun "${ENV_PY}" -m sharp_train.scripts.overfit --steps 300 --render-size 256 --stage A

# --- Pilot training (multi-GPU, Step 4) — switch to this once the dataset exists ---
# srun torchrun --nproc_per_node=4 -m sharp_train.scripts.train \
#     --config "${REPO}/training/sharp_train/config/pilot.yaml"
