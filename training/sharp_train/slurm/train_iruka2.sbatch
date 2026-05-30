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
ENV=/data/uni0/users/kuno/envs/sharp
ENV_PY="${ENV}/bin/python"

# gsplat 1.5.3 ships no prebuilt CUDA backend for py3.13, so its extension is JIT-compiled on
# first render. That needs a real `ninja` (present in the env) and `nvcc` — but the pyenv shims
# shadow both, so put the env bin and CUDA bin AHEAD of everything. Cache the built extension on
# uni0 (ayu home is ~full and slow).
# Select a CUDA toolkit at runtime: neither the bare /usr/local/cuda symlink nor a single pinned
# version is uniform across nodes (the symlink resolves per-node, e.g. beluga->12.2; and 12.8 is
# absent on some nodes, e.g. beluga tops out at 12.5). Pick the first present toolkit that accepts
# the nodes' gcc 13.3 -- CUDA >= 12.5 (avoid <= 12.4). This is what the green acceptance used.
CUDA_DIR=""
for v in 12.8 12.6 12.5; do
  if [ -x "/usr/local/cuda-${v}/bin/nvcc" ]; then CUDA_DIR="/usr/local/cuda-${v}"; break; fi
done
if [ -z "${CUDA_DIR}" ]; then
  echo "ERROR: no CUDA>=12.5 toolkit found on $(hostname) for the gsplat JIT build" >&2
  exit 1
fi
echo "using CUDA toolkit: ${CUDA_DIR}"
export PATH="${ENV}/bin:${CUDA_DIR}/bin:${PATH}"
export CUDA_HOME="${CUDA_DIR}"
export TORCH_EXTENSIONS_DIR=/data/uni0/users/kuno/3dfm_distortion/cache/torch_extensions

export PYTHONPATH="${REPO}/src:${REPO}/training"
export HF_HOME=/data/uni0/users/kuno/3dfm_distortion/cache/huggingface
export TORCH_CUDA_ARCH_LIST=8.0          # iruka2 = A100 (sm_80); use 9.0 for H100 (ryugu/isegoi)
export TORCH_HOME=/data/uni0/users/kuno/3dfm_distortion/cache/torch
export TMPDIR=/tmp

echo "host: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- Overfit sanity (single GPU): the Step 1b acceptance ---
srun "${ENV_PY}" -m sharp_train.scripts.overfit --steps 300 --render-size 256 --stage A

# --- Pilot training (multi-GPU, Step 4) — switch to this once the dataset exists ---
# srun torchrun --nproc_per_node=4 -m sharp_train.scripts.train \
#     --config "${REPO}/training/sharp_train/config/pilot.yaml"
