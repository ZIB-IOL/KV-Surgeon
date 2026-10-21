#!/bin/bash
# SLURM job submitted by controller.py: runs one run of a wandb sweep.
#
#   runner.sh <codebase_path> <sweep_id> <conda_env>
#
# Optional environment: HF_HOME (model cache), WANDB_DIR, LOCAL_SCRATCH
# (node-local tmp dir), http_proxy / https_proxy.

codebase_path=$1
sweep_id=$2
conda_env=$3

echo "codebase_path: $codebase_path"
echo "sweep_id: $sweep_id"
echo "conda_env: $conda_env"

cd "$codebase_path" || exit 1

date; hostname; id; pwd

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$conda_env"
which python
nvidia-smi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
python -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('torch.cuda.device_count():', torch.cuda.device_count())" || true

if [ -n "$LOCAL_SCRATCH" ] && [ -d "$LOCAL_SCRATCH" ]; then
  mkdir -p "$LOCAL_SCRATCH/$USER/tmp"
  export TMPDIR="$LOCAL_SCRATCH/$USER/tmp"
fi

# Run the agent directly in this allocation: a nested srun step can lose its
# GPU binding.
wandb agent --count 1 "$sweep_id"
agent_exit=$?

# A finished or stopped sweep makes the agent fail immediately. Exit 0 then,
# so controller.py does not mistake it for a broken node.
if [ $agent_exit -ne 0 ]; then
  sweep_state=$(python -c "import wandb; print(wandb.Api().sweep('$sweep_id').state)" 2>/dev/null)
  echo "wandb agent exited $agent_exit; sweep state: ${sweep_state:-unknown}"
  case "$sweep_state" in
    RUNNING|PENDING|"") exit $agent_exit ;;
    *) echo "Sweep is not running; treating as success."; exit 0 ;;
  esac
fi
