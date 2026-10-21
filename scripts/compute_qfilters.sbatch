#!/bin/bash
#SBATCH --job-name=qfilters
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200000M
#SBATCH --time=04:00:00
#
# Computes the Pile Q-Filters (qfilters/pile/<model>.pt) for all five models.
# Existing files are skipped.
#
#   CONDA_ENV=compaction sbatch scripts/compute_qfilters.sbatch

set -o pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
date; hostname; pwd

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-compaction}"
which python
nvidia-smi

MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-4B-Instruct-2507"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
  "Qwen/Qwen3-30B-A3B"
)

fail=0
for model in "${MODELS[@]}"; do
  out="qfilters/pile/$(basename "$model").pt"
  if [ -f "$out" ]; then
    echo "== $model: $out exists, skipping"
    continue
  fi
  echo "== $model -> $out"
  if python scripts/compute_qfilters.py --model "$model" --out "$out"; then
    echo "== $model: done"
  else
    echo "== $model: FAILED (continuing with the remaining models)"
    fail=1
  fi
done

ls -la qfilters/pile/ 2>/dev/null || echo "no filter files"
exit $fail
