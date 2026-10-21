#!/bin/bash
#SBATCH --job-name=surrogate-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200000M
#SBATCH --time=10:00:00
#
# Runs scripts/surrogate_check.py; summarize with scripts/surrogate_check_analysis.py.
#
#   sbatch scripts/surrogate_check.sbatch
#   MODEL=meta-llama/Llama-3.1-8B-Instruct OUT=results/surrogate_check_llama.csv \
#       sbatch scripts/surrogate_check.sbatch

set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-compaction}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results

MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT="${OUT:-results/surrogate_check.csv}"
python scripts/surrogate_check.py \
  --model "$MODEL" --dataset quality --n-articles 5 \
  --query-config ss-plus-repeat --fast-config repeat \
  --max-queries 8192 --draws 8 \
  --out "$OUT" ${ARGS:-}
