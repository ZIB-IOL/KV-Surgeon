#!/bin/bash
#SBATCH --job-name=head-budget-opt
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --requeue
#
# Step 1 of the head-budget optimization: influence curves on QuALITY dev
# articles 90-99, one per array task. Step 2 (the solve) is described in
# head_budget_optimization/run.py.
#
#   MODEL=Qwen/Qwen3-4B sbatch scripts/head-budget-optimization.sh
#
# Qwen3-30B-A3B needs about 192 GB host RAM and 36 h per article:
#
#   MODEL=Qwen/Qwen3-30B-A3B sbatch --cpus-per-gpu=8 --mem-per-cpu=24G \
#       --time=50:00:00 scripts/head-budget-optimization.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-compaction}"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
ARTICLE_IDX=$((90 + SLURM_ARRAY_TASK_ID))
start_time=$(date +%s)

python -u -m head_budget_optimization.run \
  --target-ratio 0.05 \
  --model-name "$MODEL" \
  --dataset-name "quality" \
  --n-articles 1 \
  --start-article "$ARTICLE_IDX" \
  --n-eval-points 51 \
  --max-ratio 0.5 \
  --solve-ratios 0.01,0.02,0.05,0.1 \
  --algorithm-config best \
  --method highest_attn_keys_rms_nnls2_-3_3_lsq \
  --query-config ss-plus-repeat \
  --skip-solve

elapsed=$(( $(date +%s) - start_time ))
echo "Total time: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m $((elapsed % 60))s"
