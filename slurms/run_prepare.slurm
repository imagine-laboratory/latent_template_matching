#!/bin/bash
# Step 1. Builds the fixed partition every later stage depends on: the
# template / evaluation / test split, stratified by fruit-count quintile, and
# the five disjoint template replicas. Writes template_eval_folds.json.
#
#   sbatch slurms/run_prepare.slurm
#
#SBATCH --job-name=prepare
#SBATCH --output=slurm_prepare_%j.txt
#SBATCH --error=slurm_prepare_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

OUT=${1:-output/plan}

module load cuda/12.4.0
source "${VAE_ENV:-$HOME/envs/vae_env}/bin/activate"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

python -u prepare_experiment.py \
    --csv        "${LABELS_CSV:-data/labels_contents.csv}" \
    --images_dir "${IMAGES_DIR:-data/FULL_UNIFIED}" \
    --output_dir "$OUT" \
    --replicas 5 --seed 42
