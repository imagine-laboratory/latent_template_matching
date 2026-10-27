#!/bin/bash
# Held-out comparison against YOLOv8 and Grounding DINO over 1000 paired
# bootstrap trials on the 738-image test set.
#
#   sbatch slurms/run_baselines.slurm
#
# Takes the baseline plan of step 4, not the partition of step 1: it carries
# the bootstrap indices and the fixed template bank.
#
# The original campaign split the 1000 trials across three jobs with
# --resume_from / --stop_at to fit the queue's wall-clock limit; the same run is
# reproduced in one job here.
#
#SBATCH --job-name=baselines
#SBATCH --output=slurm_baselines_%j.txt
#SBATCH --error=slurm_baselines_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00

set -euo pipefail

OUT=${1:-output/baselines}

module load cuda/12.4.0
source "${VAE_ENV:-$HOME/envs/vae_env}/bin/activate"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

python -u run_baseline_comparison.py \
    --plan       "${BASELINE_PLAN:-output/plan_baselines/baseline_plan.json}" \
    --images_dir "${IMAGES_DIR:-data/FULL_UNIFIED}" \
    --csv        "${LABELS_CSV:-data/labels_contents.csv}" \
    --checkpoint "${CHECKPOINT:-checkpoints/best.pt}" \
    --output_dir "$OUT" \
    --device cuda
