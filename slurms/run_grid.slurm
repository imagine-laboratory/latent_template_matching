#!/bin/bash
# Full factorial experiment: 360 configurations on each of the 5 template
# replicas, for 1800 runs on the evaluation set.
#
#   sbatch slurms/run_grid.slurm
#
#SBATCH --job-name=grid
#SBATCH --output=slurm_grid_%j.txt
#SBATCH --error=slurm_grid_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

set -euo pipefail

OUT=${1:-output/grid}

module load cuda/12.4.0
source "${VAE_ENV:-$HOME/envs/vae_env}/bin/activate"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

python -u run_grid_search.py \
    --plan       "${PLAN:-output/plan/template_eval_folds.json}" \
    --images_dir "${IMAGES_DIR:-data/FULL_UNIFIED}" \
    --csv        "${LABELS_CSV:-data/labels_contents.csv}" \
    --checkpoint "${CHECKPOINT:-checkpoints/best.pt}" \
    --base_output "$OUT" \
    --no_save_images
