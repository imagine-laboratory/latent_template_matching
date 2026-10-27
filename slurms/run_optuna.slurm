#!/bin/bash
# Multi-objective Bayesian search over the matching hyperparameters, scored on
# the evaluation set. Takes the number of trials and an output directory:
#
#   sbatch slurms/run_optuna.slurm 1000 output/optuna
#
#SBATCH --job-name=optuna
#SBATCH --output=slurm_optuna_%j.txt
#SBATCH --error=slurm_optuna_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

set -euo pipefail

N_TRIALS=${1:-1000}
OUT=${2:-output/optuna}

module load cuda/12.4.0
source "${VAE_ENV:-$HOME/envs/vae_env}/bin/activate"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

python -u run_optuna.py \
    --plan       "${PLAN:-output/plan/template_eval_folds.json}" \
    --images_dir "${IMAGES_DIR:-data/FULL_UNIFIED}" \
    --csv        "${LABELS_CSV:-data/labels_contents.csv}" \
    --checkpoint "${CHECKPOINT:-checkpoints/best.pt}" \
    --base_output "$OUT" \
    --n_trials "$N_TRIALS" \
    --device cuda \
    --no_save_images
