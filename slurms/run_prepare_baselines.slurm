#!/bin/bash
# Step 4. Builds the plan of the held-out comparison: the single bank of 90
# templates at the selected configuration and the 1,000 bootstrap resamples of
# the test set. Writes baseline_plan.json, which run_baselines.slurm consumes.
#
#   sbatch slurms/run_prepare_baselines.slurm
#
#SBATCH --job-name=prep_base
#SBATCH --output=slurm_prep_base_%j.txt
#SBATCH --error=slurm_prep_base_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

OUT=${1:-output/plan_baselines}

module load cuda/12.4.0
source "${VAE_ENV:-$HOME/envs/vae_env}/bin/activate"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

python -u prepare_baseline_experiment.py \
    --plan       "${PLAN:-output/plan/template_eval_folds.json}" \
    --csv        "${LABELS_CSV:-data/labels_contents.csv}" \
    --images_dir "${IMAGES_DIR:-data/FULL_UNIFIED}" \
    --output_dir "$OUT" \
    --n_templates 90 --n_bootstrap 1000 --seed 42 \
    --feature_source feats --roi_size 9 --pool_ks 9 --thresh 0.58 --mode average \
    --gdino_conf 0.1 --yolo_conf 0.1
