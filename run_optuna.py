#!/usr/bin/env python
"""
run_optuna.py  –  Multi-objective Bayesian search over VAE matching parameters.

Uses Optuna + NSGA-II to find the Pareto front of (maximize F1_center, minimize MAE),
guided by the same plan JSON produced by prepare_experiment.py.

Hyperparameters searched
  thresh             float  [0.20, 0.60]
  majority_threshold float  [0.20, 0.70]   (only used when mode=majority_voting)
  max_crops          int    {30, 60, 90}
  roi_size           int    {9, 11, 13, 15}
  pool_ks            int    {5, 7, 9, 11}
  mode               cat    {average}
  feature_source     cat    {feats}

The replica is drawn uniformly at random for each trial so results generalise
across template splits.  Failed trials are silently discarded.

Usage:
  python run_optuna.py \\
      --plan        ./extra/output/grid_search/template_eval_folds.json \\
      --images_dir  ./extra/FULL_UNIFIED \\
      --csv         ./extra/labels_contents.csv \\
      --checkpoint  ./extra/best.pt \\
      --base_output ./extra/output/optuna \\
      --n_trials    300 \\
      --device      cuda

After the search, results.csv holds every trial and pareto_front.csv
holds the non-dominated solutions.
"""

import os
import sys
import json
import subprocess
import time
import argparse
import csv as csv_mod
import re
import random

import optuna
from optuna.samplers import NSGAIISampler

# ============================================================
# Helpers (shared with run_grid_search.py logic)
# ============================================================

def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_metrics_file(metrics_path: str) -> dict:
    metrics = {
        "mae": None, "rmse": None, "mape": None,
        "precision": None, "recall": None, "f1": None, "ap": None,
        "p_center": None, "r_center": None, "f1_center": None, "ap_center": None,
        "infer_s": None, "gpu_alloc_mb": None, "gpu_reserved_mb": None, "host_peak_mb": None,
    }
    if not os.path.isfile(metrics_path):
        return metrics
    content = open(metrics_path).read()
    for key, pattern in [
        ("mae",       r"MAE=([\d.]+)"),
        ("rmse",      r"RMSE=([\d.]+)"),
        ("mape",      r"MAPE=([\d.]+)"),
        ("precision", r"Precision=([\d.]+)"),
        ("recall",    r"Recall=([\d.]+)"),
        ("f1",        r"F1=([\d.]+)"),
        ("ap",        r"AP=([\d.]+)"),
        ("p_center",  r"P_center=([\d.]+)"),
        ("r_center",  r"R_center=([\d.]+)"),
        ("f1_center", r"F1_center=([\d.]+)"),
        ("ap_center", r"AP_center=([\d.]+)"),
        ("infer_s",         r"Infer_s=([\d.]+)"),
        ("gpu_alloc_mb",    r"GPU_alloc_MB=([\d.]+)"),
        ("gpu_reserved_mb", r"GPU_reserved_MB=([\d.]+)"),
        ("host_peak_mb",    r"Host_peak_MB=([\d.]+)"),
    ]:
        m = re.search(pattern, content)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def write_template_csv(templates: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["filename", "class", "cx", "cy", "w", "h"])
        writer.writeheader()
        for t in templates:
            writer.writerow({
                "filename": t["stem"] + ".txt",
                "class": 0,
                "cx": t["cx"], "cy": t["cy"],
                "w": t["w"],   "h": t["h"],
            })


def write_eval_csv(eval_stems: list, full_csv_path: str, output_path: str) -> None:
    eval_set = set(eval_stems)
    with open(full_csv_path, newline="") as fin, \
         open(output_path, "w", newline="") as fout:
        reader = csv_mod.DictReader(fin)
        writer = csv_mod.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            stem = os.path.splitext(row["filename"])[0]
            if stem in eval_set:
                writer.writerow(row)


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Multi-objective Optuna search over latent template matching parameters.")
    ap.add_argument("--plan",         required=True,
                    help="Path to template_eval_folds.json (output of prepare_experiment.py).")
    ap.add_argument("--images_dir",   default="./extra/FULL_UNIFIED")
    ap.add_argument("--csv",          default="./extra/labels_contents.csv")
    ap.add_argument("--checkpoint",   default="./extra/best.pt")
    ap.add_argument("--base_output",  default="./extra/output/optuna")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--n_trials",     type=int, default=300,
                    help="Total number of Optuna trials (default: 300).")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--limit",        type=int, default=0,
                    help="Max eval images per trial (0 = all).")
    ap.add_argument("--python",       default=sys.executable)
    ap.add_argument("--no_save_images", action="store_true",
                    help="Skip saving overlay images (faster).")
    ap.add_argument("--resize_img",   type=int, default=256)
    ap.add_argument("--study_name",   default="vae_pareto",
                    help="Optuna study name (used for SQLite persistence).")
    return ap.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.base_output, exist_ok=True)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "main_bigmamma2.py")
    images_dir  = os.path.abspath(args.images_dir)
    csv_path    = os.path.abspath(args.csv)
    checkpoint  = os.path.abspath(args.checkpoint)
    base_output = os.path.abspath(args.base_output)
    python      = args.python

    # ----------------------------------------------------------
    # Load plan
    # ----------------------------------------------------------
    with open(args.plan) as f:
        plan = json.load(f)

    replicas          = plan["replicas"]
    eval_stems        = plan["eval_set"]
    templates_per_rep = plan["templates_per_replica"]
    plan_seed         = plan.get("seed", 42)

    print(f"Plan: {replicas} replicas, "
          f"{len(plan['template_set'])} template imgs, "
          f"{len(eval_stems)} eval imgs.")

    # ----------------------------------------------------------
    # Build eval CSV once (shared across all trials)
    # ----------------------------------------------------------
    eval_csv_path = os.path.join(base_output, "eval_set.csv")
    if not os.path.isfile(eval_csv_path):
        print(f"Building eval CSV → {eval_csv_path}")
        write_eval_csv(eval_stems, csv_path, eval_csv_path)

    # ----------------------------------------------------------
    # Results CSV (written after every trial)
    # ----------------------------------------------------------
    results_csv = os.path.join(base_output, "results.csv")
    csv_fields = [
        "trial", "replica",
        "feature_source", "mode", "max_crops", "roi_size",
        "resize_img", "pool_ks", "thresh", "majority_threshold",
        "mae", "rmse", "mape",
        "precision", "recall", "f1", "ap",
        "p_center", "r_center", "f1_center", "ap_center",
        "elapsed_s", "infer_s", "gpu_alloc_mb", "gpu_reserved_mb", "host_peak_mb",
        "status",
    ]
    results_fh  = open(results_csv, "w", newline="")
    results_writer = csv_mod.DictWriter(results_fh, fieldnames=csv_fields)
    results_writer.writeheader()

    # ----------------------------------------------------------
    # Extract cache: avoid re-running extraction for same params
    # key: (replica, feature_source, roi_size, resize_img, max_crops)
    # ----------------------------------------------------------
    extract_cache = {}
    rng = random.Random(args.seed)

    # ----------------------------------------------------------
    # Optuna objective
    # ----------------------------------------------------------
    def objective(trial: optuna.Trial):
        # Sample hyperparameters
        thresh         = trial.suggest_float("thresh", 0.20, 0.70)
        max_crops      = trial.suggest_categorical("max_crops", [5, 30, 60, 90])
        roi_size       = trial.suggest_categorical("roi_size", [7, 9, 11])
        pool_ks        = trial.suggest_categorical("pool_ks", [3, 5, 7])
        mode           = trial.suggest_categorical("mode",
                             ["majority_voting", "average"])
        feature_source = trial.suggest_categorical("feature_source",
                             ["feats", "z", "mu"])

        # majority_threshold is only meaningful for majority_voting
        if mode == "majority_voting":
            maj_th = trial.suggest_float("majority_threshold", 0.20, 0.70)
        else:
            maj_th = None

        # Skip rules (mirror run_grid_search.py)
        if mode in ("average", "majority_voting") and max_crops == 1:
            raise optuna.exceptions.TrialPruned()
        if mode == "single" and max_crops > 1:
            raise optuna.exceptions.TrialPruned()

        # Pick a replica at random for this trial
        replica   = rng.randint(0, replicas - 1)
        templates = templates_per_rep[replica]

        # --- Extract (cached) ---
        resize_img = args.resize_img
        ext_key    = (replica, feature_source, roi_size, resize_img, max_crops)
        ext_tag    = f"{feature_source}_r{resize_img}_c{max_crops}_roi{roi_size}"
        extract_dir = os.path.join(base_output, str(replica), "templates", ext_tag)

        if ext_key not in extract_cache:
            os.makedirs(extract_dir, exist_ok=True)
            tmpl_csv = os.path.join(extract_dir, "templates.csv")
            write_template_csv(templates[:max_crops], tmpl_csv)

            extract_cmd = [
                python, main_script, "extract",
                "--images_dir",      images_dir,
                "--csv",             tmpl_csv,
                "--checkpoint",      checkpoint,
                "--output_dir",      extract_dir,
                "--max_crops",       str(max_crops),
                "--roi_size",        str(roi_size),
                "--resize_img",      str(resize_img),
                "--feature_source",  feature_source,
                "--device",          args.device,
                "--seed",            str(plan_seed),
            ]
            res = run_cmd(extract_cmd)
            if res.returncode != 0:
                print(f"  [trial {trial.number}] extract FAILED for {ext_tag}")
                if res.stderr:
                    print(res.stderr[:500])
                raise optuna.exceptions.TrialPruned()
            extract_cache[ext_key] = extract_dir
        else:
            extract_dir = extract_cache[ext_key]

        proto_pt         = os.path.join(extract_dir, "latent_prototype.pt")
        template_bank_pt = os.path.join(extract_dir, "template_bank.pt")

        if mode == "majority_voting":
            pt_file = template_bank_pt
        else:
            pt_file = proto_pt

        if not os.path.isfile(pt_file):
            print(f"  [trial {trial.number}] .pt file missing: {pt_file}")
            raise optuna.exceptions.TrialPruned()

        # --- Count run ---
        mode_short = {"majority_voting": "majvote", "average": "avg", "single": "single"}[mode]
        maj_suffix = f"_mt{maj_th:.3f}" if maj_th is not None else ""
        count_tag  = (f"{mode_short}/{feature_source}"
                      f"_r{resize_img}_c{max_crops}_roi{roi_size}_pk{pool_ks}"
                      f"/th{thresh:.3f}{maj_suffix}")
        count_dir  = os.path.join(base_output, str(replica), "counts",
                                  f"t{trial.number}_{count_tag}")

        count_cmd = [
            python, main_script, "count",
            "--images_dir",    images_dir,
            "--prototype_pt",  pt_file,
            "--checkpoint",    checkpoint,
            "--output_dir",    count_dir,
            "--resize_img",    str(resize_img),
            "--thresh",        str(thresh),
            "--pool_ks",       str(pool_ks),
            "--roi_size",      str(roi_size),
            "--csv",           eval_csv_path,
            "--feature_source", feature_source,
            "--mode",          mode,
            "--device",        args.device,
            "--seed",          str(plan_seed),
            "--only_annotated",
            "--limit",         str(args.limit) if args.limit > 0 else "0",
        ]
        if args.no_save_images:
            count_cmd.append("--no_save_images")
        if maj_th is not None:
            count_cmd += ["--majority_threshold", str(maj_th)]

        t0      = time.time()
        res     = run_cmd(count_cmd)
        elapsed = round(time.time() - t0, 2)

        status  = "ok" if res.returncode == 0 else f"error (rc={res.returncode})"
        metrics = parse_metrics_file(os.path.join(count_dir, "metrics.txt"))

        f1_center = metrics.get("f1_center")
        mae       = metrics.get("mae")

        # Log to CSV regardless of outcome
        row = dict(
            trial=trial.number,
            replica=replica,
            feature_source=feature_source,
            mode=mode,
            max_crops=max_crops,
            roi_size=roi_size,
            resize_img=resize_img,
            pool_ks=pool_ks,
            thresh=round(thresh, 4),
            majority_threshold=round(maj_th, 4) if maj_th is not None else "",
            elapsed_s=elapsed,
            status=status,
            **{k: (v if v is not None else "") for k, v in metrics.items()},
        )
        results_writer.writerow(row)
        results_fh.flush()

        print(f"  trial={trial.number:4d} rep={replica} {mode_short} "
              f"fs={feature_source} c={max_crops} roi={roi_size} "
              f"pk={pool_ks} th={thresh:.3f}"
              + (f" mt={maj_th:.3f}" if maj_th is not None else "")
              + f" → MAE={mae} F1c={f1_center}  ({elapsed}s) [{status}]")

        if f1_center is None or mae is None:
            print(f"    WARN: metrics missing, pruning trial.")
            raise optuna.exceptions.TrialPruned()

        # Objectives: (maximize F1_center, minimize MAE)
        return f1_center, mae

    # ----------------------------------------------------------
    # Create study (persisted to SQLite so it survives crashes)
    # ----------------------------------------------------------
    storage_path = os.path.join(base_output, f"{args.study_name}.db")
    storage      = f"sqlite:///{storage_path}"

    sampler = NSGAIISampler(seed=args.seed)
    study   = optuna.create_study(
        study_name    = args.study_name,
        storage       = storage,
        directions    = ["maximize", "minimize"],   # F1_center ↑, MAE ↓
        sampler       = sampler,
        load_if_exists= True,
    )

    print(f"\nStarting Optuna search: {args.n_trials} trials")
    print(f"Study persisted at: {storage_path}")
    print(f"Results CSV: {results_csv}\n")

    t0_global = time.time()

    try:
        study.optimize(
            objective,
            n_trials  = args.n_trials,
            catch     = (Exception,),   # keep going on any failure
        )
    finally:
        results_fh.close()

    total_time = time.time() - t0_global

    # ----------------------------------------------------------
    # Save Pareto front
    # ----------------------------------------------------------
    pareto_trials = study.best_trials
    pareto_csv    = os.path.join(base_output, "pareto_front.csv")

    pareto_fields = [
        "trial", "f1_center", "mae",
        "feature_source", "mode", "max_crops", "roi_size",
        "pool_ks", "thresh", "majority_threshold",
    ]
    with open(pareto_csv, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=pareto_fields)
        w.writeheader()
        for t in sorted(pareto_trials, key=lambda t: -t.values[0]):
            p = t.params
            w.writerow({
                "trial":              t.number,
                "f1_center":          round(t.values[0], 4),
                "mae":                round(t.values[1], 4),
                "feature_source":     p.get("feature_source", ""),
                "mode":               p.get("mode", ""),
                "max_crops":          p.get("max_crops", ""),
                "roi_size":           p.get("roi_size", ""),
                "pool_ks":            p.get("pool_ks", ""),
                "thresh":             round(p.get("thresh", 0), 4),
                "majority_threshold": round(p.get("majority_threshold", 0), 4)
                                      if "majority_threshold" in p else "",
            })

    print(f"\n{'=' * 70}")
    print(f"Search complete.  {len(study.trials)} trials in {total_time:.0f}s")
    print(f"Pareto front: {len(pareto_trials)} solutions → {pareto_csv}")
    print(f"{'=' * 70}")

    print("\nPareto front (sorted by F1_center desc):")
    for t in sorted(pareto_trials, key=lambda t: -t.values[0]):
        p = t.params
        mt_str = f" mt={p['majority_threshold']:.3f}" if "majority_threshold" in p else ""
        print(f"  trial={t.number:4d}  F1c={t.values[0]:.4f}  MAE={t.values[1]:.4f}"
              f"  {p.get('mode','?')} fs={p.get('feature_source','?')}"
              f" c={p.get('max_crops','?')} roi={p.get('roi_size','?')}"
              f" pk={p.get('pool_ks','?')} th={p.get('thresh',0):.3f}{mt_str}")


if __name__ == "__main__":
    main()
