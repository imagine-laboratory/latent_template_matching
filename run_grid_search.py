#!/usr/bin/env python
"""
run_grid_search.py  –  Automated grid search over VAE latent template matching parameters.

Delegates all work to the existing scripts via subprocess:
  1) main_bigmamma2.py extract  → latent_template_extract.py
  2) main_bigmamma2.py count    → count_from_latent.py

Requires a plan JSON produced by prepare_experiment.py (--plan).
The plan JSON defines:
  - template_set / eval_set  (stratified image splits)
  - templates_per_replica    (pre-selected, quality-filtered crop lists per replica)
  - run order                (optionally from run_order.txt; re-generated if absent)

Usage:
  # Step 1 — prepare once:
  python prepare_experiment.py \\
      --csv         ./extra/labels_contents.csv \\
      --images_dir  ./extra/FULL_UNIFIED \\
      --output_dir  ./extra/output/grid_search \\
      --replicas    3 --seed 42

  # Step 2 — run the grid (uses the pre-built plan):
  python run_grid_search.py \\
      --plan        ./extra/output/grid_search/template_eval_folds.json \\
      --images_dir  ./extra/FULL_UNIFIED \\
      --csv         ./extra/labels_contents.csv \\
      --checkpoint  ./extra/best.pt \\
      --base_output ./extra/output/grid_search

Skip rules applied to avoid redundant experiments:
  - mode=single   + max_crops > 1  → skipped (single always uses 1st crop)
  - mode=average  + max_crops = 1  → skipped (average of 1 ≡ single)
  - mode=majority_voting + max_crops = 1 → skipped (vote of 1 ≡ single)
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
from itertools import product

# ============================================================
# Grid values  (keep in sync with prepare_experiment.py)
# ============================================================
# Fallback grid, used ONLY when run_order.txt is missing next to the plan.
# It must match prepare_experiment.py exactly: otherwise a missing run_order.txt
# would silently launch a different experiment (the old values here expanded to
# 6,480 configurations instead of 360).
FEATURE_SOURCES     = ["feats"]
MAX_CROPS_LIST      = [5, 30, 60, 90]
ROI_SIZES           = [9, 11, 13]
RESIZE_IMGS         = [256]
MODES               = ["average", "majority_voting"]
THRESHOLDS          = [0.30, 0.37, 0.44, 0.51, 0.58]
POOL_KS_LIST        = [7, 9, 11]
MAJORITY_THRESHOLDS = [0.37]   # only for majority_voting mode


# ============================================================
# Helpers
# ============================================================
def run_cmd(cmd: list, label: str) -> subprocess.CompletedProcess:
    """Run a command, stream its output, and return the CompletedProcess."""
    print(f"\n--- [{label}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def parse_metrics_file(metrics_path: str) -> dict:
    """
    Parse a metrics.txt produced by count_from_latent.py.
    Expected format:  MAE=1.23\nRMSE=4.56\nPrecision=0.75\n...
    Returns dict with keys (or empty strings if not found).
    """
    metrics = {
        "mae": "", "rmse": "", "mape": "", "precision": "", "recall": "",
        "f1": "", "ap": "",
        "p_center": "", "r_center": "", "f1_center": "", "ap_center": "",
        "infer_s": "", "gpu_alloc_mb": "", "gpu_reserved_mb": "", "host_peak_mb": "",
    }
    if not os.path.isfile(metrics_path):
        return metrics
    with open(metrics_path, "r") as f:
        content = f.read()
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
            metrics[key] = m.group(1)
    return metrics


def load_plan(plan_path: str) -> dict:
    with open(plan_path) as f:
        return json.load(f)


def load_run_order(run_order_path: str) -> list[dict] | None:
    """
    Parse run_order.txt (tab-separated key=value lines).
    Returns list of dicts or None if file doesn't exist.
    """
    if not os.path.isfile(run_order_path):
        return None
    runs = []
    with open(run_order_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            # parts[0] is the global index (we re-index below)
            run = {}
            for part in parts[1:]:
                k, _, v = part.partition("=")
                run[k.strip()] = v.strip()
            # Cast types
            run["replica"]    = int(run["replica"])
            run["resize_img"] = int(run.get("resize_img", 256))
            run["max_crops"]  = int(run["max_crops"])
            run["roi_size"]   = int(run["roi_size"])
            run["thresh"]     = float(run["thresh"])
            run["pool_ks"]    = int(run["pool_ks"])
            run["maj_th"]     = float(run["maj_th"]) if run.get("maj_th", "") != "" else None
            runs.append(run)
    return runs


def build_run_order_from_grid(replicas: int, seed: int) -> list[dict]:
    """
    Fallback: regenerate a randomised run order from the grid definition.
    Used when run_order.txt is not found alongside the plan.
    """
    rng = random.Random(seed + 9999)
    runs = []
    for (replica, feature_source, resize_img, max_crops,
         roi_size, pool_ks, mode, thresh) in product(
            range(replicas),
            FEATURE_SOURCES, RESIZE_IMGS, MAX_CROPS_LIST,
            ROI_SIZES, POOL_KS_LIST, MODES, THRESHOLDS,
    ):
        if max_crops == 1 and mode in ("average", "majority_voting"):
            continue
        if max_crops > 1 and mode == "single":
            continue
        maj_thresholds = MAJORITY_THRESHOLDS if mode == "majority_voting" else [None]
        for maj_th in maj_thresholds:
            runs.append(dict(
                replica=replica,
                feature_source=feature_source,
                resize_img=resize_img,
                max_crops=max_crops,
                roi_size=roi_size,
                pool_ks=pool_ks,
                mode=mode,
                thresh=thresh,
                maj_th=maj_th,
            ))
    rng.shuffle(runs)
    return runs


def write_template_csv(templates: list[dict], path: str) -> None:
    """
    Write a temporary CSV with the pre-selected templates for this replica
    in the same format as labels_contents.csv so the extract script can consume it.
    Columns: filename, class, cx, cy, w, h
    """
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["filename", "class", "cx", "cy", "w", "h"])
        writer.writeheader()
        for t in templates:
            writer.writerow({
                "filename": t["stem"] + ".txt",
                "class": 0,
                "cx": t["cx"],
                "cy": t["cy"],
                "w":  t["w"],
                "h":  t["h"],
            })


def write_eval_csv(eval_stems: list[str], full_csv_path: str, output_path: str) -> None:
    """
    Filter the full GT CSV to keep only rows belonging to eval_set images,
    and write a new CSV that the count step will use for evaluation.
    """
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
        description="Grid search over latent template matching parameters (plan-driven).")
    ap.add_argument("--plan",         required=True,
                    help="Path to template_eval_folds.json (output of prepare_experiment.py).")
    ap.add_argument("--images_dir",   default="./extra/FULL_UNIFIED")
    ap.add_argument("--csv",          default="./extra/labels_contents.csv",
                    help="Full GT CSV (used to build eval subset CSV).")
    ap.add_argument("--checkpoint",   default="./extra/best.pt")
    ap.add_argument("--base_output",  default="./extra/output/grid_search")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--limit",        type=int, default=0,
                    help="Max images for batch evaluation (0 = all).")
    ap.add_argument("--python",       default=sys.executable,
                    help="Python interpreter to use for subprocesses.")
    ap.add_argument("--resume_from", type=int, default=-1,
                    help="Resume at this run index. -1 (default) resumes automatically "
                         "from the last row already present in results.csv; 0 starts over.")
    ap.add_argument("--stop_at",     type=int, default=0,
                    help="Stop before this run index (0 = run to the end). Used to split "
                         "the grid across several jobs when the walltime is too short.")
    ap.add_argument("--no_save_images", action="store_true",
                    help="Skip saving overlay images in count runs (faster).")
    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.base_output, exist_ok=True)

    # Resolve absolute paths
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
    plan = load_plan(args.plan)
    replicas          = plan["replicas"]
    eval_stems        = plan["eval_set"]
    templates_per_rep = plan["templates_per_replica"]   # list[list[dict]]
    plan_seed         = plan.get("seed", 42)

    print(f"Plan loaded: {replicas} replicas, "
          f"{len(plan['template_set'])} template images, "
          f"{len(eval_stems)} eval images.")

    # ----------------------------------------------------------
    # Load or regenerate run order
    # ----------------------------------------------------------
    run_order_path = os.path.join(os.path.dirname(os.path.abspath(args.plan)), "run_order.txt")
    runs = load_run_order(run_order_path)
    if runs is None:
        print("run_order.txt not found alongside plan — regenerating from grid.")
        runs = build_run_order_from_grid(replicas, plan_seed)
    else:
        print(f"Run order loaded: {len(runs)} runs from {run_order_path}")

    # ----------------------------------------------------------
    # Build eval CSV once (shared across all runs)
    # ----------------------------------------------------------
    eval_csv_path = os.path.join(base_output, "eval_set.csv")
    if not os.path.isfile(eval_csv_path):
        print(f"Building eval CSV → {eval_csv_path}")
        write_eval_csv(eval_stems, csv_path, eval_csv_path)

    # ----------------------------------------------------------
    # Open consolidated results CSV
    # ----------------------------------------------------------
    results_csv = os.path.join(base_output, "results.csv")
    csv_fields = [
        "run_index", "replica",
        "feature_source", "max_crops", "roi_size", "resize_img",
        "mode", "thresh", "pool_ks", "majority_threshold",
        "mae", "rmse", "mape",
        "precision", "recall", "f1", "ap",
        "p_center", "r_center", "f1_center", "ap_center",
        "elapsed_s", "infer_s",
        "gpu_alloc_mb", "gpu_reserved_mb", "host_peak_mb",
        "status",
    ]
    # Resume support.  The CSV is flushed after every row, so the last row it
    # holds is the last run that actually completed: restarting after a walltime
    # kill, a node failure or an OOM costs only the run that was in flight.
    resume_from = args.resume_from
    if resume_from < 0:
        resume_from = 0
        if os.path.isfile(results_csv):
            with open(results_csv, newline="") as fh:
                done = [r for r in csv_mod.DictReader(fh) if r.get("run_index", "").isdigit()]
            if done:
                resume_from = max(int(r["run_index"]) for r in done) + 1
                print(f"Found {len(done)} rows in {results_csv}; resuming at run {resume_from}.")

    append = resume_from > 0 and os.path.isfile(results_csv)
    csv_fh = open(results_csv, "a" if append else "w", newline="")
    writer = csv_mod.DictWriter(csv_fh, fieldnames=csv_fields)
    if not append:
        writer.writeheader()

    # ----------------------------------------------------------
    # Cache: avoid re-extracting templates for same (replica, ext params)
    # extract_cache key: (replica, feature_source, roi_size, resize_img, max_crops)
    # ----------------------------------------------------------
    extract_cache = {}

    t0_global = time.time()

    # ----------------------------------------------------------
    # Execute runs in the pre-defined randomised order
    # ----------------------------------------------------------
    for run_index, run in enumerate(runs):
        if run_index < resume_from:
            continue
        if args.stop_at > 0 and run_index >= args.stop_at:
            print(f"  reached --stop_at={args.stop_at}, stopping here.")
            break
        replica        = run["replica"]
        feature_source = run["feature_source"]
        resize_img     = run["resize_img"]
        max_crops      = run["max_crops"]
        roi_size       = run["roi_size"]
        pool_ks        = run["pool_ks"]
        mode           = run["mode"]
        thresh         = run["thresh"]
        maj_th         = run.get("maj_th")  # float or None

        mode_short = {"majority_voting": "majvote", "average": "avg", "single": "single"}[mode]
        templates  = templates_per_rep[replica]

        # --- Write per-replica template CSV (cached on disk) ---
        ext_key = (replica, feature_source, roi_size, resize_img, max_crops)
        ext_tag = f"{feature_source}_r{resize_img}_c{max_crops}_roi{roi_size}"
        extract_dir = os.path.join(base_output, str(replica), "templates", ext_tag)

        if ext_key not in extract_cache:
            os.makedirs(extract_dir, exist_ok=True)

            # Write the template CSV for this replica
            tmpl_csv = os.path.join(extract_dir, "templates.csv")
            # Trim to max_crops for this run (templates list has max(MAX_CROPS_LIST) entries)
            write_template_csv(templates[:max_crops], tmpl_csv)

            extract_cmd = [
                python, main_script, "extract",
                "--images_dir", images_dir,
                "--csv",        tmpl_csv,          # <-- replica-specific template CSV
                "--checkpoint", checkpoint,
                "--output_dir", extract_dir,
                "--max_crops",  str(max_crops),
                "--roi_size",   str(roi_size),
                "--resize_img", str(resize_img),
                "--feature_source", feature_source,
                "--device", args.device,
                "--seed",   str(plan_seed),
            ]
            result = run_cmd(extract_cmd, f"extract {ext_tag}")
            extract_cache[ext_key] = extract_dir
            if result.returncode != 0:
                print(f"  WARN: extraction failed for {ext_tag}")

        extract_dir      = extract_cache[ext_key]
        proto_pt         = os.path.join(extract_dir, "latent_prototype.pt")
        template_bank_pt = os.path.join(extract_dir, "template_bank.pt")

        # Select the correct .pt file
        if mode == "majority_voting":
            if not os.path.isfile(template_bank_pt):
                print(f"  SKIP [{run_index}]: template_bank.pt missing for {ext_tag}")
                continue
            pt_file = template_bank_pt
        else:
            if not os.path.isfile(proto_pt):
                print(f"  SKIP [{run_index}]: latent_prototype.pt missing for {ext_tag}")
                continue
            pt_file = proto_pt

        # --- Count run ---
        maj_suffix = f"_mt{maj_th}" if maj_th is not None else ""
        count_tag  = (f"{mode_short}/{feature_source}"
                      f"_r{resize_img}_c{max_crops}_roi{roi_size}_pk{pool_ks}"
                      f"/th{thresh}{maj_suffix}")
        count_dir  = os.path.join(base_output, str(replica), "counts", count_tag)

        count_cmd = [
            python, main_script, "count",
            "--images_dir",   images_dir,
            "--prototype_pt", pt_file,
            "--checkpoint",   checkpoint,
            "--output_dir",   count_dir,
            "--resize_img",   str(resize_img),
            "--thresh",       str(thresh),
            "--pool_ks",      str(pool_ks),
            "--roi_size",     str(roi_size),
            "--csv",          eval_csv_path,   # <-- eval subset only
            "--feature_source", feature_source,
            "--mode",         mode,
            "--device",       args.device,
            "--seed",         str(plan_seed),
            "--only_annotated",
            "--limit",        str(args.limit) if args.limit > 0 else "0",
        ]

        if args.no_save_images:
            count_cmd.append("--no_save_images")

        if maj_th is not None:
            count_cmd += ["--majority_threshold", str(maj_th)]

        t0      = time.time()
        result  = run_cmd(count_cmd, f"count [{run_index}] {count_tag}")
        elapsed = round(time.time() - t0, 2)

        status  = "ok" if result.returncode == 0 else f"error (rc={result.returncode})"
        metrics = parse_metrics_file(os.path.join(count_dir, "metrics.txt"))

        row = dict(
            run_index=run_index,
            replica=replica,
            feature_source=feature_source,
            max_crops=max_crops,
            roi_size=roi_size,
            resize_img=resize_img,
            mode=mode,
            thresh=thresh,
            pool_ks=pool_ks,
            majority_threshold=maj_th if maj_th is not None else "",
            elapsed_s=elapsed,
            status=status,
            **metrics,
        )
        writer.writerow(row)
        csv_fh.flush()

        print(f"  [{run_index}] rep={replica} {count_tag}: "
              f"MAE={metrics.get('mae', '?')} "
              f"RMSE={metrics.get('rmse', '?')} "
              f"P={metrics.get('precision', '?')} "
              f"R={metrics.get('recall', '?')} "
              f"F1={metrics.get('f1', '?')} "
              f"AP={metrics.get('ap', '?')} "
              f"({elapsed}s) [{status}]")

    csv_fh.close()
    total_time = time.time() - t0_global
    print(f"\n{'=' * 70}")
    print(f"Grid search complete. {len(runs)} runs in {total_time:.0f}s")
    print(f"Results saved to: {results_csv}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()