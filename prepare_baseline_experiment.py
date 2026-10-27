#!/usr/bin/env python
"""
prepare_baseline_experiment.py — Prepare a reproducible bootstrap-based baseline comparison.

Experimental protocol (from Section 4.4 of the thesis):
  - Uses the held-out test_set (738 images) exclusively.
  - B=1000 bootstrap trials: each trial samples test images WITH replacement.
  - Each bootstrapped subset is shared across all 3 models within a trial.
  - N templates are selected from the ENTIRE template_set (no folds).
  - Execution order of 3×B = 3000 evaluations is fully randomized.
  - Outputs order_baselines.txt (execution order) and a plan JSON.

Usage:
  python prepare_baseline_experiment.py \
      --plan ./extra/output/grid_search/template_eval_folds.json \
      --output_dir ./extra/output/baselines_bootstrap \
      --n_templates 90 \
      --n_bootstrap 1000 \
      --seed 42

Output files (inside --output_dir):
  baseline_plan.json      — full plan (test_set, templates, bootstrap indices, config)
  order_baselines.txt     — randomized execution order for all 3000 runs
"""

import os
import json
import argparse
import random
import csv as csv_mod
from collections import defaultdict

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False

try:
    from PIL import Image
    import imagehash
    _HAVE_IMG = True
except ImportError:
    _HAVE_IMG = False


# ============================================================
# Default quality filters (same as prepare_experiment.py)
# ============================================================
# The CSV stores YOLO-normalised boxes: w is a fraction of the frame WIDTH and h
# a fraction of the frame HEIGHT.  h/w on the normalised values therefore equals
# the true pixel aspect ratio times H/W (0.5625), so the filter must be applied
# in pixels of the original frame (see prepare_experiment.py).
DEFAULT_FRAME_W = 1920
DEFAULT_FRAME_H = 1080

DEFAULT_MIN_AR = 0.6
DEFAULT_MAX_AR = 1.4
DEFAULT_MIN_AREA_FACTOR = 0.33
DEFAULT_HASH_BITS = 8
DEFAULT_HASH_THRESHOLD = 10
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Models / methods to compare
METHODS = ["proposed", "yolo", "grounding_dino"]


# ============================================================
# Helpers
# ============================================================

def pixel_aspect_ratio(w_norm: float, h_norm: float,
                       frame_w: int = DEFAULT_FRAME_W,
                       frame_h: int = DEFAULT_FRAME_H) -> float:
    """Aspect ratio h/w in pixels of the original frame."""
    if w_norm <= 0:
        return 0.0
    return (h_norm * frame_h) / (w_norm * frame_w)


def find_image_path(images_dir: str, stem: str) -> str | None:
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def load_crop(image_path: str, cx: float, cy: float, w: float, h: float):
    try:
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return None


def perceptual_hash(crop_img, hash_bits: int):
    return imagehash.phash(crop_img, hash_size=hash_bits)


def is_too_similar(new_hash, selected_hashes: list, threshold: int) -> bool:
    for h in selected_hashes:
        if abs(new_hash - h) <= threshold:
            return True
    return False


def select_templates_from_pool(
    candidates: list,
    n_templates: int,
    rng: random.Random,
    images_dir: str | None,
    min_ar: float,
    max_ar: float,
    min_area_factor: float,
    hash_bits: int,
    hash_threshold: int,
    use_images: bool,
) -> list:
    """
    Select up to n_templates from the ENTIRE candidate pool (no folds).
    Quality filters + perceptual deduplication are applied.
    """
    mean_area = (sum(c["area"] for c in candidates) / len(candidates)
                 if candidates else 0.0)
    min_area = min_area_factor * mean_area

    # Quality filter
    filtered = [
        c for c in candidates
        if min_ar <= c["ar"] <= max_ar and c["area"] >= min_area
    ]

    if not filtered:
        print("  WARN: no candidates passed quality filters.")
        return []

    rng.shuffle(filtered)

    # Greedy selection with deduplication
    selected = []
    selected_hashes = []

    for c in filtered:
        if len(selected) >= n_templates:
            break
        if use_images and _HAVE_IMG and images_dir:
            img_path = find_image_path(images_dir, c["stem"])
            if img_path is None:
                continue
            crop_img = load_crop(img_path, c["cx"], c["cy"], c["w"], c["h"])
            if crop_img is None:
                continue
            h = perceptual_hash(crop_img, hash_bits)
            if is_too_similar(h, selected_hashes, hash_threshold):
                continue
            selected_hashes.append(h)
        selected.append(c)

    if len(selected) < n_templates:
        print(f"  WARN: only {len(selected)}/{n_templates} templates selected.")

    return [
        {k: v for k, v in c.items() if k in ("stem", "box_index", "cx", "cy", "w", "h")}
        for c in selected
    ]


def generate_bootstrap_indices(test_stems: list, n_bootstrap: int, rng: random.Random) -> list:
    """
    Generate B bootstrap samples (sampling with replacement from test_stems).
    Each sample is a list of indices into test_stems.
    """
    n = len(test_stems)
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = [rng.randint(0, n - 1) for _ in range(n)]
        bootstrap_samples.append(sample)
    return bootstrap_samples


def build_execution_order(n_bootstrap: int, methods: list, rng: random.Random) -> list:
    """
    Generate all (trial, method) combinations and randomize.
    Returns list of dicts: {trial: int, method: str}
    """
    runs = []
    for trial in range(n_bootstrap):
        for method in methods:
            runs.append({"trial": trial, "method": method})
    rng.shuffle(runs)
    return runs


def write_order_file(runs: list, path: str) -> None:
    """Write order_baselines.txt with the execution order."""
    with open(path, "w") as f:
        f.write(f"# Total runs: {len(runs)}\n")
        f.write(f"# Format: run_index\\ttrial\\tmethod\n")
        for i, r in enumerate(runs):
            f.write(f"{i}\t{r['trial']}\t{r['method']}\n")


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Prepare bootstrap-based baseline comparison experiment.")
    ap.add_argument("--plan", required=True,
                    help="Path to template_eval_folds.json (from prepare_experiment.py). "
                         "Used to load test_set and template_set definitions.")
    ap.add_argument("--csv", default=None,
                    help="Path to labels_contents.csv. If not provided, inferred from plan context.")
    ap.add_argument("--images_dir", default=None,
                    help="Images directory (for perceptual-hash filtering of templates).")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write baseline_plan.json and order_baselines.txt.")
    ap.add_argument("--n_templates", type=int, default=90,
                    help="Number of templates to select from the entire template_set (default: 90).")
    ap.add_argument("--n_bootstrap", type=int, default=1000,
                    help="Number of bootstrap trials (default: 1000).")
    ap.add_argument("--seed", type=int, default=42)
    # Proposed model config (frozen from grid search optimization)
    ap.add_argument("--feature_source", default="feats",
                    help="Feature source for proposed model (default: feats).")
    ap.add_argument("--roi_size", type=int, default=9,
                    help="ROI size for proposed model (default: 9).")
    ap.add_argument("--pool_ks", type=int, default=9,
                    help="Pool kernel size for proposed model (default: 9).")
    ap.add_argument("--thresh", type=float, default=0.44,
                    help="Detection threshold for proposed model (default: 0.44).")
    ap.add_argument("--mode", default="average",
                    help="Aggregation mode for proposed model (default: average).")
    ap.add_argument("--maj_thresh", type=float, default=0.37,
                    help="Majority threshold (only for majority_voting mode, default: 0.37).")
    # YOLO config
    ap.add_argument("--yolo_model", default="yolov8x.pt",
                    help="YOLO model file (default: yolov8x.pt).")
    ap.add_argument("--yolo_conf", type=float, default=0.25,
                    help="YOLO confidence threshold (default: 0.25).")
    # Grounding DINO config
    ap.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-base",
                    help="Grounding DINO model ID (default: IDEA-Research/grounding-dino-base).")
    ap.add_argument("--gdino_conf", type=float, default=0.3,
                    help="Grounding DINO confidence threshold (default: 0.3).")
    ap.add_argument("--gdino_prompt", default="pineapple",
                    help="Text prompt for Grounding DINO (default: pineapple).")
    # Template quality filters
    ap.add_argument("--min_ar", type=float, default=DEFAULT_MIN_AR)
    ap.add_argument("--max_ar", type=float, default=DEFAULT_MAX_AR)
    ap.add_argument("--min_area_factor", type=float, default=DEFAULT_MIN_AREA_FACTOR)
    ap.add_argument("--frame_w", type=int, default=DEFAULT_FRAME_W)
    ap.add_argument("--frame_h", type=int, default=DEFAULT_FRAME_H)
    ap.add_argument("--hash_bits", type=int, default=DEFAULT_HASH_BITS)
    ap.add_argument("--hash_threshold", type=int, default=DEFAULT_HASH_THRESHOLD)
    ap.add_argument("--no_images", action="store_true",
                    help="Skip perceptual-hash deduplication (fast mode).")
    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)
    use_images = (not args.no_images) and (args.images_dir is not None) and _HAVE_IMG

    # ----------------------------------------------------------
    # 1. Load existing plan to get test_set and template_set
    # ----------------------------------------------------------
    print(f"Loading plan from: {args.plan}")
    with open(args.plan) as f:
        original_plan = json.load(f)

    test_stems = original_plan["test_set"]
    template_stems_set = set(original_plan["template_set"])
    print(f"  test_set: {len(test_stems)} images")
    print(f"  template_set: {len(template_stems_set)} images")

    # ----------------------------------------------------------
    # 2. Load CSV and build candidate pool from template_set
    # ----------------------------------------------------------
    csv_path = args.csv
    if csv_path is None:
        # Try to find the CSV relative to the plan
        plan_dir = os.path.dirname(os.path.abspath(args.plan))
        candidate_paths = [
            os.path.join(plan_dir, "..", "..", "labels_contents.csv"),
            os.path.join(plan_dir, "..", "labels_contents.csv"),
            "./extra/labels_contents.csv",
        ]
        for cp in candidate_paths:
            if os.path.isfile(cp):
                csv_path = cp
                break
        if csv_path is None:
            raise FileNotFoundError("Could not locate labels_contents.csv. Use --csv to specify.")

    print(f"Reading CSV: {csv_path}")
    candidates = []
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            stem = os.path.splitext(row["filename"])[0]
            if stem not in template_stems_set:
                continue
            cx, cy, w, h = float(row["cx"]), float(row["cy"]), float(row["w"]), float(row["h"])
            area = w * h
            ar = pixel_aspect_ratio(w, h, args.frame_w, args.frame_h)
            candidates.append({
                "stem": stem, "cx": cx, "cy": cy, "w": w, "h": h,
                "area": area, "ar": ar,
            })
    print(f"  {len(candidates)} candidate crops from template_set")
    _ma = sum(c["area"] for c in candidates) / len(candidates) if candidates else 0.0
    _bo = sum(1 for c in candidates
              if args.min_ar <= c["ar"] <= args.max_ar
              and c["area"] >= args.min_area_factor * _ma)
    _p = (100.0 * _bo / len(candidates)) if candidates else 0.0
    print(f"    after both quality filters -> selection pool: {_bo} ({_p:.1f}%)")

    # ----------------------------------------------------------
    # 3. Select N templates from entire template_set (no folds)
    # ----------------------------------------------------------
    print(f"Selecting {args.n_templates} templates from full template_set...")
    template_rng = random.Random(args.seed + 7777)
    templates = select_templates_from_pool(
        candidates=candidates,
        n_templates=args.n_templates,
        rng=template_rng,
        images_dir=args.images_dir,
        min_ar=args.min_ar,
        max_ar=args.max_ar,
        min_area_factor=args.min_area_factor,
        hash_bits=args.hash_bits,
        hash_threshold=args.hash_threshold,
        use_images=use_images,
    )
    print(f"  → {len(templates)} templates selected")

    # ----------------------------------------------------------
    # 4. Generate bootstrap samples
    # ----------------------------------------------------------
    print(f"Generating {args.n_bootstrap} bootstrap samples from {len(test_stems)} test images...")
    bootstrap_rng = random.Random(args.seed + 5555)
    bootstrap_indices = generate_bootstrap_indices(test_stems, args.n_bootstrap, bootstrap_rng)

    # ----------------------------------------------------------
    # 5. Generate randomized execution order
    # ----------------------------------------------------------
    print(f"Generating randomized execution order ({len(METHODS)} × {args.n_bootstrap} = "
          f"{len(METHODS) * args.n_bootstrap} runs)...")
    order_rng = random.Random(args.seed + 3333)
    execution_order = build_execution_order(args.n_bootstrap, METHODS, order_rng)

    # ----------------------------------------------------------
    # 6. Save plan JSON
    # ----------------------------------------------------------
    plan = {
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "n_templates": args.n_templates,
        "test_set": test_stems,
        "template_set_stems": sorted(template_stems_set),
        "templates": templates,
        "bootstrap_indices": bootstrap_indices,
        "methods": METHODS,
        "total_runs": len(execution_order),
        "proposed_config": {
            "feature_source": args.feature_source,
            "roi_size": args.roi_size,
            "pool_ks": args.pool_ks,
            "thresh": args.thresh,
            "mode": args.mode,
            "maj_thresh": args.maj_thresh,
            "max_crops": args.n_templates,
        },
        "yolo_config": {
            "model": args.yolo_model,
            "conf_threshold": args.yolo_conf,
        },
        "gdino_config": {
            "model": args.gdino_model,
            "conf_threshold": args.gdino_conf,
            "text_prompt": args.gdino_prompt,
        },
    }

    plan_path = os.path.join(args.output_dir, "baseline_plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Saved plan → {plan_path}")

    # ----------------------------------------------------------
    # 7. Save execution order
    # ----------------------------------------------------------
    order_path = os.path.join(args.output_dir, "order_baselines.txt")
    write_order_file(execution_order, order_path)
    print(f"Saved execution order → {order_path}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Baseline experiment prepared:")
    print(f"  test_set:        {len(test_stems)} images")
    print(f"  templates:       {len(templates)} (from full template_set)")
    print(f"  bootstrap trials: {args.n_bootstrap}")
    print(f"  methods:         {METHODS}")
    print(f"  total runs:      {len(execution_order)}")
    print(f"  output_dir:      {args.output_dir}")
    print(f"{'='*60}")
    print(f"\nProposed model config (frozen from grid search):")
    print(f"  feature_source={args.feature_source}, roi_size={args.roi_size}, "
          f"pool_ks={args.pool_ks}, thresh={args.thresh}, mode={args.mode}")
    print(f"\nYOLO config: model={args.yolo_model}, conf={args.yolo_conf}")
    print(f"Grounding DINO config: model={args.gdino_model}, conf={args.gdino_conf}, "
          f"prompt='{args.gdino_prompt}'")


if __name__ == "__main__":
    main()
