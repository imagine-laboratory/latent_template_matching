#!/usr/bin/env python
"""
prepare_experiment.py  –  Prepare a reproducible experiment plan for the VAE grid search.

What it does:
  1. Reads the GT CSV (filename, class, cx, cy, w, h — one row per bounding box).
  2. Computes per-image crop count and stratifies images into quintiles.
  3. Splits stratified images into template_set (25%), eval_set (38%) and test_set (38%).
  4. For each replica (0..N-1), selects `max_templates` template crops from
     template_set images using quality filters:
       - Aspect ratio: min_ar <= h/w <= max_ar  (default 0.6–1.4), evaluated in
         PIXELS of the original frame, not on the normalised YOLO values
       - Minimum crop area (relative to mean): min_area_factor (default 0.33)
       - Perceptual-hash deduplication: skip crops too similar to already-selected ones
     Template pools are disjoint across replicas (fold-based sampling within template_set).
  5. Generates a randomized run order over all (replica × grid) combinations.
  6. Saves everything to a JSON plan file and a run_order.txt.

Usage:
  python prepare_experiment.py \\
      --csv         ./extra/labels_contents.csv \\
      --images_dir  ./extra/FULL_UNIFIED \\
      --output_dir  ./extra/output/grid_search \\
      --replicas    3 \\
      --seed        42

  # Dry-run (skip image loading / perceptual hash — fast, no images needed):
  python prepare_experiment.py --csv ... --output_dir ... --no_images

Output files (inside --output_dir):
  template_eval_folds.json   – full plan (sets, templates per replica, metadata)
  run_order.txt              – randomized tab-separated run list
"""

import os
import sys
import json
import argparse
import random
import hashlib
import csv as csv_mod
from collections import defaultdict
from itertools import product as iproduct

# Optional heavy deps — only needed when --no_images is NOT set
try:
    import numpy as np
    from PIL import Image
    import imagehash
    _HAVE_IMG = True
except ImportError:
    _HAVE_IMG = False

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False

# ============================================================
# Grid definition (keep in sync with run_grid_search.py)
# ============================================================
FEATURE_SOURCES     = ["feats"]
MAX_CROPS_LIST      = [5, 30, 60, 90]
ROI_SIZES           = [9, 11, 13]
RESIZE_IMGS         = [256]
MODES               = ["majority_voting", "average"]
THRESHOLDS          = [0.30, 0.37, 0.44, 0.51, 0.58]
POOL_KS_LIST        = [7, 9, 11]
MAJORITY_THRESHOLDS = [0.37]  # only for majority_voting mode


# ============================================================
# Defaults for template quality filters
# ============================================================
DEFAULT_MIN_AR          = 0.6    # min h/w aspect ratio for a crop
DEFAULT_MAX_AR          = 1.4    # max h/w aspect ratio
DEFAULT_MIN_AREA_FACTOR = 0.33   # crop area must be > factor * mean_area
DEFAULT_HASH_BITS       = 8      # imagehash hash size (higher = more sensitive)
# ------------------------------------------------------------
# Original frame size of the dataset.
# The CSV stores YOLO-normalised boxes: w is a fraction of the frame WIDTH and h
# a fraction of the frame HEIGHT.  Because the frame is not square, h/w taken on
# the normalised values equals the true pixel aspect ratio times H/W (0.5625).
# A square fruit (h/w = 1.0 in pixels) is therefore seen as 1.78 and rejected by
# a 0.6-1.4 filter, while boxes much wider than tall are kept -- the opposite of
# the intended criterion.  The filter must be evaluated in pixels.
# ------------------------------------------------------------
DEFAULT_FRAME_W = 1920
DEFAULT_FRAME_H = 1080

DEFAULT_HASH_THRESHOLD  = 10     # max hamming distance to consider two crops "similar"
                                 # (out of hash_bits**2; 10/64 ≈ 15% difference)

# ============================================================
# Image file extensions to look for
# ============================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ============================================================
# Helpers
# ============================================================

def find_image_path(images_dir: str, stem: str) -> str | None:
    """Return the full path of an image given its stem (no extension)."""
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def load_crop(image_path: str, cx: float, cy: float, w: float, h: float):
    """
    Load a bounding-box crop from an image.
    Coordinates are normalised (YOLO format).
    Returns a PIL Image or None on failure.
    """
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
    """Return a perceptual hash for a PIL crop image."""
    return imagehash.phash(crop_img, hash_size=hash_bits)


def is_too_similar(new_hash, selected_hashes: list, threshold: int) -> bool:
    """Return True if new_hash is within `threshold` hamming distance of any selected hash."""
    for h in selected_hashes:
        if abs(new_hash - h) <= threshold:
            return True
    return False


def stratified_split(
    image_counts: dict,
    template_frac: float,
    eval_frac: float,
    rng: random.Random,
) -> tuple[list, list, list]:
    """
    Stratify images by crop count using quintiles, then split into
    (template_set, eval_set, test_set) with the given fractions.

    Returns (template_stems, eval_stems, test_stems) as sorted lists for determinism.
    """
    stems = sorted(image_counts.keys())
    counts = [image_counts[s] for s in stems]

    if _HAVE_PANDAS:
        import pandas as pd
        s = pd.Series(counts, index=stems)
        # qcut with 5 bins; duplicates='drop' handles uneven distributions
        labels = pd.qcut(s, q=5, labels=False, duplicates="drop")
        strata = defaultdict(list)
        for stem, label in zip(stems, labels):
            strata[int(label)].append(stem)
    else:
        # Fallback: manual quintile binning
        sorted_by_count = sorted(stems, key=lambda s: image_counts[s])
        n = len(sorted_by_count)
        strata = defaultdict(list)
        for i, stem in enumerate(sorted_by_count):
            bucket = min(4, int(i / n * 5))
            strata[bucket].append(stem)

    template_stems, eval_stems, test_stems = [], [], []
    for bucket in sorted(strata.keys()):
        bucket_stems = sorted(strata[bucket])  # sort for reproducibility
        rng.shuffle(bucket_stems)
        n_template = max(1, round(len(bucket_stems) * template_frac))
        remainder  = len(bucket_stems) - n_template
        # Split remainder between eval and test proportionally
        eval_relative = eval_frac / (eval_frac + (1.0 - template_frac - eval_frac))
        n_eval = round(remainder * eval_relative)
        template_stems.extend(bucket_stems[:n_template])
        eval_stems.extend(bucket_stems[n_template:n_template + n_eval])
        test_stems.extend(bucket_stems[n_template + n_eval:])

    return sorted(template_stems), sorted(eval_stems), sorted(test_stems)


def pixel_aspect_ratio(w_norm: float, h_norm: float,
                       frame_w: int = DEFAULT_FRAME_W,
                       frame_h: int = DEFAULT_FRAME_H) -> float:
    """Aspect ratio h/w in pixels of the original frame (0.0 for degenerate boxes)."""
    if w_norm <= 0:
        return 0.0
    return (h_norm * frame_h) / (w_norm * frame_w)


def build_candidate_pool(boxes: list, image_counts: dict) -> list:
    """
    From the full box list, keep only boxes belonging to template_set images.
    Returns list of dicts with keys: stem, box_index, cx, cy, w, h, area, ar.
    """
    candidates = []
    for box in boxes:
        area = box["w"] * box["h"]
        ar   = pixel_aspect_ratio(box["w"], box["h"])
        candidates.append({**box, "area": area, "ar": ar})
    return candidates


def select_templates_for_replica(
    candidates:      list,
    n_templates:     int,
    fold_idx:        int,
    n_folds:         int,
    rng:             random.Random,
    images_dir:      str | None,
    min_ar:          float,
    max_ar:          float,
    min_area_factor: float,
    hash_bits:       int,
    hash_threshold:  int,
    use_images:      bool,
) -> list:
    """
    Select up to `n_templates` crops from `candidates` for one replica.

    Strategy:
      1. Apply aspect-ratio and minimum-area filters.
      2. Divide filtered candidates into `n_folds` disjoint folds (by stem),
         use `fold_idx` fold as the primary pool for this replica.
      3. Shuffle the pool, then greedily select crops, rejecting those
         perceptually too similar to already-selected ones (if use_images).
      4. If the fold pool is exhausted before reaching n_templates, fill up
         from the remaining candidates (still applying similarity check).
    """
    mean_area = (sum(c["area"] for c in candidates) / len(candidates)
                 if candidates else 0.0)
    min_area  = min_area_factor * mean_area

    # 1. Quality filter
    filtered = [
        c for c in candidates
        if min_ar <= c["ar"] <= max_ar and c["area"] >= min_area
    ]

    if not filtered:
        print(f"  WARN: no candidates passed quality filters — returning empty template list.")
        return []

    # 2. Fold assignment (by stem so same image never spans folds).
    # The shuffle that partitions stems into folds must be the SAME across all
    # replicas — only fold_idx changes.  Use a shared seed derived from the
    # global seed so every call with the same seed+n_folds sees the same partition.
    unique_stems = sorted({c["stem"] for c in filtered})
    shared_rng = random.Random(42_000 + n_folds)   # same for every replica
    shared_rng.shuffle(unique_stems)

    fold_stems = set()
    for i, stem in enumerate(unique_stems):
        if i % n_folds == fold_idx:
            fold_stems.add(stem)

    fold_pool    = [c for c in filtered if c["stem"] in fold_stems]
    reserve_pool = [c for c in filtered if c["stem"] not in fold_stems]

    rng.shuffle(fold_pool)
    rng.shuffle(reserve_pool)

    # 3. Greedy selection with deduplication
    selected       = []
    selected_hashes = []

    def try_add(crop_candidate):
        nonlocal selected, selected_hashes
        if use_images and _HAVE_IMG and images_dir:
            img_path = find_image_path(images_dir, crop_candidate["stem"])
            if img_path is None:
                return False
            crop_img = load_crop(
                img_path,
                crop_candidate["cx"], crop_candidate["cy"],
                crop_candidate["w"],  crop_candidate["h"],
            )
            if crop_img is None:
                return False
            h = perceptual_hash(crop_img, hash_bits)
            if is_too_similar(h, selected_hashes, hash_threshold):
                return False
            selected_hashes.append(h)
        selected.append(crop_candidate)
        return True

    for c in fold_pool:
        if len(selected) >= n_templates:
            break
        try_add(c)

    if len(selected) < n_templates:
        for c in reserve_pool:
            if len(selected) >= n_templates:
                break
            try_add(c)

    if len(selected) < n_templates:
        print(f"  WARN: replica fold {fold_idx}: only {len(selected)}/{n_templates} "
              f"templates selected (pool exhausted or all filtered by similarity).")

    return [
        {k: v for k, v in c.items() if k in ("stem", "box_index", "cx", "cy", "w", "h")}
        for c in selected
    ]


def build_run_order(replicas: int, rng: random.Random) -> list[dict]:
    """
    Generate all (replica, feature_source, max_crops, roi_size, mode,
    thresh, pool_ks, maj_th) combinations, apply skip rules, then shuffle.
    Returns list of dicts.
    """
    runs = []
    for (replica, feature_source, resize_img, max_crops,
         roi_size, pool_ks, mode, thresh) in iproduct(
            range(replicas),
            FEATURE_SOURCES, RESIZE_IMGS, MAX_CROPS_LIST,
            ROI_SIZES, POOL_KS_LIST, MODES, THRESHOLDS,
    ):
        # Skip rules (same as run_grid_search.py)
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
                maj_th=maj_th if maj_th is not None else "",
            ))

    rng.shuffle(runs)
    return runs


def write_run_order(runs: list[dict], path: str) -> None:
    """Write the run order to a tab-separated .txt file matching the example format."""
    with open(path, "w") as f:
        for i, r in enumerate(runs):
            parts = [str(i)]
            parts.append(f"replica={r['replica']}")
            parts.append(f"feature_source={r['feature_source']}")
            parts.append(f"resize_img={r['resize_img']}")
            parts.append(f"max_crops={r['max_crops']}")
            parts.append(f"roi_size={r['roi_size']}")
            parts.append(f"mode={r['mode']}")
            parts.append(f"thresh={r['thresh']}")
            parts.append(f"pool_ks={r['pool_ks']}")
            if r["maj_th"] != "":
                parts.append(f"maj_th={r['maj_th']}")
            f.write("\t".join(parts) + "\n")


# ============================================================
# CLI
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Prepare reproducible experiment plan (splits + templates + run order).")
    ap.add_argument("--csv",          required=True,
                    help="Path to labels_contents.csv (YOLO format, one box per row).")
    ap.add_argument("--images_dir",   default=None,
                    help="Directory with images (needed for perceptual-hash filtering). "
                         "If omitted, similarity check is skipped.")
    ap.add_argument("--output_dir",   required=True,
                    help="Where to write template_eval_folds.json and run_order.txt.")
    ap.add_argument("--replicas",     type=int, default=3,
                    help="Number of replicas / folds (default: 3).")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--template_split", type=float, default=0.25,
                    help="Fraction of images per stratum allocated to template_set (default: 0.25).")
    ap.add_argument("--eval_split", type=float, default=0.38,
                    help="Fraction of images per stratum allocated to eval_set (default: 0.38). "
                         "Remainder goes to test_set.")
    ap.add_argument("--min_ar",       type=float, default=DEFAULT_MIN_AR)
    ap.add_argument("--max_ar",       type=float, default=DEFAULT_MAX_AR)
    ap.add_argument("--min_area_factor", type=float, default=DEFAULT_MIN_AREA_FACTOR)
    ap.add_argument("--frame_w", type=int, default=DEFAULT_FRAME_W,
                    help="Width in px of the original frames (default: 1920).")
    ap.add_argument("--frame_h", type=int, default=DEFAULT_FRAME_H,
                    help="Height in px of the original frames (default: 1080).")
    ap.add_argument("--hash_bits",    type=int,   default=DEFAULT_HASH_BITS)
    ap.add_argument("--hash_threshold", type=int, default=DEFAULT_HASH_THRESHOLD)
    ap.add_argument("--no_images",    action="store_true",
                    help="Skip image loading entirely (no perceptual-hash dedup). Fast mode.")
    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    use_images = (not args.no_images) and (args.images_dir is not None)
    if use_images and not _HAVE_IMG:
        print("WARN: Pillow / imagehash not installed. Falling back to --no_images mode.")
        use_images = False

    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)

    # ----------------------------------------------------------
    # 1. Parse CSV → per-image box list
    # ----------------------------------------------------------
    print("Reading CSV ...")
    image_boxes = defaultdict(list)   # stem -> list of box dicts
    global_box_index = 0

    with open(args.csv, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            stem = os.path.splitext(row["filename"])[0]
            image_boxes[stem].append({
                "stem":      stem,
                "box_index": global_box_index,
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "w":  float(row["w"]),
                "h":  float(row["h"]),
            })
            global_box_index += 1

    image_counts = {stem: len(boxes) for stem, boxes in image_boxes.items()}
    all_boxes    = [box for boxes in image_boxes.values() for box in boxes]

    print(f"  {len(image_counts)} images, {len(all_boxes)} total boxes.")

    # ----------------------------------------------------------
    # 2. Stratified split
    # ----------------------------------------------------------
    test_frac = 1.0 - args.template_split - args.eval_split
    print(f"Stratifying by crop count (quintiles) and splitting "
          f"{args.template_split:.0%} / {args.eval_split:.0%} / {test_frac:.0%} ...")
    template_stems, eval_stems, test_stems = stratified_split(
        image_counts, args.template_split, args.eval_split, rng)
    print(f"  template_set: {len(template_stems)} images")
    print(f"  eval_set:     {len(eval_stems)} images")
    print(f"  test_set:     {len(test_stems)} images")

    # ----------------------------------------------------------
    # 3. Build candidate pool (template_set boxes only)
    # ----------------------------------------------------------
    template_stem_set = set(template_stems)
    candidates = [b for b in all_boxes if b["stem"] in template_stem_set]
    # Annotate with area and aspect ratio
    for c in candidates:
        c["area"] = c["w"] * c["h"]
        c["ar"]   = pixel_aspect_ratio(c["w"], c["h"], args.frame_w, args.frame_h)

    print(f"  {len(candidates)} candidate crops in template_set.")

    # Report the survival rate of each filter.  The original script printed
    # nothing here, so an over-restrictive filter went unnoticed: the selection
    # only warns when the pool runs out before n_templates.
    _ma = sum(c["area"] for c in candidates) / len(candidates) if candidates else 0.0
    _ar = sum(1 for c in candidates if args.min_ar <= c["ar"] <= args.max_ar)
    _aa = sum(1 for c in candidates if c["area"] >= args.min_area_factor * _ma)
    _bo = sum(1 for c in candidates
              if args.min_ar <= c["ar"] <= args.max_ar
              and c["area"] >= args.min_area_factor * _ma)
    _p = lambda n: (100.0 * n / len(candidates)) if candidates else 0.0
    print(f"    aspect ratio {args.min_ar}-{args.max_ar} (px): {_ar} pass ({_p(_ar):.1f}%)")
    print(f"    area >= {args.min_area_factor} * mean:        {_aa} pass ({_p(_aa):.1f}%)")
    print(f"    both filters -> selection pool:   {_bo} pass ({_p(_bo):.1f}%)")

    # ----------------------------------------------------------
    # 4. Select templates per replica
    # ----------------------------------------------------------
    max_templates = max(MAX_CROPS_LIST)
    replicas_data = []

    for fold_idx in range(args.replicas):
        print(f"Selecting templates for replica {fold_idx} (fold {fold_idx}/{args.replicas}) ...")
        # Fresh seeded rng per replica for reproducibility
        replica_rng = random.Random(args.seed + fold_idx * 1000)
        templates = select_templates_for_replica(
            candidates      = candidates,
            n_templates     = max_templates,
            fold_idx        = fold_idx,
            n_folds         = args.replicas,
            rng             = replica_rng,
            images_dir      = args.images_dir,
            min_ar          = args.min_ar,
            max_ar          = args.max_ar,
            min_area_factor = args.min_area_factor,
            hash_bits       = args.hash_bits,
            hash_threshold  = args.hash_threshold,
            use_images      = use_images,
        )
        replicas_data.append(templates)
        print(f"  → {len(templates)} templates selected.")

    # ----------------------------------------------------------
    # 5. Randomised run order
    # ----------------------------------------------------------
    print("Generating randomised run order ...")
    run_order_rng = random.Random(args.seed + 9999)
    runs = build_run_order(args.replicas, run_order_rng)
    print(f"  {len(runs)} total runs.")

    # ----------------------------------------------------------
    # 6. Save JSON plan
    # ----------------------------------------------------------
    mean_area = sum(c["area"] for c in candidates) / len(candidates) if candidates else 0.0

    plan = {
        "seed":         args.seed,
        "replicas":     args.replicas,
        "folds_N":      args.replicas,
        "max_templates": max_templates,
        "template_split": args.template_split,
        "eval_split":   args.eval_split,
        "test_split":   round(1.0 - args.template_split - args.eval_split, 4),
        "aspect_ratio_filter": f"{args.min_ar} <= h/w <= {args.max_ar} "
                               f"(in pixels of a {args.frame_w}x{args.frame_h} frame)",
        "frame_w":      args.frame_w,
        "frame_h":      args.frame_h,
        "min_area": f"> {args.min_area_factor} * mean_area  (mean_area={mean_area:.6f})",
        "hash_bits":    args.hash_bits,
        "hash_threshold": args.hash_threshold,
        "similarity_check_used": use_images,
        "template_set": template_stems,
        "eval_set":     eval_stems,
        "test_set":     test_stems,
        "templates_per_replica": replicas_data,
    }

    json_path = os.path.join(args.output_dir, "template_eval_folds.json")
    with open(json_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Saved plan → {json_path}")

    # ----------------------------------------------------------
    # 7. Save run order
    # ----------------------------------------------------------
    run_order_path = os.path.join(args.output_dir, "run_order.txt")
    write_run_order(runs, run_order_path)
    print(f"Saved run order → {run_order_path}")

    print("\nDone.")
    print(f"  template_set : {len(template_stems)} images")
    print(f"  eval_set     : {len(eval_stems)} images")
    print(f"  test_set     : {len(test_stems)} images")
    print(f"  replicas     : {args.replicas}")
    print(f"  templates/replica: {[len(r) for r in replicas_data]}")
    print(f"  total runs   : {len(runs)}")


if __name__ == "__main__":
    main()