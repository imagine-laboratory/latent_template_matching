#!/usr/bin/env python3
"""One pass of each detector over the 738 test images, with per-image output.

The held-out comparison stored one score per bootstrap trial, which leaves four
things unverifiable from the results file: the statistic observed on the
original images, whether the three methods saw identical resamples, whether a
grouped bootstrap is needed, and what a permutation test would say.  All four
follow from per-image counts, because every metric the thesis uses is a
function of sums over images:

    MAE  = mean |pred - gt|
    F1   = 2*sum(TP) / (2*sum(TP) + sum(FP) + sum(FN))

So one pass is enough.  The 1,000 paired resamples can then be recomputed
offline from the plan's indices, with a single template bank, instead of
re-running the detectors a thousand times.

It also writes the template bank and its SHA-256, which is what makes the
method under test one fixed detector rather than one per process.

Usage (on the cluster, from /work/gbrizuela/VAE-big_mamma):

    python3 dump_per_image.py \
        --plan        new_output/plan_baselines/baseline_plan.json \
        --images_dir  /data/gbrizuela/datasets/FULL_UNIFIED \
        --csv         /data/gbrizuela/datasets/labels_contents.csv \
        --checkpoint  /work/gbrizuela/VAE/output/checkpoints/betaKL@0.1/best.pt \
        --out_dir     new_output/per_image
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline_comparison import (  # noqa: E402
    ProposedDetector, YOLODetector, GroundingDINODetector,
    read_yolo_csv, yolo_to_xyxy_pixels,
    match_predictions_to_gt, match_predictions_to_gt_center,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="./per_image")
    ap.add_argument("--resize_img", type=int, default=256)
    ap.add_argument("--iou_threshold", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plan = json.load(open(args.plan))
    cfg = plan["proposed_config"]
    print("proposed_config:", json.dumps(cfg, sort_keys=True))
    json.dump(cfg, open(os.path.join(args.out_dir, "proposed_config.json"), "w"),
              indent=2, sort_keys=True)

    stems = plan["test_set"]
    # Same extension probe, in the same order, that run_baseline_comparison.py
    # uses to resolve the test set.
    paths = {}
    for st in stems:
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(args.images_dir, st + ext)
            if os.path.isfile(cand):
                paths[st] = cand
                break
    print(f"resolved {len(paths)}/{len(stems)} test image paths")
    if len(paths) != len(stems):
        missing = [st for st in stems if st not in paths]
        sys.exit(f"{len(missing)} test images not found under {args.images_dir} "
                 f"with .png/.jpg/.jpeg, first: {missing[0]}")
    stems = [st for st in stems if st in paths]
    gt = read_yolo_csv(args.csv)        # stem -> [(cx, cy, w, h)] normalised

    det = {}
    det["proposed"] = ProposedDetector(
        checkpoint_path=args.checkpoint, templates_info=plan["templates"],
        images_dir=args.images_dir, csv_path=args.csv,
        feature_source=cfg["feature_source"], roi_size=cfg["roi_size"],
        pool_ks=cfg["pool_ks"], thresh=cfg["thresh"], mode=cfg["mode"],
        maj_thresh=cfg["maj_thresh"], max_crops=cfg["max_crops"],
        resize_img=args.resize_img, device=args.device)

    # The bank is what the three parallel jobs failed to share.  Save it and
    # its digest so any later run can prove it used the same one.
    bank = torch.stack([t.cpu() for t in det["proposed"].template_bank])
    bank_path = os.path.join(args.out_dir, "template_bank.pt")
    torch.save(bank, bank_path)
    sha = hashlib.sha256(open(bank_path, "rb").read()).hexdigest()
    print(f"template bank: {tuple(bank.shape)}  sha256 {sha}")
    open(os.path.join(args.out_dir, "template_bank.sha256"), "w").write(sha + "\n")

    det["yolo"] = YOLODetector(
        model_path=plan["yolo_config"]["model"],
        conf_threshold=plan["yolo_config"]["conf_threshold"],
        resize_img=args.resize_img, device=args.device)
    det["grounding_dino"] = GroundingDINODetector(
        model_id=plan["gdino_config"]["model"],
        conf_threshold=plan["gdino_config"]["conf_threshold"],
        text_prompt=plan["gdino_config"]["text_prompt"],
        resize_img=args.resize_img, device=args.device)

    out = os.path.join(args.out_dir, "per_image_scores.csv")
    cols = ["method", "stem", "gt_count", "pred_count",
            "tp_center", "fp_center", "fn_center", "tp_iou", "fp_iou", "fn_iou"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, d in det.items():
            for i, s in enumerate(stems):
                centers, boxes, scores = d.detect_image(paths[s])
                # Same conversion and matching evaluate_trial() applies, so the
                # per-image counts aggregate to the metrics already reported.
                gt_boxes = [yolo_to_xyxy_pixels(cx, cy, bw, bh,
                                                args.resize_img, args.resize_img)
                            for (cx, cy, bw, bh) in gt.get(s, [])]
                tpi, fpi, fni, _ = match_predictions_to_gt(
                    boxes, gt_boxes, args.iou_threshold)
                tpc, fpc, fnc, _ = match_predictions_to_gt_center(
                    centers, gt_boxes, sigma=0.5)
                w.writerow(dict(method=name, stem=s, gt_count=len(gt_boxes),
                                pred_count=len(boxes),
                                tp_center=tpc, fp_center=fpc, fn_center=fnc,
                                tp_iou=tpi, fp_iou=fpi, fn_iou=fni))
                if i % 200 == 0:
                    print(f"  {name}: {i}/{len(stems)}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
