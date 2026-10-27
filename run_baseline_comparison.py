#!/usr/bin/env python
"""
run_baseline_comparison.py — Execute the bootstrap-based baseline comparison.

Follows the protocol from prepare_baseline_experiment.py:
  - Loads the baseline_plan.json
  - Executes 3000 runs (3 methods × 1000 trials) in the pre-randomized order
  - Each trial uses the SAME bootstrapped test image subset across all 3 models
  - Saves results_baselines.csv with trial, method, and all metrics

Usage:
  python run_baseline_comparison.py \
      --plan ./extra/output/baselines_bootstrap/baseline_plan.json \
      --images_dir ./extra/FULL_UNIFIED \
      --csv ./extra/labels_contents.csv \
      --checkpoint ./extra/best.pt \
      --output_dir ./extra/output/baselines_bootstrap \
      --device cuda

Requirements:
  pip install ultralytics transformers torch torchvision
"""

import os
import sys
import json
import time
import argparse
import csv as csv_mod
import glob
from typing import Dict, List, Tuple

import resource

import numpy as np
import cv2
import torch
import torch.nn.functional as F


# ============================================================
# Cost instrumentation
# ============================================================
# The three detectors live in ONE process and stay resident at the same time,
# so torch.cuda.max_memory_allocated() during a run includes the weights of all
# of them.  Three quantities are therefore recorded:
#   model_mb    - what each detector allocates when it is built (its own weights)
#   gpu_peak_mb - peak allocated during the run (reset before each run)
#   gpu_delta_mb- peak minus the resident baseline = incremental cost of the run
# Without reset_peak_memory_stats() before each run the peak would be the
# maximum since process start and every method would report the same number.

def _cuda_mb(fn):
    """Return (result, MB allocated by fn) measured on the current device."""
    if not torch.cuda.is_available():
        return fn(), 0.0
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return out, (torch.cuda.memory_allocated() - before) / 1024 ** 2


def _gpu_now_mb():
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 ** 2


def _gpu_peak_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_reserved() / 1024 ** 2)


def _host_peak_mb():
    # ru_maxrss is KB on Linux; it is a process-wide high-water mark, so it only
    # grows.  Useful as a final figure, not as a per-run delta.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ============================================================
# Evaluation helpers (shared across all methods)
# ============================================================

def read_yolo_csv(csv_path: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    mapping: Dict[str, List[Tuple[float, float, float, float]]] = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv_mod.DictReader(f)
        for row in rdr:
            stem = os.path.splitext(row["filename"])[0]
            cx, cy, w, h = map(float, (row["cx"], row["cy"], row["w"], row["h"]))
            mapping.setdefault(stem, []).append((cx, cy, w, h))
    return mapping


def yolo_to_xyxy_pixels(cx, cy, w, h, W, H):
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return (x1, y1, x2, y2)


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold):
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    tp_flags = [0] * n_pred
    if n_pred == 0:
        return 0, 0, n_gt, tp_flags
    if n_gt == 0:
        return 0, n_pred, 0, tp_flags

    pairs = []
    for pi in range(n_pred):
        for gi in range(n_gt):
            iou = compute_iou(pred_boxes[pi], gt_boxes[gi])
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(key=lambda x: -x[0])

    matched_pred = set()
    matched_gt = set()
    for iou_val, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            tp_flags[pi] = 1

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn, tp_flags


def match_predictions_to_gt_center(pred_centers, gt_boxes, sigma=0.5):
    n_pred = len(pred_centers)
    n_gt = len(gt_boxes)
    tp_flags = [0] * n_pred
    if n_pred == 0:
        return 0, 0, n_gt, tp_flags
    if n_gt == 0:
        return 0, n_pred, 0, tp_flags

    gt_centers = []
    gt_radii = []
    for (x1, y1, x2, y2) in gt_boxes:
        gcx = (x1 + x2) / 2.0
        gcy = (y1 + y2) / 2.0
        diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        gt_centers.append((gcx, gcy))
        gt_radii.append(sigma * diag)

    pairs = []
    for pi, (pcx, pcy) in enumerate(pred_centers):
        for gi, ((gcx, gcy), radius) in enumerate(zip(gt_centers, gt_radii)):
            dist = np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
            if dist <= radius:
                pairs.append((dist, pi, gi))
    pairs.sort(key=lambda x: x[0])

    matched_pred = set()
    matched_gt = set()
    for _, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            tp_flags[pi] = 1

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn, tp_flags


def compute_ap(scores, tp_flags, total_gt):
    if total_gt == 0 or len(scores) == 0:
        return 0.0
    scores = np.array(scores, dtype=np.float64)
    tp_flags = np.array(tp_flags, dtype=np.float64)
    order = np.argsort(-scores)
    tp_sorted = tp_flags[order]
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1.0 - tp_sorted)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def compute_trial_metrics(preds_list, gts_list,
                          all_scores_iou, all_tp_flags_iou, total_gt_iou,
                          all_scores_center, all_tp_flags_center, total_gt_center):
    """Compute all metrics for a single trial."""
    preds = np.array(preds_list)
    gts = np.array(gts_list)
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    mask_nonzero = gts > 0
    if mask_nonzero.any():
        mape = float(np.mean(np.abs(preds[mask_nonzero] - gts[mask_nonzero]) / gts[mask_nonzero]) * 100.0)
    else:
        mape = 0.0

    total_tp = sum(all_tp_flags_iou)
    total_fp = len(all_tp_flags_iou) - total_tp
    total_fn = total_gt_iou - total_tp

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ap = compute_ap(all_scores_iou, all_tp_flags_iou, total_gt_iou)

    total_tp_c = sum(all_tp_flags_center)
    total_fp_c = len(all_tp_flags_center) - total_tp_c
    total_fn_c = total_gt_center - total_tp_c
    p_center = total_tp_c / (total_tp_c + total_fp_c) if (total_tp_c + total_fp_c) > 0 else 0.0
    r_center = total_tp_c / (total_tp_c + total_fn_c) if (total_tp_c + total_fn_c) > 0 else 0.0
    f1_center = 2 * p_center * r_center / (p_center + r_center) if (p_center + r_center) > 0 else 0.0
    ap_center = compute_ap(all_scores_center, all_tp_flags_center, total_gt_center)

    return dict(
        mae=mae, rmse=rmse, mape=mape,
        precision=precision, recall=recall, f1=f1, ap=ap,
        p_center=p_center, r_center=r_center, f1_center=f1_center, ap_center=ap_center,
    )


# ============================================================
# Proposed method (VAE latent template matching)
# ============================================================

class ProposedDetector:
    """
    Wraps the VAE latent-space template matching pipeline for batch evaluation.
    Loads model and templates once, then evaluates per-image.
    """

    def __init__(self, checkpoint_path, templates_info, images_dir, csv_path,
                 feature_source, roi_size, pool_ks, thresh, mode, maj_thresh,
                 max_crops, resize_img, device):
        from models.vae import VAE
        from tools.utils import select_device, set_seed
        from experiments.count_from_latent import (
            _get_feature_map, compute_response_map, peak_pick,
            merge_close_peaks, peaks_to_adaptive_boxes,
            standardize_channelwise_featmap,
        )

        self.device = select_device(device)
        self.resize_img = resize_img
        self.feature_source = feature_source
        self.roi_size = roi_size
        self.pool_ks = pool_ks
        self.thresh = thresh
        self.mode = mode
        self.maj_thresh = maj_thresh
        self.max_crops = max_crops

        # Store function references
        self._get_feature_map = _get_feature_map
        self._compute_response_map = compute_response_map
        self._peak_pick = peak_pick
        self._merge_close_peaks = merge_close_peaks
        self._peaks_to_adaptive_boxes = peaks_to_adaptive_boxes

        # Load model
        self.model = VAE().to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # Extract templates (need to run the encoder on template images)
        self.template_bank = self._extract_templates(
            templates_info, images_dir, csv_path)

        # Build average template if in average mode
        if self.mode == "average" and len(self.template_bank) > 0:
            self.avg_template = torch.stack(self.template_bank).mean(dim=0)
        else:
            self.avg_template = None

    def _extract_templates(self, templates_info, images_dir, csv_path):
        """Extract latent templates using RoIAlign from the encoder."""
        from torchvision.ops import roi_align

        templates_info = templates_info[:self.max_crops]
        template_bank = []

        for t in templates_info:
            stem = t["stem"]
            cx, cy, w, h = t["cx"], t["cy"], t["w"], t["h"]

            # Find and load image
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                p = os.path.join(images_dir, stem + ext)
                if os.path.isfile(p):
                    img_path = p
                    break
            if img_path is None:
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.resize_img, self.resize_img), interpolation=cv2.INTER_AREA)
            x = img.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)

            # Get feature map
            _, _, H, W = x.shape
            zH, zW = H // 8, W // 8
            noise = torch.zeros((1, 4, zH, zW), device=self.device)

            with torch.no_grad():
                fmap = self._get_feature_map(self.model, x, self.feature_source, noise)

            # Convert YOLO box to pixel coords in input space
            x1_px = (cx - w / 2.0) * self.resize_img
            y1_px = (cy - h / 2.0) * self.resize_img
            x2_px = (cx + w / 2.0) * self.resize_img
            y2_px = (cy + h / 2.0) * self.resize_img

            # Scale to feature map space
            scale = fmap.shape[2] / self.resize_img
            x1_f = x1_px * scale
            y1_f = y1_px * scale
            x2_f = x2_px * scale
            y2_f = y2_px * scale

            # RoIAlign
            rois = torch.tensor([[0, x1_f, y1_f, x2_f, y2_f]], dtype=torch.float32,
                                device=self.device)
            roi_out = roi_align(fmap, rois, output_size=self.roi_size,
                                spatial_scale=1.0, aligned=True)  # (1, C, roi_size, roi_size)

            template = roi_out.squeeze(0)  # (C, roi_size, roi_size)

            # Channel-wise standardization
            mean_c = template.mean(dim=(1, 2), keepdim=True)
            std_c = template.std(dim=(1, 2), keepdim=True)
            template = (template - mean_c) / (std_c + 1e-6)

            template_bank.append(template.cpu())

        return template_bank

    def detect_image(self, img_path):
        """
        Run detection on a single image.
        Returns: (pred_centers, pred_boxes, scores) where
          pred_centers: list of (cx, cy) in pixel space
          pred_boxes: list of (x1, y1, x2, y2) in pixel space
          scores: list of confidence scores
        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return [], [], []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize_img, self.resize_img), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)

        _, _, H, W = x.shape
        zH, zW = H // 8, W // 8
        noise = torch.zeros((1, 4, zH, zW), device=self.device)

        with torch.no_grad():
            fmap = self._get_feature_map(self.model, x, self.feature_source, noise)

        h_resp, w_resp = fmap.shape[2], fmap.shape[3]
        scale_y = self.resize_img / float(h_resp)
        scale_x = self.resize_img / float(w_resp)

        if self.mode == "average":
            if self.avg_template is None:
                return [], [], []
            resp01 = self._compute_response_map(fmap, self.avg_template.to(self.device))
            peaks_hw, peak_vals = self._peak_pick(resp01, thresh=self.thresh, pool_ks=self.pool_ks)
            resp_np = resp01[0, 0].cpu().numpy()
            pred_boxes = self._peaks_to_adaptive_boxes(
                peaks_hw, resp_np, scale_y, scale_x, roi_size=self.roi_size)

        elif self.mode == "majority_voting":
            N = len(self.template_bank)
            if N == 0:
                return [], [], []
            agreement = torch.zeros((h_resp, w_resp), device=self.device, dtype=torch.float32)
            mean_corr = torch.zeros((h_resp, w_resp), device=self.device, dtype=torch.float32)
            for proto_i in self.template_bank:
                resp01_i = self._compute_response_map(fmap, proto_i.to(self.device))
                vote = (resp01_i >= self.thresh).float()
                agreement += vote[0, 0]
                mean_corr += resp01_i[0, 0]
            agreement_norm = agreement / max(N, 1)
            mean_corr = mean_corr / max(N, 1)
            agr_t = agreement_norm.unsqueeze(0).unsqueeze(0)
            peaks_hw, peak_vals = self._peak_pick(agr_t, thresh=self.maj_thresh, pool_ks=self.pool_ks)
            peaks_hw, peak_vals, _ = self._merge_close_peaks(
                peaks_hw, peak_vals, scale_y, scale_x, 30.0, base_box_size=30.0)
            mean_corr_np = mean_corr.cpu().numpy()
            pred_boxes = self._peaks_to_adaptive_boxes(
                peaks_hw, mean_corr_np, scale_y, scale_x, roi_size=self.roi_size)

        else:
            # single — use first template
            if len(self.template_bank) == 0:
                return [], [], []
            resp01 = self._compute_response_map(fmap, self.template_bank[0].to(self.device))
            peaks_hw, peak_vals = self._peak_pick(resp01, thresh=self.thresh, pool_ks=self.pool_ks)
            resp_np = resp01[0, 0].cpu().numpy()
            pred_boxes = self._peaks_to_adaptive_boxes(
                peaks_hw, resp_np, scale_y, scale_x, roi_size=self.roi_size)

        # Compute centers and scores
        pred_centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for (x1, y1, x2, y2) in pred_boxes]
        scores = peak_vals.tolist() if len(peak_vals) > 0 else []

        return pred_centers, pred_boxes, scores


# ============================================================
# YOLO detector wrapper
# ============================================================

class YOLODetector:
    """Wraps YOLO for consistent interface."""

    def __init__(self, model_path, conf_threshold, resize_img, device):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.resize_img = resize_img
        self.device = device

    def detect_image(self, img_path):
        """
        Returns: (pred_centers, pred_boxes, scores)
        All in self.resize_img coordinate space.
        """
        result = self.model.predict(img_path, verbose=False, device=self.device, imgsz=640)[0]
        img_h, img_w = result.orig_shape
        boxes = result.boxes

        pred_boxes = []
        pred_centers = []
        scores = []

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            # Scale to resize_img
            x1_s = x1 * self.resize_img / img_w
            y1_s = y1 * self.resize_img / img_h
            x2_s = x2 * self.resize_img / img_w
            y2_s = y2 * self.resize_img / img_h
            pred_boxes.append((x1_s, y1_s, x2_s, y2_s))
            pred_centers.append(((x1_s + x2_s) / 2.0, (y1_s + y2_s) / 2.0))
            scores.append(conf)

        return pred_centers, pred_boxes, scores


# ============================================================
# Grounding DINO detector wrapper
# ============================================================

class GroundingDINODetector:
    """Wraps Grounding DINO for consistent interface."""

    def __init__(self, model_id, conf_threshold, text_prompt, resize_img, device):
        from PIL import Image as PILImage

        try:
            from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
            ProcessorClass = GroundingDinoProcessor
            ModelClass = GroundingDinoForObjectDetection
        except ImportError:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            ProcessorClass = AutoProcessor
            ModelClass = AutoModelForZeroShotObjectDetection

        self.processor = ProcessorClass.from_pretrained(model_id)
        self.model = ModelClass.from_pretrained(model_id).to(device)
        self.model.eval()
        self.conf_threshold = conf_threshold
        self.text_prompt = text_prompt
        self.resize_img = resize_img
        self.device = device
        self.PILImage = PILImage

    def detect_image(self, img_path):
        """
        Returns: (pred_centers, pred_boxes, scores)
        All in self.resize_img coordinate space.
        """
        img_pil = self.PILImage.open(img_path).convert("RGB")
        img_w, img_h = img_pil.size

        inputs = self.processor(images=img_pil, text=self.text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.sigmoid()
        scores_t = logits.max(dim=-1).values[0]
        boxes_cxcywh = outputs.pred_boxes[0]

        cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()
        scores_np = scores_t.cpu().numpy()

        # Filter by confidence
        mask = scores_np >= self.conf_threshold
        filtered_boxes = boxes_xyxy[mask]
        filtered_scores = scores_np[mask]

        pred_boxes = []
        pred_centers = []
        scores = []

        for box, score in zip(filtered_boxes, filtered_scores):
            bx1, by1, bx2, by2 = box
            x1_s = bx1 * self.resize_img / img_w
            y1_s = by1 * self.resize_img / img_h
            x2_s = bx2 * self.resize_img / img_w
            y2_s = by2 * self.resize_img / img_h
            pred_boxes.append((x1_s, y1_s, x2_s, y2_s))
            pred_centers.append(((x1_s + x2_s) / 2.0, (y1_s + y2_s) / 2.0))
            scores.append(float(score))

        return pred_centers, pred_boxes, scores


# ============================================================
# Evaluate one trial for one method
# ============================================================

def evaluate_trial(detector, img_paths, gt_ann, resize_img, iou_threshold):
    """
    Run a detector on a list of image paths, compute all metrics.
    Returns a metrics dict.
    """
    preds_list, gts_list = [], []
    all_scores_iou, all_tp_flags_iou = [], []
    all_scores_center, all_tp_flags_center = [], []
    total_gt_iou = 0
    total_gt_center = 0

    for path in img_paths:
        stem = os.path.splitext(os.path.basename(path))[0]

        pred_centers, pred_boxes, scores = detector.detect_image(path)

        # GT boxes
        gt_boxes = []
        if stem in gt_ann:
            for (cx, cy, w, h) in gt_ann[stem]:
                gt_boxes.append(yolo_to_xyxy_pixels(cx, cy, w, h, resize_img, resize_img))

        count_pred = len(pred_boxes)
        count_gt = len(gt_boxes)

        preds_list.append(count_pred)
        gts_list.append(count_gt)
        total_gt_iou += count_gt
        total_gt_center += count_gt

        # IoU-based matching
        tp, fp, fn, tp_flags_img = match_predictions_to_gt(
            pred_boxes, gt_boxes, iou_threshold)
        all_scores_iou.extend(scores)
        all_tp_flags_iou.extend(tp_flags_img)

        # Center-based matching
        tp_c, fp_c, fn_c, tp_flags_center_img = match_predictions_to_gt_center(
            pred_centers, gt_boxes, sigma=0.5)
        all_scores_center.extend(scores)
        all_tp_flags_center.extend(tp_flags_center_img)

    metrics = compute_trial_metrics(
        preds_list, gts_list,
        all_scores_iou, all_tp_flags_iou, total_gt_iou,
        all_scores_center, all_tp_flags_center, total_gt_center,
    )
    return metrics


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Execute bootstrap-based baseline comparison (3 methods × B trials).")
    ap.add_argument("--plan", required=True,
                    help="Path to baseline_plan.json (from prepare_baseline_experiment.py).")
    ap.add_argument("--images_dir", default="./extra/FULL_UNIFIED")
    ap.add_argument("--csv", default="./extra/labels_contents.csv",
                    help="Full GT CSV.")
    ap.add_argument("--checkpoint", default="./extra/best.pt",
                    help="VAE checkpoint for proposed model.")
    ap.add_argument("--output_dir", default="./extra/output/baselines_bootstrap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resize_img", type=int, default=256)
    ap.add_argument("--iou_threshold", type=float, default=0.15)
    ap.add_argument("--resume_from", type=int, default=0,
                    help="Resume from run index (skip already completed runs).")
    ap.add_argument("--stop_at", type=int, default=0,
                    help="Stop before this run index (0 = run to the end). Used to split "
                         "the 3,000 evaluations across several jobs when the partition "
                         "walltime is shorter than the full run.")
    ap.add_argument("--skip_yolo", action="store_true",
                    help="Skip YOLO (run proposed + GD only).")
    ap.add_argument("--skip_gdino", action="store_true",
                    help="Skip Grounding DINO.")
    ap.add_argument("--skip_proposed", action="store_true",
                    help="Skip proposed method.")
    return ap.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load plan
    # ----------------------------------------------------------
    print(f"Loading plan: {args.plan}")
    with open(args.plan) as f:
        plan = json.load(f)

    test_stems = plan["test_set"]
    templates = plan["templates"]
    bootstrap_indices = plan["bootstrap_indices"]
    n_bootstrap = plan["n_bootstrap"]
    proposed_cfg = plan["proposed_config"]
    yolo_cfg = plan["yolo_config"]
    gdino_cfg = plan["gdino_config"]

    print(f"  test_set: {len(test_stems)} images")
    print(f"  templates: {len(templates)}")
    print(f"  bootstrap trials: {n_bootstrap}")

    # ----------------------------------------------------------
    # 2. Load execution order
    # ----------------------------------------------------------
    order_path = os.path.join(os.path.dirname(os.path.abspath(args.plan)), "order_baselines.txt")
    if not os.path.isfile(order_path):
        order_path = os.path.join(args.output_dir, "order_baselines.txt")
    if not os.path.isfile(order_path):
        raise FileNotFoundError(f"order_baselines.txt not found at {order_path}")

    execution_order = []
    with open(order_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            execution_order.append({
                "run_index": int(parts[0]),
                "trial": int(parts[1]),
                "method": parts[2],
            })
    print(f"  execution order: {len(execution_order)} runs")

    # ----------------------------------------------------------
    # 3. Load GT annotations
    # ----------------------------------------------------------
    print(f"Loading GT: {args.csv}")
    gt_ann = read_yolo_csv(args.csv)

    # ----------------------------------------------------------
    # 4. Resolve image paths for test_set
    # ----------------------------------------------------------
    stem_to_path = {}
    for stem in test_stems:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(args.images_dir, stem + ext)
            if os.path.isfile(p):
                stem_to_path[stem] = p
                break
    print(f"  resolved {len(stem_to_path)}/{len(test_stems)} test image paths")

    # ----------------------------------------------------------
    # 5. Initialize detectors
    # ----------------------------------------------------------
    detectors = {}
    model_mb  = {}          # GPU memory each detector allocates for its own weights

    if not args.skip_proposed:
        print("Initializing proposed detector (VAE latent TM)...")
        detectors["proposed"], model_mb["proposed"] = _cuda_mb(lambda: ProposedDetector(
            checkpoint_path=args.checkpoint,
            templates_info=templates,
            images_dir=args.images_dir,
            csv_path=args.csv,
            feature_source=proposed_cfg["feature_source"],
            roi_size=proposed_cfg["roi_size"],
            pool_ks=proposed_cfg["pool_ks"],
            thresh=proposed_cfg["thresh"],
            mode=proposed_cfg["mode"],
            maj_thresh=proposed_cfg["maj_thresh"],
            max_crops=proposed_cfg["max_crops"],
            resize_img=args.resize_img,
            device=args.device,
        ))
        print(f"  → {len(detectors['proposed'].template_bank)} templates loaded")

    if not args.skip_yolo:
        print(f"Initializing YOLO ({yolo_cfg['model']})...")
        detectors["yolo"], model_mb["yolo"] = _cuda_mb(lambda: YOLODetector(
            model_path=yolo_cfg["model"],
            conf_threshold=yolo_cfg["conf_threshold"],
            resize_img=args.resize_img,
            device=args.device,
        ))

    if not args.skip_gdino:
        print(f"Initializing Grounding DINO ({gdino_cfg['model']})...")
        detectors["grounding_dino"], model_mb["grounding_dino"] = _cuda_mb(lambda: GroundingDINODetector(
            model_id=gdino_cfg["model"],
            conf_threshold=gdino_cfg["conf_threshold"],
            text_prompt=gdino_cfg["text_prompt"],
            resize_img=args.resize_img,
            device=args.device,
        ))

    # ----------------------------------------------------------
    # 6. Open results file
    # ----------------------------------------------------------
    results_path = os.path.join(args.output_dir, "results_baselines.csv")
    csv_fields = [
        "run_index", "trial", "method",
        "mae", "rmse", "mape",
        "precision", "recall", "f1", "ap",
        "p_center", "r_center", "f1_center", "ap_center",
        "elapsed_s", "images", "s_per_image",
        "gpu_peak_mb", "gpu_reserved_mb", "gpu_delta_mb", "host_peak_mb",
    ]

    # Append only when there is something to append to.  Deciding on
    # --resume_from alone breaks the parallel split: each chunk writes into its
    # own output directory, so b (resume_from=1000) and c (resume_from=2000)
    # would open a file that does not exist yet in append mode and never write
    # the header.
    file_mode = "a" if (args.resume_from > 0 and os.path.isfile(results_path)) else "w"
    csv_fh = open(results_path, file_mode, newline="")
    writer = csv_mod.DictWriter(csv_fh, fieldnames=csv_fields)
    if file_mode == "w":
        writer.writeheader()

    # ----------------------------------------------------------
    # 7. Execute runs in randomized order
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # 6b. Warm-up and resident baseline
    # ----------------------------------------------------------
    # The first inference of each model allocates cuDNN workspaces and caches
    # that are not repeated.  Because the execution order is randomized, the
    # first occurrence of each method would otherwise report an inflated peak.
    warm_paths = [p for p in list(stem_to_path.values())[:3]]
    if warm_paths:
        print("Warming up detectors ...")
        for name, det in detectors.items():
            try:
                for wp in warm_paths:
                    det.detect_image(wp)
            except Exception as e:
                print(f"  WARN: warm-up failed for {name}: {e}")

    gpu_baseline_mb = _gpu_now_mb()      # resident cost of all loaded detectors
    print(f"\nResident GPU memory after loading all detectors: {gpu_baseline_mb:.1f} MB")
    for name in sorted(model_mb):
        print(f"    {name:<16s} own weights: {model_mb[name]:.1f} MB")

    cost_path = os.path.join(args.output_dir, "cost_models.csv")
    with open(cost_path, "w", newline="") as cf:
        cw = csv_mod.writer(cf)
        cw.writerow(["method", "model_mb", "gpu_baseline_mb"])
        for name in sorted(model_mb):
            cw.writerow([name, round(model_mb[name], 1), round(gpu_baseline_mb, 1)])
    print(f"Per-model memory → {cost_path}\n")

    t0_global = time.time()
    completed = 0
    skipped = 0

    print(f"\n{'='*70}")
    print(f"Starting execution ({len(execution_order)} total runs, "
          f"resume_from={args.resume_from})")
    print(f"{'='*70}\n")

    for run_idx, run in enumerate(execution_order):
        if run_idx < args.resume_from:
            skipped += 1
            continue
        if args.stop_at > 0 and run_idx >= args.stop_at:
            print(f"  reached --stop_at={args.stop_at}, stopping here.")
            break

        trial = run["trial"]
        method = run["method"]

        # Skip methods that are not loaded
        if method not in detectors:
            skipped += 1
            continue

        # Get bootstrapped image paths for this trial
        sample_indices = bootstrap_indices[trial]
        trial_img_paths = []
        for idx in sample_indices:
            stem = test_stems[idx]
            if stem in stem_to_path:
                trial_img_paths.append(stem_to_path[stem])

        if not trial_img_paths:
            print(f"  [{run_idx}] trial={trial} method={method}: NO IMAGES — skipping")
            skipped += 1
            continue

        # Run evaluation.  The peak must be reset here: without it the value
        # read back would be the maximum since process start, i.e. whatever the
        # heaviest method had reached earlier in the randomized order.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        metrics = evaluate_trial(
            detectors[method], trial_img_paths, gt_ann,
            args.resize_img, args.iou_threshold)
        elapsed = round(time.time() - t0, 2)
        gpu_peak, gpu_resv = _gpu_peak_mb()

        # Write result
        row = {
            "run_index": run_idx,
            "trial": trial,
            "method": method,
            "elapsed_s": elapsed,
            "images": len(trial_img_paths),
            "s_per_image": round(elapsed / max(len(trial_img_paths), 1), 5),
            "gpu_peak_mb": round(gpu_peak, 1),
            "gpu_reserved_mb": round(gpu_resv, 1),
            "gpu_delta_mb": round(gpu_peak - gpu_baseline_mb, 1),
            "host_peak_mb": round(_host_peak_mb(), 1),
        }
        for k, v in metrics.items():
            row[k] = round(v, 6)
        writer.writerow(row)
        csv_fh.flush()

        completed += 1
        if completed % 50 == 0 or completed <= 5:
            print(f"  [{run_idx}] trial={trial:4d} method={method:<16s}: "
                  f"MAE={metrics['mae']:.2f} F1c={metrics['f1_center']:.4f} "
                  f"({elapsed:.1f}s) [{completed}/{len(execution_order) - args.resume_from}]")

    csv_fh.close()
    total_time = time.time() - t0_global

    print(f"\n{'='*70}")
    print(f"Baseline comparison complete.")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Results:    {results_path}")
    print(f"  Host peak RSS: {_host_peak_mb():.0f} MB")
    if torch.cuda.is_available():
        print(f"  GPU resident baseline: {gpu_baseline_mb:.0f} MB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
