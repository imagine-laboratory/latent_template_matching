# experiments/count_from_latent.py
import os
import json
import argparse
import glob
from typing import Tuple, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time
import resource

from models.vae import VAE
from tools.utils import select_device, set_seed, create_directory

# ----------------------------
# IO helpers
# ----------------------------
def load_image_rgb(path: str, resize: int) -> np.ndarray:
    """
    Load an image from 'path', convert BGR->RGB, and resize to (resize, resize).
    Returns an RGB uint8 array with shape (H, W, 3).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
    return img

def to_tensor_chw01(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert an RGB uint8 image (H, W, 3) in [0,255] to a float32 tensor (C, H, W) in [0,1].
    """
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)

def standardize_channelwise_featmap(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Standardize a feature map per-channel over spatial dims: (z - mean)/std.
    z: (B, C, H, W)
    Returns a tensor with the same shape.
    """
    mean = z.mean(dim=(2,3), keepdim=True)
    std = z.std(dim=(2,3), keepdim=True)
    return (z - mean) / (std + eps)

def fmap_to_rgb(fmap: torch.Tensor) -> np.ndarray:
    """
    Convert a feature map (1, C, H, W) into an RGB image.
    Uses PCA down to 3 components to visualize highly-dimensional spaces.
    No resizing is applied; the output matches the latent's spatial resolution.
    """
    fmap_squeeze = fmap.squeeze(0).float() # (C, H, W)
    C, H, W = fmap_squeeze.shape
    if C >= 3:
        x = fmap_squeeze.view(C, -1).T # (H*W, C)
        x = x - x.mean(dim=0)
        # Using PCA to reduce to 3 channels for RGB
        U, S, V = torch.pca_lowrank(x, q=3)
        pca_img = U.view(H, W, 3).cpu().numpy()
        # Min-Max Normalization to [0,1]
        pca_min = pca_img.min(axis=(0,1), keepdims=True)
        pca_max = pca_img.max(axis=(0,1), keepdims=True)
        pca_img = (pca_img - pca_min) / (pca_max - pca_min + 1e-8)
        pca_img = (pca_img * 255).astype(np.uint8)
    else:
        # Fallback if less than 3 channels
        mean_img = fmap_squeeze.mean(dim=0).cpu().numpy()
        mean_img = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)
        mean_img = (mean_img * 255).astype(np.uint8)
        pca_img = cv2.cvtColor(mean_img, cv2.COLOR_GRAY2RGB)
    
    # We return the latent RGB representation at its exact original resolution
    return pca_img

def overlay_heatmap(image_rgb: np.ndarray, heat_upsampled01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Colorize a heatmap (float [0,1]) and blend it over the original image.
    image_rgb: uint8 (H, W, 3)
    heat: float (H, W) in [0,1]
    alpha: blending factor for the heatmap.
    """
    heat_uint8 = (np.clip(heat_upsampled01, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heat_rgb, alpha, 0)
    return overlay

def draw_peaks_on_overlay(overlay: np.ndarray, peaks_hw: np.ndarray, scale_y: float, scale_x: float, radius: int = 15, thickness: int = 2) -> np.ndarray:
    """
    Draw green circles on the overlay at predicted peak locations.
    peaks_hw: array of (h, w) indices in the downsampled response map.
    down_factor: scale factor to map peak indices to image pixels (e.g., 256/32 = 8).
    """
    out = overlay.copy()
    for (ph, pw) in peaks_hw:
        cy = int((ph + 0.5) * scale_y)
        cx = int((pw + 0.5) * scale_x)
        cv2.circle(out, (cx, cy), radius=radius, color=(0,255,0), thickness=thickness)  # predicted
    return out

def draw_pred_boxes_on_overlay(overlay: np.ndarray,
                               boxes: List[Tuple[float,float,float,float]],
                               thickness: int = 2) -> np.ndarray:
    """
    Draw green predicted bounding boxes on the overlay.
    boxes: list of (x1, y1, x2, y2) in pixel coordinates.
    """
    out = overlay.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      color=(0, 255, 0), thickness=thickness)
    return out

def draw_gt_boxes(overlay: np.ndarray,
                  boxes: List[Tuple[float,float,float,float]],
                  proto_box: Tuple[float,float,float,float] = None,
                  thickness: int = 2) -> np.ndarray:
    """
    Draw GT bounding boxes.
    - Red = all ground truth boxes
    - Blue = the GT box used for latent prototype (if provided)
    The proto_box comparison uses a small absolute tolerance to match floats.
    """
    out = overlay.copy()
    for (x1, y1, x2, y2) in boxes:
        color = (255, 0, 0)  # red
        if proto_box is not None:
            # compare with small tolerance
            if all(abs(a - b) < 1e-2 for a, b in zip([x1,y1,x2,y2], proto_box)):
                color = (0, 0, 255)  # blue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out

# ----------------------------
# CSV helpers
# ----------------------------
def read_yolo_csv(csv_path: str) -> Dict[str, List[Tuple[float,float,float,float]]]:
    """
    Read a YOLO-style CSV with columns:
      filename, class, cx, cy, w, h
    Returns a dict: {stem: [(cx,cy,w,h), ...], ...}
    where 'stem' is the filename without extension (matching image stems).
    """
    mapping: Dict[str, List[Tuple[float,float,float,float]]] = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            stem = os.path.splitext(row["filename"])[0]  # strip .txt
            cx, cy, w, h = map(float, (row["cx"], row["cy"], row["w"], row["h"]))
            mapping.setdefault(stem, []).append((cx, cy, w, h))
    return mapping

def yolo_to_xyxy_pixels(cx, cy, w, h, W, H):
    """
    Convert normalized YOLO (cx, cy, w, h) in [0,1] to pixel-space (x1,y1,x2,y2).
    W, H are the image width and height in pixels.
    """
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return (x1, y1, x2, y2)

# ----------------------------
# Feature selection
# ----------------------------
def _get_feature_map(
    model: VAE,
    x: torch.Tensor,
    feature_source: Literal["feats", "z", "mu"],
    noise: torch.Tensor
) -> torch.Tensor:
    """
    Returns the spatial feature map to correlate with the prototype.
    - "feats": last spatial map before μ/σ projection, shape (B, 512, H/8, W/8)
    - "z":     reparameterized latent (scaled by 0.18215), shape (B, 4, H/8, W/8)
    - "mu":    mean (μ) of the bottleneck distribution, shape (B, 4, H/8, W/8)
    The encoder is called with return_features accordingly.
    """
    model.eval()
    with torch.no_grad():
        if feature_source == "feats":
            # encoder requires noise; pass zeros to be safe/deterministic
            z, mu, logvar, feats = model.encoder(x, noise, return_features=True)
            return feats
        else:
            z, mu, logvar = model.encoder(x, noise, return_features=False)
            if feature_source == "z":
                return z
            elif feature_source == "mu":
                return mu
            else:
                raise ValueError(f"Unknown feature_source={feature_source}")

# ----------------------------
# Correlation + peak picking
def compute_response_map(z: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """
    Compute a normalized cross-correlation response map using conv2d.
    Steps:
      1) Standardize the feature map per-channel over spatial dims.
      2) Standardize the prototype per-channel over its (k,k) spatial dims.
      3) Convolve z_norm with proto as a single filter (1 output channel).
      4) Min-max normalize the response to [0,1] per-image.
    Args:
      z:     (1, C, h, w) feature map from the image
      proto: (C, k, k) prototype kernel
    Returns:
      resp01: (1,1,h,w) normalized response map in [0,1]
    """
    device = z.device
    B, C, h, w = z.shape           # B=1, C=channels, h/w=spatial size of fmap (e.g., 32x32 for 256/8)
    Ck, k, k2 = proto.shape        # proto has shape (C, k, k); Ck must equal C; k2==k
    assert C == Ck, f"Channel mismatch: z has {C}, prototype has {Ck}"

    # Standardize the input features channel-wise over spatial dims.
    # z_norm: (1, C, h, w)
    # mean/std: (1, C, 1, 1) broadcast over (h, w)
    z_norm = standardize_channelwise_featmap(z)

    # Standardize prototype per-channel across its spatial support.
    # proto before norm: (C, k, k)
    # proto.mean/std:    (C, 1, 1) broadcast over (k, k)
    # proto after norm:  (C, k, k)
    proto = (proto - proto.mean(dim=(1,2), keepdim=True)) / (proto.std(dim=(1,2), keepdim=True) + 1e-6)

    # Prepare weight as a conv filter: (out_channels=1, in_channels=C, k, k)
    # weight: (1, C, k, k)
    weight = proto.unsqueeze(0).to(device)  # (1, C, k, k)

    # Same padding to keep spatial size: pad = k//2 (assumes odd k)
    pad = k // 2

    # Convolution:
    # input:  z_norm  (1, C, h, w)
    # weight: weight  (1, C, k, k)
    # stride=1, padding=pad  -> output: (1, 1, h, w)
    resp = torch.nn.functional.conv2d(z_norm, weight=weight, stride=1, padding=pad)  # (1, 1, h, w)

    # Per-image (per-sample) min-max normalization to [0,1]
    # resp_min/max: (1, 1, 1, 1) reduced over (h, w)
    # resp01:       (1, 1, h, w)
    resp_min = resp.amin(dim=(2,3), keepdim=True)
    resp_max = resp.amax(dim=(2,3), keepdim=True)
    resp01 = (resp - resp_min) / (resp_max - resp_min + 1e-8)
    return resp01


def merge_close_peaks(peaks_hw: np.ndarray,
                      vals: np.ndarray,
                      scale_y: float,
                      scale_x: float,
                      min_dist_px: float,
                      base_box_size: float = 30.0
                      ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float,float,float,float]]]:
    """
    Greedy NMS-by-distance in PIXEL space.
    Two peaks whose centers are closer than `min_dist_px` (e.g. when their drawn
    circles overlap) are merged: the one with the highest response value is kept
    and the rest are suppressed. This avoids drawing several stacked circles for
    the same object — typical in `majority_voting` where many templates can
    activate slightly shifted positions around the same instance.

    The bounding box of each kept detection is the envelope of all merged
    members' individual boxes (center ± half base_box_size). More merged peaks
    spread over a wider area → larger box.

    Args:
      peaks_hw:      (N, 2) integer array with (h, w) indices on the latent grid.
      vals:          (N,) float array with each peak's response value.
      scale_y/x:     pixel-per-grid-cell factors (image_size / grid_size).
      min_dist_px:   minimum allowed distance between two kept peaks, in pixels.
      base_box_size: side length of each individual peak's square before merging.
    Returns:
      kept_peaks_hw, kept_vals, kept_boxes
      kept_boxes: list of (x1,y1,x2,y2) in pixel space (one per kept peak).
    """
    half = base_box_size / 2.0

    # Convert grid indices to pixel coordinates (cell centers).
    ph = peaks_hw[:, 0].astype(np.float32)
    pw = peaks_hw[:, 1].astype(np.float32)
    cy = (ph + 0.5) * float(scale_y)
    cx = (pw + 0.5) * float(scale_x)

    if peaks_hw.shape[0] <= 1 or min_dist_px <= 0:
        boxes = [(float(cx[i] - half), float(cy[i] - half),
                  float(cx[i] + half), float(cy[i] + half))
                 for i in range(peaks_hw.shape[0])]
        return peaks_hw, vals, boxes

    # Process peaks from strongest to weakest; suppress weaker ones within min_dist.
    order = np.argsort(-vals)
    suppressed = np.zeros(peaks_hw.shape[0], dtype=bool)
    keep_idx: List[int] = []
    groups: Dict[int, List[int]] = {}   # kept_idx -> list of all member indices
    min_d2 = float(min_dist_px) * float(min_dist_px)

    for i in order:
        if suppressed[i]:
            continue
        keep_idx.append(int(i))
        # Find every still-alive peak within min_dist_px of i.
        dy = cy - cy[i]
        dx = cx - cx[i]
        d2 = dy * dy + dx * dx
        nearby = (~suppressed) & (d2 < min_d2)
        groups[int(i)] = list(np.where(nearby)[0])
        suppressed |= nearby
        suppressed[i] = True

    keep_idx_arr = np.array(keep_idx, dtype=np.int64)

    # Build bounding boxes: envelope of all member peaks' individual boxes.
    kept_boxes: List[Tuple[float,float,float,float]] = []
    for ki in keep_idx:
        members = groups[ki]
        x1 = min(float(cx[m] - half) for m in members)
        y1 = min(float(cy[m] - half) for m in members)
        x2 = max(float(cx[m] + half) for m in members)
        y2 = max(float(cy[m] + half) for m in members)
        kept_boxes.append((x1, y1, x2, y2))

    return peaks_hw[keep_idx_arr], vals[keep_idx_arr], kept_boxes


# ----------------------------
# Detection metrics helpers
# ----------------------------
def peaks_to_boxes(peaks_hw: np.ndarray, scale_y: float, scale_x: float,
                   box_size: float) -> List[Tuple[float,float,float,float]]:
    """
    Convert peak grid indices to pixel-space square bounding boxes (x1,y1,x2,y2).
    Each box is a square of side `box_size` centered at the peak's pixel location.
    """
    boxes = []
    half = box_size / 2.0
    for (ph, pw) in peaks_hw:
        cy = (ph + 0.5) * scale_y
        cx = (pw + 0.5) * scale_x
        boxes.append((cx - half, cy - half, cx + half, cy + half))
    return boxes


def measure_peak_spread(resp_map: np.ndarray, ph: int, pw: int, half_max_frac: float = 0.5) -> Tuple[float, float]:
    """
    Measure the half-width at half-maximum (HWHM) of a peak in the response map
    along the vertical and horizontal axes.

    Args:
      resp_map: 2D numpy array (h, w) — raw response map (before NMS).
      ph, pw: integer grid coordinates of the peak.
      half_max_frac: fraction of the peak value used as the threshold (default 0.5).

    Returns:
      (spread_h, spread_w): HWHM in grid cells along each axis.
    """
    h, w = resp_map.shape
    peak_val = resp_map[ph, pw]
    threshold = peak_val * half_max_frac

    # Measure spread along h-axis (vertical)
    spread_h = 0.5  # minimum: half a cell
    for dh in range(1, h):
        if ph + dh >= h or resp_map[ph + dh, pw] < threshold:
            spread_h = float(dh)
            break
    else:
        spread_h = float(h - ph)
    # Also check negative direction and average
    spread_h_neg = 0.5
    for dh in range(1, h):
        if ph - dh < 0 or resp_map[ph - dh, pw] < threshold:
            spread_h_neg = float(dh)
            break
    else:
        spread_h_neg = float(ph + 1)
    spread_h = (spread_h + spread_h_neg) / 2.0

    # Measure spread along w-axis (horizontal)
    spread_w = 0.5
    for dw in range(1, w):
        if pw + dw >= w or resp_map[ph, pw + dw] < threshold:
            spread_w = float(dw)
            break
    else:
        spread_w = float(w - pw)
    spread_w_neg = 0.5
    for dw in range(1, w):
        if pw - dw < 0 or resp_map[ph, pw - dw] < threshold:
            spread_w_neg = float(dw)
            break
    else:
        spread_w_neg = float(pw + 1)
    spread_w = (spread_w + spread_w_neg) / 2.0

    return spread_h, spread_w


def peaks_to_adaptive_boxes(
    peaks_hw: np.ndarray,
    resp_map: np.ndarray,
    scale_y: float,
    scale_x: float,
    roi_size: int,
    min_box_factor: float = 0.5,
    max_box_factor: float = 2.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Hybrid option 2+3: build adaptive predicted boxes using the peak spread
    in the response map, with roi_size as a floor.

    For each peak:
      - Measure HWHM in h and w directions on the response map.
      - Convert spread to pixels: box_h = 2 * spread_h * scale_y, box_w = 2 * spread_w * scale_x
      - Floor: roi_size * scale (the template footprint in pixel space)
      - Ceiling: max_box_factor * floor

    Args:
      peaks_hw: (N, 2) array of (h, w) grid indices.
      resp_map: 2D numpy array (h, w) — raw response map before NMS.
      scale_y, scale_x: grid-to-pixel scale factors.
      roi_size: template spatial size in grid cells.
      min_box_factor: multiplier for the floor (1.0 = roi_size * scale).
      max_box_factor: max multiplier over the floor.

    Returns:
      List of (x1, y1, x2, y2) boxes in pixel space.
    """
    boxes = []
    floor_h = roi_size * scale_y * min_box_factor
    floor_w = roi_size * scale_x * min_box_factor
    ceil_h = floor_h * max_box_factor
    ceil_w = floor_w * max_box_factor

    for (ph, pw) in peaks_hw:
        cy = (ph + 0.5) * scale_y
        cx = (pw + 0.5) * scale_x

        spread_h, spread_w = measure_peak_spread(resp_map, int(ph), int(pw))

        # Convert spread to pixel-space box dimensions
        box_h = 2.0 * spread_h * scale_y
        box_w = 2.0 * spread_w * scale_x

        # Clamp between floor and ceiling
        box_h = max(floor_h, min(box_h, ceil_h))
        box_w = max(floor_w, min(box_w, ceil_w))

        x1 = cx - box_w / 2.0
        y1 = cy - box_h / 2.0
        x2 = cx + box_w / 2.0
        y2 = cy + box_h / 2.0
        boxes.append((x1, y1, x2, y2))

    return boxes


def compute_iou(box_a: Tuple[float,float,float,float],
                box_b: Tuple[float,float,float,float]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(
    pred_boxes: List[Tuple[float,float,float,float]],
    gt_boxes: List[Tuple[float,float,float,float]],
    iou_threshold: float
) -> Tuple[int, int, int, List[int]]:
    """
    Greedy matching of predicted boxes to GT boxes by IoU.
    Returns (tp, fp, fn, tp_flags) where tp_flags[i] = 1 if pred i is a TP.
    """
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


def match_predictions_to_gt_center(
    pred_centers: List[Tuple[float, float]],
    gt_boxes: List[Tuple[float, float, float, float]],
    sigma: float = 0.5,
) -> Tuple[int, int, int, List[int]]:
    """
    Center-to-center distance matching.
    A prediction is TP if its center falls within sigma * diagonal of the
    closest unmatched GT box.

    Args:
      pred_centers: list of (cx, cy) predicted centers in pixel space.
      gt_boxes: list of (x1, y1, x2, y2) GT boxes in pixel space.
      sigma: fraction of GT diagonal used as matching radius.

    Returns:
      (tp, fp, fn, tp_flags) same format as match_predictions_to_gt.
    """
    n_pred = len(pred_centers)
    n_gt = len(gt_boxes)
    tp_flags = [0] * n_pred
    if n_pred == 0:
        return 0, 0, n_gt, tp_flags
    if n_gt == 0:
        return 0, n_pred, 0, tp_flags

    # Compute GT centers and matching radii
    gt_centers = []
    gt_radii = []
    for (x1, y1, x2, y2) in gt_boxes:
        gcx = (x1 + x2) / 2.0
        gcy = (y1 + y2) / 2.0
        diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        gt_centers.append((gcx, gcy))
        gt_radii.append(sigma * diag)

    # Build (distance, pred_idx, gt_idx) pairs for valid matches
    pairs = []
    for pi, (pcx, pcy) in enumerate(pred_centers):
        for gi, ((gcx, gcy), radius) in enumerate(zip(gt_centers, gt_radii)):
            dist = np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
            if dist <= radius:
                pairs.append((dist, pi, gi))
    pairs.sort(key=lambda x: x[0])  # greedy: closest first

    matched_pred = set()
    matched_gt = set()
    for dist_val, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            tp_flags[pi] = 1

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn, tp_flags


def compute_ap(scores: np.ndarray, tp_flags: np.ndarray, total_gt: int) -> float:
    """
    Compute Average Precision from predictions sorted by confidence.
    Uses all-point interpolation (VOC 2010+ style).
    """
    if total_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp_sorted = tp_flags[order].astype(np.float64)
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1.0 - tp_sorted)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Prepend sentinel (recall=0, precision=1)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Area under the P-R curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def peak_pick(resp01: torch.Tensor, thresh: float = 0.6, pool_ks: int = 3):
    """
    Detect local maxima in the response map using max-pooling non-maximum suppression.
    Args:
      resp01: tensor (1,1,h,w) in [0,1]   -> normalized response map
      thresh: keep peaks with response >= thresh
      pool_ks: window size for local max pooling (e.g., 3 or 5)
    Returns:
      peaks_hw: ndarray of shape (N, 2) with (h, w) indices (integers)
      vals:     ndarray of shape (N,) with corresponding response values
    """
    # Apply max-pooling to find the local maximum value in each neighborhood.
    # resp01: (1, 1, h, w)
    # pooled: (1, 1, h, w) -> each pixel contains the local max in its pool_ks×pool_ks neighborhood
    pooled = torch.nn.functional.max_pool2d(resp01, kernel_size=pool_ks, stride=1, padding=pool_ks//2)

    # Boolean mask of peak candidates:
    # - resp01 >= pooled  -> only true where the pixel is equal to the local max
    # - resp01 >= thresh  -> filter by minimum response threshold
    # is_peak: (1, 1, h, w), dtype=bool
    is_peak = (resp01 >= pooled) & (resp01 >= thresh)

    # Get coordinates of all peaks
    # peaks: (N, 4) tensor, each row = [batch_idx, channel_idx, h_idx, w_idx]
    peaks = is_peak.nonzero(as_tuple=False)

    if peaks.numel() == 0:
        # If no peaks found, return empty arrays with correct shapes
        return np.zeros((0,2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    # Extract row/col indices of peaks
    # peaks[:, 2] -> h indices, shape (N,)
    # peaks[:, 3] -> w indices, shape (N,)
    h_idx = peaks[:, 2].cpu().numpy()   # (N,)
    w_idx = peaks[:, 3].cpu().numpy()   # (N,)

    # Gather corresponding response values at those positions
    # resp01[0,0,h_idx,w_idx] -> tensor (N,)
    # vals: numpy array (N,)
    vals = resp01[0,0,h_idx, w_idx].cpu().numpy()

    # Stack (h_idx, w_idx) into array of shape (N,2)
    peaks_hw = np.stack([h_idx, w_idx], axis=1)

    return peaks_hw, vals


# ----------------------------
# Main
# ----------------------------
def main():
    """
    Entry point for counting objects by correlating a latent prototype with feature maps.
    Produces visual overlays and (optionally) MAE/RMSE metrics when GT CSV is provided.
    """
    ap = argparse.ArgumentParser(description="Count pineapples by matching latent template.")
    ap.add_argument("--images_dir", required=True, help="Folder with .png images (also used to resolve --image_stem)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prototype_pt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", default=None, help="Path to FULL_UNIFIED_labels.csv for ground truth (optional)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resize_img", type=int, default=256)
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--pool_ks", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help="Max images to evaluate (0 = all).")
    ap.add_argument("--only_annotated", action="store_true", help="Evaluate only images present in CSV")
    ap.add_argument("--no_save_images", action="store_true",
                    help="Skip the per-image overlay PDFs and report metrics only")
    # single-image options
    ap.add_argument("--use_prototype_image", action="store_true",
                    help="Use the same image recorded in latent_prototype_meta.json")
    ap.add_argument("--image_path", type=str, default=None, help="Run on a single explicit image path (.png)")
    ap.add_argument("--image_stem", type=str, default=None, help="Run on a single stem within images_dir")
    # NEW: feature source for inference
    ap.add_argument("--feature_source", type=str, default="auto",
                    choices=["auto", "feats", "z", "mu"],
                    help="Which map to correlate at inference. 'auto' uses the prototype's metadata if available.")
    # NEW: mode for template usage
    ap.add_argument("--mode", type=str, default="single", choices=["single", "average", "majority_voting"],
                    help="Template matching mode: 'single' (default), 'average', 'majority_voting'")
    ap.add_argument("--majority_threshold", type=float, default=0.35,
                    help="Fraction of templates that must agree for majority_voting mode (default: 0.35)")
    ap.add_argument("--merge_radius", type=float, default=30.0,
                    help="Minimum distance (in PIXELS of the resized image) between two kept peaks. "
                         "Peaks whose drawn circles overlap or are closer than this are merged into "
                         "the strongest one. Set to 0 to disable. Default: 30 (= 2x default draw radius).")
    ap.add_argument("--pred_box_size", type=float, default=30.0,
                    help="Side length (pixels) of predicted bounding boxes for detection metrics. "
                         "Squares are centered at each detected peak. Default: 30.")
    ap.add_argument("--roi_size", type=int, default=7,
                    help="Template spatial size in grid cells. Used as floor for adaptive box sizing. Default: 7.")
    ap.add_argument("--iou_threshold", type=float, default=0.15,
                    help="IoU threshold for matching predicted boxes to GT boxes. Default: 0.15.")

    args = ap.parse_args()

    # --- cost instrumentation -------------------------------------------
    # Peak GPU memory must be measured in THIS process: run_grid_search.py
    # launches this script as a subprocess and therefore owns no CUDA context.
    # reset_peak_memory_stats() is essential; without it the peak reported is
    # the maximum since process start, which would carry over across stages.
    _t_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # ---------------------------------------------------------------------

    # Set device, seeds, and make sure output directory exists.
    device = select_device(args.device)
    set_seed(args.seed, deterministic=True, cudnn_benchmark=False)
    create_directory(args.output_dir)

    # Load VAE model and checkpoint weights (strict=False to be tolerant to missing keys).
    model = VAE().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")


    # Load prototype(s) and meta depending on mode
    proto_dir = os.path.dirname(args.prototype_pt)
    meta_json = os.path.join(proto_dir, "latent_prototype_meta.json")
    proto_meta = None
    if args.mode == "majority_voting":
        # Load template bank (list of tensors)
        proto_bank = torch.load(args.prototype_pt, map_location="cpu")
        if isinstance(proto_bank, list):
            proto_bank = [p.float() for p in proto_bank]
        else:
            raise RuntimeError("Template bank file must be a list of tensors.")
        if os.path.isfile(meta_json):
            with open(meta_json, "r") as f:
                proto_meta = json.load(f)
            print(f"Loaded prototype meta: {meta_json}")
            # If proto_meta is a list, use the first element for single-image logic
            if isinstance(proto_meta, list) and len(proto_meta) > 0:
                proto_meta_single = proto_meta[0]
            else:
                proto_meta_single = proto_meta
        else:
            print("WARNING: Prototype metadata not found; running without meta checks.")
            proto_meta_single = None
        proto = proto_bank[0]  # for channel check only
    else:
        # single or average mode: load one tensor (or first from list)
        proto_loaded = torch.load(args.prototype_pt, map_location="cpu")
        if isinstance(proto_loaded, list):
            if len(proto_loaded) == 0:
                raise RuntimeError("Prototype file is an empty list.")
            proto = proto_loaded[0].float()
        else:
            proto = proto_loaded.float()
        if os.path.isfile(meta_json):
            with open(meta_json, "r") as f:
                proto_meta = json.load(f)
            print(f"Loaded prototype meta: {meta_json}")
            proto_meta_single = proto_meta
        else:
            print("WARNING: Prototype metadata not found; running without meta checks.")
            proto_meta = None
            proto_meta_single = None

    # Decide which feature source to use for correlation.
    # - "auto": respect the feature_source stored in the prototype meta if present, else default to "z".
    # - explicit choice: use the CLI value.
    if args.feature_source == "auto":
        # Ensure proto_meta is a dict, not a list
        meta_for_feature = proto_meta
        if isinstance(meta_for_feature, list):
            if len(meta_for_feature) > 0:
                meta_for_feature = meta_for_feature[0]
            else:
                meta_for_feature = None
        if meta_for_feature and "feature_source" in meta_for_feature:
            feature_source = str(meta_for_feature["feature_source"])
            print(f"[auto] Using feature_source from meta: {feature_source}")
        else:
            feature_source = "z"  # fallback to legacy behavior
            print(f"[auto] No meta feature_source; falling back to: {feature_source}")
    else:
        feature_source = args.feature_source
        print(f"Using feature_source from CLI: {feature_source}")

    # Build list of candidate images to process and read the prototype's source box if available.
    img_paths: List[str] = []
    proto_box = None

    # 1) exact image via prototype meta (use the exact same image used to extract the prototype)
    if args.use_prototype_image:
        if not proto_meta_single:
            raise FileNotFoundError("--use_prototype_image is set but no latent_prototype_meta.json was found.")
        # Ensure proto_meta_single is a dict, not a list
        if isinstance(proto_meta_single, list):
            if len(proto_meta_single) == 0:
                raise FileNotFoundError("Prototype metadata list is empty.")
            proto_meta_single = proto_meta_single[0]
        meta_img_path = proto_meta_single.get("image_path")
        if meta_img_path is None or not os.path.isfile(meta_img_path):
            # If meta stores only a stem, try resolving it inside images_dir.
            stem = os.path.splitext(os.path.basename(proto_meta_single.get("image_path","")))[0]
            candidate = os.path.join(args.images_dir, stem + ".png")
            if not os.path.isfile(candidate):
                raise FileNotFoundError("Could not resolve prototype image from metadata.")
            meta_img_path = candidate
        if "box_xyxy" in proto_meta_single:
            proto_box = tuple(proto_meta_single["box_xyxy"])
            print(f"Prototype was extracted from box: {proto_box}")
        img_paths = [meta_img_path]

    # 2) explicit single image path
    elif args.image_path is not None:
        if not os.path.isfile(args.image_path):
            raise FileNotFoundError(f"--image_path not found: {args.image_path}")
        img_paths = [args.image_path]

    # 3) specific stem within images_dir (resolve to images_dir/<stem>.png)
    elif args.image_stem is not None:
        candidate = os.path.join(args.images_dir, args.image_stem + ".png")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"--image_stem '{args.image_stem}' not found in {args.images_dir}")
        img_paths = [candidate]

    # 4) fallback: batch process .png files in images_dir (optionally limited by --limit)
    else:
        img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
        
    # Optionally load GT annotations from CSV and filter images if --only_annotated is set.
    gt_ann = {}
    if args.csv:
        gt_ann = read_yolo_csv(args.csv)
        if args.only_annotated:
            stems_with_ann = set(gt_ann.keys())
            img_paths = [p for p in img_paths if os.path.splitext(os.path.basename(p))[0] in stems_with_ann]

    # Apply --limit AFTER annotation filtering (only meaningful for the batch fallback).
    # Single-image branches (use_prototype_image / image_path / image_stem) already
    # produced 1-element lists, so this is effectively a no-op for them.
    if args.limit > 0 and len(img_paths) > args.limit:
        img_paths = img_paths[:args.limit]

    # Convenience: cache prototype channel count to check compatibility later.
    proto_C = int(proto.shape[0])

    # Process each image: forward pass -> correlation -> peak picking -> visualization -> metrics
    down_factor = args.resize_img // 8  # used to map latent grid coords to image pixels
    preds, gts = [], []
    # Detection metrics accumulators (IoU-based)
    total_tp, total_fp, total_fn = 0, 0, 0
    all_det_scores: List[float] = []
    all_det_tp_flags: List[int] = []
    total_gt_count = 0
    # Center-to-center distance metrics accumulators
    total_tp_center, total_fp_center, total_fn_center = 0, 0, 0

    # Per-image results CSV
    per_image_csv_path = os.path.join(args.output_dir, "results.csv")
    per_image_fields = [
        "stem", "pred_count", "gt_count", "abs_error",
        "tp_iou", "fp_iou", "fn_iou", "precision_iou", "recall_iou", "f1_iou",
        "tp_center", "fp_center", "fn_center", "precision_center", "recall_center", "f1_center",
    ]
    per_image_csv_fh = open(per_image_csv_path, "w", newline="")
    per_image_writer = csv.DictWriter(per_image_csv_fh, fieldnames=per_image_fields)
    per_image_writer.writeheader()

    for path in img_paths:
        # Load and tensorize image
        img = load_image_rgb(path, resize=args.resize_img)
        x = to_tensor_chw01(img).unsqueeze(0).to(device)

        _, _, H, W = x.shape
        zH, zW = H // 8, W // 8
        # Deterministic noise for feats/mu; random noise for z if desired (to reflect sampling).
        if feature_source in ("feats", "mu"):
            noise = torch.zeros((1, 4, zH, zW), device=device)
        else:
            noise = torch.randn((1, 4, zH, zW), device=device)

        # Extract the chosen feature map: feats / z / mu
        with torch.no_grad():
            fmap = _get_feature_map(model, x, feature_source=feature_source, noise=noise)  # (1,C,h,w)

        # Early, explicit check that prototype and fmap have the same channel count.
        if fmap.shape[1] != proto_C:
            raise RuntimeError(
                f"Prototype channels ({proto_C}) != feature map channels ({fmap.shape[1]}). "
                f"Ensure extract and count use the SAME --feature_source. "
                f"(Prototype meta feature_source={proto_meta[0].get('feature_source') if isinstance(proto_meta, list) and len(proto_meta) > 0 else proto_meta.get('feature_source') if isinstance(proto_meta, dict) else 'unknown'})"
            )

        # Cross-correlate feature map with prototype(s) and detect peaks.
        if args.mode == "single":
            # Use the first prototype (default behavior)
            resp01 = compute_response_map(fmap, proto.to(device))
            peaks_hw, peak_vals = peak_pick(resp01, thresh=args.thresh, pool_ks=args.pool_ks)
            count_pred = int(peaks_hw.shape[0])
            pred_boxes = None  # computed after scale_y/x are known

        elif args.mode == "average":
            # Use the average prototype (already loaded as proto)
            resp01 = compute_response_map(fmap, proto.to(device))
            peaks_hw, peak_vals = peak_pick(resp01, thresh=args.thresh, pool_ks=args.pool_ks)
            count_pred = int(peaks_hw.shape[0])
            pred_boxes = None  # computed after scale_y/x are known

        elif args.mode == "majority_voting":
            # Agreement-based majority voting (detector and visualization unified):
            # 1) For each template, compute its full response map and binarize at `thresh`.
            # 2) Sum the binary masks → "agreement map" where each pixel value = number of
            #    templates that activate at that location.
            # 3) Normalize by N → values in [0, 1] = fraction of templates that agree.
            # 4) Run NMS + threshold (`majority_threshold`) on this map to extract detections.
            #
            # Rationale: a single dense map is both the decision surface and the
            # visualization. Detection and overlay are guaranteed to be consistent.
            h, w = fmap.shape[2:]
            N = len(proto_bank)

            # Dense agreement map: sum of per-template binary activation masks.
            # Also accumulate mean correlation (continuous) for spread-based box sizing.
            # Spatial merging of nearby detections is handled downstream by
            # peak_pick (NMS) + merge_close_peaks.
            agreement = torch.zeros((h, w), device=device, dtype=torch.float32)
            mean_corr = torch.zeros((h, w), device=device, dtype=torch.float32)
            for proto_i in proto_bank:
                resp01_i = compute_response_map(fmap, proto_i.to(device))  # (1,1,h,w)
                vote = (resp01_i >= args.thresh).float()                   # (1,1,h,w), {0,1}
                agreement += vote[0, 0]
                mean_corr += resp01_i[0, 0]

            # Normalize to [0, 1] (fraction of templates agreeing per pixel).
            agreement_norm = agreement / max(N, 1)
            mean_corr = mean_corr / max(N, 1)  # continuous [0,1] mean correlation

            # NMS + majority threshold on the agreement map itself.
            agr_t = agreement_norm.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
            peaks_hw, peak_vals = peak_pick(agr_t, thresh=args.majority_threshold, pool_ks=args.pool_ks)
            # Merge peaks whose drawn circles would overlap (or are too close in pixels).
            # This prevents the over-counting / "stacked circles" effect of voting,
            # where many templates fire slightly off-center on the same instance.
            # (Only needed in majority_voting; in single/average modes pool_ks already
            #  performs the NMS on a smooth response map.)
            _scale_y_merge = args.resize_img / float(h)
            _scale_x_merge = args.resize_img / float(w)
            peaks_hw, peak_vals, _merge_boxes = merge_close_peaks(
                peaks_hw, peak_vals, _scale_y_merge, _scale_x_merge,
                args.merge_radius, base_box_size=args.pred_box_size
            )
            count_pred = int(peaks_hw.shape[0])

            # Build adaptive boxes using mean correlation map (continuous, smoother)
            # for spread measurement. The mean_corr has broader peaks than the
            # quasi-binary agreement map, giving better object size estimates.
            mean_corr_np = mean_corr.cpu().numpy()  # (h, w)
            pred_boxes = peaks_to_adaptive_boxes(
                peaks_hw, mean_corr_np, _scale_y_merge, _scale_x_merge,
                roi_size=args.roi_size,
            )

            # The same agreement map is also the heatmap shown to the user.
            resp01 = agr_t

        # Compute scale factors from response map to image pixels (used for boxes & viz)
        _, _, h_resp, w_resp = resp01.shape
        scale_y = args.resize_img / float(h_resp)
        scale_x = args.resize_img / float(w_resp)

        # For single/average modes, pred_boxes are not yet computed (no merging).
        # Build adaptive boxes from peaks using spread measurement (hybrid option 2+3).
        if pred_boxes is None:
            resp_np_raw = resp01[0, 0].cpu().numpy()
            pred_boxes = peaks_to_adaptive_boxes(
                peaks_hw, resp_np_raw, scale_y, scale_x,
                roi_size=args.roi_size,
            )

        # If GT annotations are available for this image, convert them to pixel-space.
        stem = os.path.splitext(os.path.basename(path))[0]
        gt_boxes = []
        if args.csv and stem in gt_ann:
            for (cx, cy, w, h) in gt_ann[stem]:
                gt_boxes.append(yolo_to_xyxy_pixels(cx, cy, w, h, args.resize_img, args.resize_img))
        count_gt = len(gt_boxes) if args.csv else -1  # -1 means unknown / not computing metrics

        # Accumulate metrics only when GT exists.
        if count_gt >= 0:
            preds.append(count_pred)
            gts.append(count_gt)

            # Detection metrics: match predicted boxes to GT (IoU-based)
            tp, fp, fn, tp_flags_img = match_predictions_to_gt(
                pred_boxes, gt_boxes, args.iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt_count += len(gt_boxes)
            all_det_scores.extend(peak_vals.tolist())
            all_det_tp_flags.extend(tp_flags_img)

            # Center-to-center distance metrics
            pred_centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for (x1, y1, x2, y2) in pred_boxes]
            tp_c, fp_c, fn_c, _ = match_predictions_to_gt_center(
                pred_centers, gt_boxes, sigma=0.5)
            total_tp_center += tp_c
            total_fp_center += fp_c
            total_fn_center += fn_c

            # Per-image precision / recall / F1 (IoU-based)
            prec_img = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_img  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_img   = 2 * prec_img * rec_img / (prec_img + rec_img) if (prec_img + rec_img) > 0 else 0.0

            # Per-image precision / recall / F1 (center-based)
            prec_c_img = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec_c_img  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c_img   = 2 * prec_c_img * rec_c_img / (prec_c_img + rec_c_img) if (prec_c_img + rec_c_img) > 0 else 0.0

            per_image_writer.writerow({
                "stem":             stem,
                "pred_count":       count_pred,
                "gt_count":         count_gt,
                "abs_error":        abs(count_pred - count_gt),
                "tp_iou":           tp,
                "fp_iou":           fp,
                "fn_iou":           fn,
                "precision_iou":    round(prec_img, 4),
                "recall_iou":       round(rec_img, 4),
                "f1_iou":           round(f1_img, 4),
                "tp_center":        tp_c,
                "fp_center":        fp_c,
                "fn_center":        fn_c,
                "precision_center": round(prec_c_img, 4),
                "recall_center":    round(rec_c_img, 4),
                "f1_center":        round(f1_c_img, 4),
            })
            per_image_csv_fh.flush()
        else:
            # No GT available — write counts only, leave metric columns empty
            per_image_writer.writerow({
                "stem":       stem,
                "pred_count": count_pred,
                "gt_count":   "",
                "abs_error":  "",
                **{k: "" for k in per_image_fields[4:]},
            })
            per_image_csv_fh.flush()

        # Every counter and per-image row above is already written, so the only
        # remaining work for this image is the overlay PDF.  Grid search and
        # Optuna run thousands of configurations and never look at the images.
        if args.no_save_images:
            if count_gt >= 0:
                print(f"{stem}: pred={count_pred}, gt={count_gt}")
            else:
                print(f"{stem}: pred={count_pred}")
            continue

        # Visualization: overlay heatmap + peaks (+ optional GT boxes).
        resp_np = resp01[0,0].cpu().numpy()
        #overlay = overlay_heatmap(img, cv2.resize(resp_np, (args.resize_img, args.resize_img)))
        #overlay = draw_peaks_on_overlay(overlay, peaks_hw, down_factor)

        # 1) Upsample heatmap to image size using NEAREST (no interpolation drift)
        heat_up = cv2.resize(
            resp_np,
            (args.resize_img, args.resize_img),
            interpolation=cv2.INTER_LINEAR#INTER_NEAREST
        )

        # Overlay on original image
        overlay_orig = overlay_heatmap(img, heat_up)
        overlay_orig = draw_peaks_on_overlay(overlay_orig, peaks_hw, scale_y, scale_x)

        # Overlay on latent representation (resolución original sin escalar)
        fmap_rgb = fmap_to_rgb(fmap)
        overlay_latent = overlay_heatmap(fmap_rgb, resp_np)  # unscaled heatmap
        # Peaks are already in latent grid coordinates, so scale=1.0 for the latent overlay
        overlay_latent = draw_peaks_on_overlay(overlay_latent, peaks_hw, 1.0, 1.0, radius=2, thickness=1)

        if gt_boxes:
            overlay_orig = draw_gt_boxes(overlay_orig, gt_boxes, proto_box=proto_box)
            
            # Escalar boxes al tamaño exacto de la retícula latente
            latent_gt_boxes = [(x1/scale_x, y1/scale_y, x2/scale_x, y2/scale_y) for (x1, y1, x2, y2) in gt_boxes]
            latent_proto = tuple(b/s for b, s in zip(proto_box, [scale_x, scale_y, scale_x, scale_y])) if proto_box else None
            overlay_latent = draw_gt_boxes(overlay_latent, latent_gt_boxes, proto_box=latent_proto, thickness=1)


        # Save original overlay image as PDF
        out_path_orig = os.path.join(args.output_dir, f"{stem}_orig.pdf")
        h_px, w_px = overlay_orig.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(w_px / 100, h_px / 100), dpi=100)
        ax.imshow(overlay_orig)
        ax.axis("off")
        fig.savefig(out_path_orig, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # # Save latent overlay image
        # out_path_latent = os.path.join(args.output_dir, f"{stem}_latente.pdf")
        # h_l, w_l = overlay_latent.shape[:2]
        # fig_l, ax_l = plt.subplots(1, 1, figsize=(w_l / 100, h_l / 100), dpi=100)
        # ax_l.imshow(overlay_latent)
        # ax_l.axis("off")
        # fig_l.savefig(out_path_latent, format="pdf", bbox_inches="tight", pad_inches=0)
        # plt.close(fig_l)

        if count_gt >= 0:
            print(f"{stem}: pred={count_pred}, gt={count_gt}, saved={out_path_orig} & _latente.jpg")
        else:
            print(f"{stem}: pred={count_pred}, saved={out_path_orig} & _latente.jpg")

    per_image_csv_fh.close()
    print(f"\nPer-image results saved to: {per_image_csv_path}")

    # Metrics (only if we had GTs for >0 images and we processed more than 1 image)
    if len(gts) > 0 and len(img_paths) > 1:
        preds = np.array(preds)
        gts = np.array(gts)
        mae = np.mean(np.abs(preds - gts))
        rmse = np.sqrt(np.mean((preds - gts) ** 2))
        # MAPE: mean absolute percentage error (skip images with gt=0 to avoid div-by-zero)
        mask_nonzero = gts > 0
        if mask_nonzero.any():
            mape = np.mean(np.abs(preds[mask_nonzero] - gts[mask_nonzero]) / gts[mask_nonzero]) * 100.0
        else:
            mape = 0.0

        # Detection metrics (IoU-based)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        ap = compute_ap(
            np.array(all_det_scores, dtype=np.float64),
            np.array(all_det_tp_flags, dtype=np.float64),
            total_gt_count,
        )

        # Center-to-center distance metrics
        precision_center = total_tp_center / (total_tp_center + total_fp_center) if (total_tp_center + total_fp_center) > 0 else 0.0
        recall_center = total_tp_center / (total_tp_center + total_fn_center) if (total_tp_center + total_fn_center) > 0 else 0.0
        f1_center = 2 * precision_center * recall_center / (precision_center + recall_center) if (precision_center + recall_center) > 0 else 0.0

        print(f"\n[Metrics] MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")
        print(f"[Detection @IoU={args.iou_threshold:.2f}] "
              f"Precision={precision:.4f}  Recall={recall:.4f}  "
              f"F1={f1:.4f}  Accuracy={accuracy:.4f}  AP={ap:.4f}")
        print(f"[Center-distance @sigma=0.5] "
              f"P_center={precision_center:.4f}  R_center={recall_center:.4f}  "
              f"F1_center={f1_center:.4f}")
        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            f.write(f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nMAPE={mape:.1f}\n")
            f.write(f"Precision={precision:.4f}\nRecall={recall:.4f}\n")
            f.write(f"F1={f1:.4f}\nAccuracy={accuracy:.4f}\nAP={ap:.4f}\n")
            f.write(f"IoU_threshold={args.iou_threshold}\n")
            f.write(f"Pred_box_size={args.pred_box_size}\n")
            f.write(f"P_center={precision_center:.4f}\nR_center={recall_center:.4f}\n")
            f.write(f"F1_center={f1_center:.4f}\n")
            # --- cost: wall-clock, peak GPU and peak host memory ---
            _elapsed = time.perf_counter() - _t_start
            if torch.cuda.is_available():
                _gpu_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
                _gpu_resv  = torch.cuda.max_memory_reserved() / 1024 ** 2
            else:
                _gpu_alloc = _gpu_resv = 0.0
            # ru_maxrss is KB on Linux, bytes on macOS
            _host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            f.write(f"Infer_s={_elapsed:.3f}\n")
            f.write(f"GPU_alloc_MB={_gpu_alloc:.1f}\n")
            f.write(f"GPU_reserved_MB={_gpu_resv:.1f}\n")
            f.write(f"Host_peak_MB={_host:.1f}\n")

if __name__ == "__main__":
    main()