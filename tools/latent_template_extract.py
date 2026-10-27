# tools/latent_template_extract.py
import os
import json
import argparse
import csv
from typing import List, Dict, Tuple, Literal

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

import numpy as np
import cv2

from models.vae import VAE
from tools.utils import select_device, set_seed, create_directory


# ----------------------------
# Helpers
# ----------------------------
def load_image_rgb(path: str, resize: int) -> np.ndarray:
    """
    Read image from disk and return an RGB uint8 image resized to (resize, resize).
    Returns:
      img: np.ndarray uint8, shape (H, W, 3) = (resize, resize, 3), values in [0,255]
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
    return img

def to_tensor_chw01(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert RGB uint8 image (H, W, 3) in [0,255] to float32 tensor (C, H, W) in [0,1].
    Args:
      img_rgb_uint8: np.uint8, (H, W, 3)
    Returns:
      x: torch.float32, (3, H, W) in [0,1]
    """
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # (3, H, W)
    return torch.from_numpy(x)

def read_yolo_csv(csv_path: str) -> Dict[str, List[Tuple[float,float,float,float]]]:
    """
    Parse YOLO-style CSV into a dict of normalized boxes per image stem.
    CSV columns: filename(.txt), class, cx, cy, w, h  -- all normalized to [0,1]
    Returns:
      mapping: {stem(str): [(cx,cy,w,h), ...], ...}
    """
    mapping: Dict[str, List[Tuple[float,float,float,float]]] = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            txtname = row["filename"]  # e.g., "5b84...a.txt"
            stem = os.path.splitext(txtname)[0]  # "5b84...a"
            cx, cy, w, h = map(float, (row["cx"], row["cy"], row["w"], row["h"]))
            mapping.setdefault(stem, []).append((cx, cy, w, h))
    return mapping

def yolo_to_xyxy_pixels(cx, cy, w, h, W, H):
    """
    Convert normalized YOLO box (cx,cy,w,h) to absolute pixel (x1,y1,x2,y2).
    Args:
      cx,cy,w,h: floats in [0,1]
      W,H: image width/height in pixels
    Returns:
      (x1,y1,x2,y2): floats in pixel coordinates
    """
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return (x1, y1, x2, y2)

def standardize_channelwise(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Channel-wise z-score over spatial dims (k,k).
    Accepts:
      t: (C, k, k) or (1, C, k, k)
    Returns:
      same shape as input; each channel standardized individually.
    """
    orig = t
    if t.dim() == 3:
        t = t.unsqueeze(0)  # (1, C, k, k)
    # mean/std over spatial dims (k,k) -> shapes (1, C, 1, 1)
    mean = t.mean(dim=(2,3), keepdim=True)
    std  = t.std(dim=(2,3), keepdim=True)
    t = (t - mean) / (std + eps)
    return t.view(orig.shape)  # back to (C,k,k) if input was 3D


# ----------------------------
# Core
# ----------------------------
def _get_feature_map(
    model: VAE,
    x: torch.Tensor,
    feature_source: Literal["feats", "z", "mu"],
    noise: torch.Tensor
) -> torch.Tensor:
    """
    Forward the encoder and return the chosen spatial map to extract RoI from.
    Args:
      model: VAE
      x:     (B, 3, H, W) float32 in [0,1]
      feature_source: "feats" | "z" | "mu"
      noise: (B, 4, H/8, W/8) float32 (required by encoder API)
    Returns:
      fmap:
        - "feats": (B, 512, H/8, W/8)  last spatial map before μ/σ projection
        - "z":     (B,   4, H/8, W/8)  reparameterized latent (scaled in encoder)
        - "mu":    (B,   4, H/8, W/8)  mean of q(z|x)
    """
    model.eval()
    with torch.no_grad():
        if feature_source == "feats":
            # encoder returns (z, mu, logvar, feats) when return_features=True
            z, mu, logvar, feats = model.encoder(x, noise, return_features=True)
            return feats  # (B, 512, h, w)
        else:
            # encoder returns (z, mu, logvar) when return_features=False
            z, mu, logvar = model.encoder(x, noise, return_features=False)
            if feature_source == "z":
                return z    # (B, 4, h, w)
            elif feature_source == "mu":
                return mu   # (B, 4, h, w)
            else:
                raise ValueError(f"Unknown feature_source={feature_source}")

def extract_latent_roi_template(
    model: VAE,
    image_path: str,
    box_xyxy_img: Tuple[float,float,float,float],
    resize_img: int = 256,
    roi_size: int = 9,
    feature_source: Literal["feats","z","mu"] = "feats",
) -> Tuple[torch.Tensor, dict]:
    """
    Extract a RoI-aligned prototype from the chosen latent map.
    Args:
      model: VAE (already moved to device)
      image_path: path to .png
      box_xyxy_img: (x1,y1,x2,y2) in image pixel coords (matching resized image)
      resize_img: final image size (H=W)
      roi_size: output RoI size (k) -> prototype is (C, k, k)
      feature_source: which encoder map to use ("feats"=512ch, "z"/"mu"=4ch)
    Returns:
      proto: torch.float32, shape (C, roi_size, roi_size)
      meta:  dict with reproducibility info
    """
    device = next(model.parameters()).device

    # Load & preprocess image
    img = load_image_rgb(image_path, resize=resize_img)         # np.uint8, (H, W, 3)
    H = W = resize_img
    x = to_tensor_chw01(img).unsqueeze(0).to(device)            # torch.float32, (1, 3, H, W)

    # Prepare noise for encoder call to match expected latent grid size.
    # Using zeros for deterministic feats/mu; random for z (if desired).
    zH, zW = H // 8, W // 8
    if feature_source in ("feats", "mu"):
        noise = torch.zeros((1, 4, zH, zW), device=device)      # (1, 4, H/8, W/8)
    else:
        noise = torch.randn((1, 4, zH, zW), device=device)      # (1, 4, H/8, W/8)

    # Get the requested spatial map from the encoder
    fmap = _get_feature_map(model, x, feature_source=feature_source, noise=noise)  # (1, C, h, w)
    _, C, h, w = fmap.shape  # C=512 if "feats", else C=4

    # Spatial scale factor expected by RoIAlign: maps image pixels -> fmap pixels.
    # If H=W=256 and h=w=32, spatial_scale = 32/256 = 0.125
    spatial_scale = h / float(H)

    # Prepare ROI tensor in image pixel coordinates, including batch index
    x1, y1, x2, y2 = box_xyxy_img
    rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=device)  # (num_rois=1, 5)

    # RoIAlign samples from fmap inside the given image-space box using the spatial_scale
    # Input:  fmap  (1, C, h, w)
    # Boxes:  rois  (1, 5) = [batch_idx, x1, y1, x2, y2] in image pixels
    # Output: roi_feat (1, C, roi_size, roi_size)
    roi_feat = roi_align(
        input=fmap,
        boxes=rois,
        output_size=(roi_size, roi_size),
        spatial_scale=spatial_scale,
        sampling_ratio=-1,
        aligned=True
    )  # -> (1, C, k, k)

    # Detach to CPU and drop the batch dim
    proto = roi_feat[0].detach().cpu()  # (C, k, k)

    # Channel-wise standardization to help correlation stability downstream
    proto = standardize_channelwise(proto)  # (C, k, k)

    # Metadata useful for reproducing extraction and validating compatibility
    meta = dict(
        image_path=image_path,
        box_xyxy=[float(x1), float(y1), float(x2), float(y2)],
        resize_img=resize_img,
        roi_size=roi_size,
        feature_source=feature_source,
        latent_shape=[int(C), int(h), int(w)],   # C,h,w of fmap used
        spatial_scale=float(spatial_scale),
        note="Channel-wise standardized prototype",
    )
    return proto, meta


def main():
    """
    CLI tool to extract a latent (RoIAlign) template from annotated images.
    Saves:
      - latent_prototype.pt  : torch float32 tensor, (C, k, k)
      - latent_prototype_meta.json : metadata including feature_source, latent C/h/w, etc.
      - templates/latent_prototype_vis.png : visual representation of the template
    """
    ap = argparse.ArgumentParser(description="Extract a latent (RoIAlign) template from annotated images.")
    ap.add_argument("--images_dir", required=True, help="Folder with .png images")
    ap.add_argument("--csv", required=True, help="YOLO CSV with normalized cx,cy,w,h")
    ap.add_argument("--checkpoint", required=False, default=None, help="Path to VAE checkpoint (best.pt). If omitted, uses random weights (not recommended).")
    ap.add_argument("--output_dir", required=True, help="Where to save prototype .pt + metadata.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--resize_img", type=int, default=256)
    ap.add_argument("--roi_size", type=int, default=9)
    ap.add_argument("--max_crops", type=int, default=1, help="How many crops to average into the prototype (1 = single crop).")
    ap.add_argument("--image_stem", type=str, default=None, help="Optional: force using a specific image stem (without extension).")
    # NEW: choose which latent to use for the prototype
    ap.add_argument("--feature_source", type=str, default="feats", choices=["feats", "z", "mu"],
                    help="Map to extract RoI from: 'feats' (512ch pre-projection), 'z' (4ch reparam, scaled), or 'mu' (4ch mean).")
    args = ap.parse_args()

    device = select_device(args.device)
    set_seed(args.seed, deterministic=True, cudnn_benchmark=False)
    create_directory(args.output_dir)

    # Build / load VAE
    model = VAE().to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint provided. Proceeding with randomly initialized VAE (prototype quality will be poor).")

    # ---------------------------------------------------------------
    # Build the list of (stem, cx, cy, w, h) from the CSV passed by
    # run_grid_search.py.  prepare_experiment.py writes a per-replica
    # filtered CSV with exactly the pre-selected, quality-filtered crops
    # in order — we just consume them here.
    # The first `max_crops` entries are used (matching the grid search param).
    # ---------------------------------------------------------------
    ann = read_yolo_csv(args.csv)  # dict: stem -> list of (cx,cy,w,h)

    CURATED_BOXES = []
    for stem, boxes in ann.items():
        for (cx, cy, w, h) in boxes:
            CURATED_BOXES.append((stem, cx, cy, w, h))

    if not CURATED_BOXES:
        raise RuntimeError(
            f"No boxes found in CSV: {args.csv}. "
            "Make sure prepare_experiment.py wrote the template CSV correctly."
        )

    print(f"Loaded {len(CURATED_BOXES)} candidate crops from CSV "
          f"(will use first {args.max_crops}).")


    # Clean up stale visualization crops from previous runs to avoid confusion.
    crop_dir = os.path.join(args.output_dir, "images")
    if os.path.isdir(crop_dir):
        for fname in os.listdir(crop_dir):
            if fname.startswith("template_") and fname.endswith(".jpg"):
                try:
                    os.remove(os.path.join(crop_dir, fname))
                except OSError:
                    pass

    used = 0
    protos: List[torch.Tensor] = []
    proto_meta = None

    for (stem, cx, cy, w, h) in CURATED_BOXES:
        img_path = os.path.join(args.images_dir, stem + ".png")
        if not os.path.isfile(img_path):
            print(f"WARNING: image not found for stem={stem}, skipping")
            continue
        x1, y1, x2, y2 = yolo_to_xyxy_pixels(cx, cy, w, h, args.resize_img, args.resize_img)
        proto, meta = extract_latent_roi_template(
            model, img_path, (x1, y1, x2, y2),
            resize_img=args.resize_img,
            roi_size=args.roi_size,
            feature_source=args.feature_source,
        )
        protos.append(proto)
        proto_meta = meta
        used += 1
        print(f"Added crop from {stem}  box={x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")

        # Save visual crop of the prototype for inspection
        img_np = load_image_rgb(img_path, resize=args.resize_img)
        crop_img = img_np[int(y1):int(y2), int(x1):int(x2)]
        if crop_img.size > 0:
            os.makedirs(crop_dir, exist_ok=True)
            crop_path = os.path.join(crop_dir, f"template_{used}_{stem}.jpg")
            cv2.imwrite(crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

        if used >= args.max_crops:
            break

    if len(protos) == 0:
        raise RuntimeError("No prototypes extracted. Check CSV, paths, and stems.")

    # If multiple crops requested, average them channel-wise to build a single prototype.
    # proto_stack: (N, C, k, k) -> proto_mean: (C, k, k)
    with torch.no_grad():
        proto_stack = torch.stack(protos, dim=0)  # (N, C, k, k)
        proto_mean = proto_stack.mean(dim=0)      # (C, k, k)

    # Save prototype tensor (averaged)
    proto_path = os.path.join(args.output_dir, "latent_prototype.pt")
    meta_path  = os.path.join(args.output_dir, "latent_prototype_meta.json")
    torch.save(proto_mean, proto_path)

    # Save individual prototypes as a list (needed by majority_voting mode)
    bank_path = os.path.join(args.output_dir, "template_bank.pt")
    torch.save(protos, bank_path)
    print(f"Saved template bank ({len(protos)} templates) to: {bank_path}")

    # Save concise metadata (includes latent C/h/w for downstream compatibility checks)
    save_meta = dict(
        num_crops=len(protos),
        image_path=proto_meta["image_path"],
        box_xyxy=proto_meta["box_xyxy"],
        resize_img=proto_meta["resize_img"],
        roi_size=proto_meta["roi_size"],
        feature_source=proto_meta["feature_source"],
        latent_shape=proto_meta["latent_shape"],   # [C, h, w]
        spatial_scale=proto_meta["spatial_scale"],
        note=proto_meta["note"]
    )
    with open(meta_path, "w") as f:
        json.dump(save_meta, f, indent=2)

    print(f"Saved prototype tensor to: {proto_path}")
    print(f"Saved metadata to:        {meta_path}")

    # ----------------------------
    # Visualization Export
    # ----------------------------
    # Create templates directory
    vis_dir = "templates"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Average across the channel dimension and normalize for visualization
    proto_vis = proto_mean.mean(dim=0).cpu().numpy()
    proto_vis = (proto_vis - proto_vis.min()) / (proto_vis.max() - proto_vis.min() + 1e-8) * 255.0
    proto_vis = proto_vis.astype(np.uint8)
    
    # Resize image to make the small template (e.g., 9x9) visible
    display_size = max(100, args.roi_size * 10)
    proto_vis_resized = cv2.resize(proto_vis, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    
    vis_filename = f"{args.image_stem}_vis.png" if args.image_stem else "latent_prototype_vis.png"
    vis_path = os.path.join(vis_dir, vis_filename)
    cv2.imwrite(vis_path, proto_vis_resized)
    
    print(f"Saved template visualization to: {vis_path}")

if __name__ == "__main__":
    main()