#!/usr/bin/env python3
"""
depth.py

Takes:
 - col_dir: color frames (frame_*.png)
 - dep_dir: depth frames (frame_*.tiff) -- aligned to color
 - mask_dir: folder with subfolders: red, green, gray, yellow_1, yellow_2
   each containing masks named same as frames (frame_*.png) or any matching basename

Produces in output_dir:
 - combined_side_by_side.mp4  (color | depth_colorized) preview video with mask overlays
 - depth_only_masked.mp4      (depth_colorized) video with overlays
 - depths.csv                 CSV with per-frame per-object z in mm (min, max, com) and mean_com_mm

Preview appears while processing. Press 'q' to quit early (video/files remain).
"""

import os
import cv2
import numpy as np
import argparse
import csv
import re
from typing import List, Tuple

# ---------- Parameters ----------
DEPTH_UNITS = 0.0010000000474974513  # meters per depth unit (from your data)
FPS = 30
VIDEO_CODEC = "mp4v"
# --------------------------------

INT_RE = re.compile(r"(\d+)")
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(INT_RE, s)]

# ---------- Global Crop Box ----------
global_crop_box = None  # (x_min, y_min, x_max, y_max)

def parse_crop_box_from_mask(mask_dir: str):
    """Read crop coordinates from mask.txt inside mask_dir."""
    global global_crop_box
    mask_txt = os.path.join(mask_dir, "mask.txt")
    if not os.path.exists(mask_txt):
        raise FileNotFoundError(f"❌ mask.txt not found in {mask_dir}")

    crop = {}
    with open(mask_txt, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=")
                crop[k.strip()] = int(v.strip())

    x_min = crop.get("x_min")
    y_min = crop.get("y_min")
    x_max = crop.get("x_max")
    y_max = crop.get("y_max")

    if None in (x_min, y_min, x_max, y_max):
        raise ValueError(f"❌ Invalid crop box in mask.txt: {crop}")

    global_crop_box = (x_min, y_min, x_max, y_max)
    print(f"✅ Loaded crop box from mask.txt: {global_crop_box}")
    return global_crop_box


def crop_color_frame(img: np.ndarray) -> np.ndarray:
    """Crop color frame using crop box from mask.txt."""
    global global_crop_box
    if global_crop_box is None:
        raise RuntimeError("Crop box not loaded. Call parse_crop_box_from_mask() first.")
    x_min, y_min, x_max, y_max = global_crop_box
    return img[y_min:y_max, x_min:x_max]


def crop_depth_frame(img: np.ndarray) -> np.ndarray:
    """Crop depth frame using the same crop box."""
    global global_crop_box
    if global_crop_box is None:
        raise RuntimeError("Crop box not loaded. Call parse_crop_box_from_mask() first.")
    x_min, y_min, x_max, y_max = global_crop_box
    return img[y_min:y_max, x_min:x_max]


def find_matching_file(basename: str, directory: str, exts: List[str]):
    for ext in exts:
        candidate = os.path.join(directory, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def safe_read_mask(path: str, target_shape):
    if not path or not os.path.exists(path):
        return np.zeros(target_shape, dtype=np.uint8)
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return np.zeros(target_shape, dtype=np.uint8)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape != target_shape:
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binmask = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    return binmask


def depth_to_mm(depth_raw: np.ndarray):
    return (depth_raw.astype(np.float32) * DEPTH_UNITS * 1000.0)


def compute_object_depth_stats(depth_mm: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    valid = (mask > 0) & (depth_mm > 0)
    if not np.any(valid):
        return None, None, None
    vals = depth_mm[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None, None
    z_min = float(np.min(vals))
    z_max = float(np.max(vals))
    z_median = float(np.median(vals))
    return z_min, z_max, z_median


def colorize_depth_mm(depth_mm: np.ndarray, clip_min: float = None, clip_max: float = None):
    d = depth_mm.copy()
    valid = d > 0
    if clip_min is None or clip_max is None:
        if np.any(valid):
            mn = float(np.min(d[valid]))
            mx = float(np.max(d[valid]))
        else:
            mn, mx = 0.0, 1.0
    else:
        mn, mx = clip_min, clip_max
    if mx <= mn:
        mx = mn + 1.0
    norm = np.zeros_like(d, dtype=np.uint8)
    norm[valid] = np.clip(((d[valid] - mn) / (mx - mn) * 255.0), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def draw_mask_overlay(img: np.ndarray, mask: np.ndarray, color=(0,255,255), alpha=0.4):
    overlay = img.copy()
    color_bgr = tuple(int(c) for c in color)
    mask_bool = (mask > 0)
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color_bgr) * alpha).astype(np.uint8)
    return overlay


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Compute per-object depth (Z in mm) using masks and depth frames; make preview videos and CSV.")
    parser.add_argument("--col_dir", required=True)
    parser.add_argument("--dep_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--show_preview", action="store_true")
    parser.add_argument("--video_codec", default=VIDEO_CODEC)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load crop box from mask.txt
    parse_crop_box_from_mask(args.mask_dir)

    mask_subs = {
        "red": "red",
        "green": "green",
        "gray": "gray",
        "yellow_1": "yellow_1",
        "yellow_2": "yellow_2",
        "gold": "gold"
    }

    col_files = [f for f in os.listdir(args.col_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    col_files.sort(key=natural_key)
    if len(col_files) == 0:
        print("No color frames found in", args.col_dir)
        return

    csv_path = os.path.join(args.output_dir, "depths.csv")
    csv_fieldnames = ["frame"]
    for key in mask_subs.keys():
        csv_fieldnames += [f"{key}_min_mm", f"{key}_max_mm", f"{key}_com_mm"]
    csv_fieldnames += ["mean_com_mm"]

    sample_color = cv2.imread(os.path.join(args.col_dir, col_files[0]))
    sample_color = crop_color_frame(sample_color)
    h, w = sample_color.shape[:2]
    combined_w = w * 2

    fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
    combined_path = os.path.join(args.output_dir, "combined_side_by_side.mp4")
    depth_only_path = os.path.join(args.output_dir, "depth_only_masked.mp4")

    vw_combined = cv2.VideoWriter(combined_path, fourcc, args.fps, (combined_w, h))
    vw_depth = cv2.VideoWriter(depth_only_path, fourcc, args.fps, (w, h))

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        csv_writer.writeheader()

        print(f"Processing {len(col_files)} frames...")
        for idx, fname in enumerate(col_files, 1):
            basename, _ = os.path.splitext(fname)
            color_path = os.path.join(args.col_dir, fname)
            depth_path = find_matching_file(basename, args.dep_dir, [".tiff", ".tif", ".png"])
            if depth_path is None:
                print(f"[WARN] no depth for {basename}; skipping")
                continue

            color = cv2.imread(color_path)
            if color is None:
                print(f"[WARN] Could not read color frame {color_path}, skipping...")
                continue
            color = crop_color_frame(color)

            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                print(f"[WARN] Could not read depth frame {depth_path}, skipping...")
                continue
            if depth_raw.ndim == 3:
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
            if depth_raw.dtype != np.uint16:
                depth_raw = depth_raw.astype(np.uint16)
            depth_raw = crop_depth_frame(depth_raw)

            depth_mm = depth_to_mm(depth_raw)

            row = {"frame": basename}
            com_vals = []
            overlay_color = color.copy()
            overlay_depth = colorize_depth_mm(depth_mm)
            overlay_colors = {
                "red": (0,0,255),
                "green": (0,255,0),
                "gray": (255,0,0),
                "yellow_1": (0,255,255),
                "yellow_2": (0,128,255),
                "gold": (0,215,255)
            }

            for key, sub in mask_subs.items():
                mask_path = find_matching_file(basename, os.path.join(args.mask_dir, sub),
                                               [".png",".jpg",".jpeg",".tif",".tiff"])
                mask = safe_read_mask(mask_path, (h,w))
                zmin,zmax,zcom = compute_object_depth_stats(depth_mm, mask)
                row[f"{key}_min_mm"] = "" if zmin is None else f"{zmin:.3f}"
                row[f"{key}_max_mm"] = "" if zmax is None else f"{zmax:.3f}"
                row[f"{key}_com_mm"] = "" if zcom is None else f"{zcom:.3f}"
                if zcom is not None:
                    com_vals.append(zcom)

                if np.any(mask):
                    overlay_color = draw_mask_overlay(overlay_color, mask, color=overlay_colors.get(key,(0,255,255)), alpha=0.35)
                    overlay_depth = draw_mask_overlay(overlay_depth, mask, color=overlay_colors.get(key,(0,255,255)), alpha=0.35)

            row["mean_com_mm"] = "" if not com_vals else f"{np.mean(com_vals):.3f}"
            csv_writer.writerow(row)

            combined = np.hstack((overlay_color, overlay_depth))
            vw_combined.write(combined)
            vw_depth.write(overlay_depth)

            if args.show_preview:
                cv2.imshow("Preview (q to quit)", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(col_files)} frames...")

    vw_combined.release()
    vw_depth.release()
    if args.show_preview:
        cv2.destroyAllWindows()

    print("Done. Outputs saved to:", args.output_dir)
    print(" -", combined_path)
    print(" -", depth_only_path)
    print(" -", csv_path)


if __name__ == "__main__":
    main()
