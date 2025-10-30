#!/usr/bin/env python3
"""
realtime_objects_xyz_with_video.py

Same as your original script but adds a final video writer that:
 - writes a full-frame video (original frame size)
 - overlays semi-transparent ROI masks (saved previously into out_dir/mask/<obj>/frame_XXXX.png)
 - draws per-frame frame number and per-object state circles:
     * solid circle = untouched
     * empty circle = moving/other
     * semi-transparent filled circle = checking
 - mask.txt in mask directory contains crop box in the short single-value format:
       x_min=209
       y_min=69
       x_max=511
       y_max=419
"""

import os
import re
import cv2
import csv
import math
import argparse
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

# ----------------- Defaults / Tunables -----------------
DEFAULT_DEPTH_UNITS = 0.0010000000474974513  # meters per unit (override via --depth_units if needed)
YELLOW_TRACK_START = 101
GRAY_JUMP_START = 101
GRAY_JUMP_THRESH = 79.4  # px
MIN_CONTOUR_AREA = 65
AREA_KEEP_FRAC = 0.35
# -------------------------------------------------------

INT_RE = re.compile(r"(\d+)")
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(INT_RE, s)]

def parse_frame_num(fname: str) -> Optional[int]:
    m = re.search(r"frame_(\d+)\.(png|jpg|jpeg|tif|tiff)$", fname, re.IGNORECASE)
    return int(m.group(1)) if m else None

def log_msg(msg: str, quiet: bool = False):
    if quiet:
        print(msg)
    else:
        tqdm.write(msg)

# ----------------- ArUco crop helpers ------------------
def build_aruco_detector():
    import cv2.aruco as aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    return aruco.ArucoDetector(aruco_dict, parameters)

def detect_crop_box(img_bgr: np.ndarray, detector) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 3 or len(ids) > 4:
        return None

    selected_points = []
    for i, corner in enumerate(corners):
        marker_id = ids[i][0]
        pts = corner.reshape((4, 2))
        # match your reference rule:
        if marker_id in [1, 2, 3]:
            sel = pts[0]
        elif marker_id == 0:
            sel = pts[1]
        else:
            sel = None
        if sel is not None:
            selected_points.append(sel)

    if len(selected_points) < 2:
        return None

    pts = np.array(selected_points, dtype=int)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def scan_crop_box(col_dir: str, all_color_files: List[str], no_progress: bool) -> Optional[Tuple[int,int,int,int]]:
    detector = build_aruco_detector()
    # restrict to files that match frame_*.png-like
    annot = [(f, parse_frame_num(f)) for f in all_color_files]
    annot = [t for t in annot if t[1] is not None]
    # Phase 1: 100→50
    phase1 = [t for t in annot if 50 <= t[1] <= 100]
    phase1.sort(key=lambda x: x[1], reverse=True)
    it1 = phase1 if no_progress else tqdm(phase1, desc="Phase 1: ArUco scan (100→50)", unit="frame")
    for fname, fnum in it1:
        img = cv2.imread(os.path.join(col_dir, fname))
        if img is None:
            continue
        box = detect_crop_box(img, detector)
        if box is not None:
            log_msg(f"✅ Crop from {fname}: {box}", quiet=no_progress)
            return box
    # Phase 2: 101→end
    phase2 = [t for t in annot if t[1] and t[1] >= 101]
    phase2.sort(key=lambda x: x[1])
    it2 = phase2 if no_progress else tqdm(phase2, desc="Phase 2: ArUco scan (101→end)", unit="frame")
    for fname, fnum in it2:
        img = cv2.imread(os.path.join(col_dir, fname))
        if img is None:
            continue
        box = detect_crop_box(img, detector)
        if box is not None:
            log_msg(f"✅ Crop from {fname}: {box}", quiet=no_progress)
            return box
    log_msg("⚠️ No ArUco crop detected; proceeding un-cropped.", quiet=no_progress)
    return None

# ----------------- Masking utilities -------------------
def union_significant_contours(binary_mask: np.ndarray, min_area: int, area_frac_keep: float):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    contours.sort(key=cv2.contourArea, reverse=True)
    largest = cv2.contourArea(contours[0])
    keep = [c for c in contours if cv2.contourArea(c) >= area_frac_keep * largest]
    if not keep:
        keep = [contours[0]]
    out = np.zeros_like(binary_mask)
    cv2.drawContours(out, keep, -1, 255, thickness=cv2.FILLED)
    return out

def compute_masks(image_bgr: np.ndarray, hsv: np.ndarray, colors: List[str],
                  frame_idx: int, prev_two_yellow: Optional[Tuple[Tuple[int,int],Tuple[int,int]]]) -> Tuple[Dict[str,np.ndarray], Optional[Tuple[Tuple[int,int],Tuple[int,int]]]]:
    kernel = np.ones((3,3), np.uint8)
    masks = {}

    if "red" in colors:
        m1 = cv2.inRange(hsv, np.array([0, 120, 110]), np.array([10, 240, 235]))
        m2 = cv2.inRange(hsv, np.array([165, 120, 110]), np.array([180, 240, 235]))
        m = cv2.bitwise_or(m1, m2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        mf = union_significant_contours(m, MIN_CONTOUR_AREA+15, AREA_KEEP_FRAC)
        if mf is not None:
            masks["red"] = mf

    if "green" in colors:
        m = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([70, 150, 190]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        mf = union_significant_contours(m, MIN_CONTOUR_AREA+15, AREA_KEEP_FRAC)
        if mf is not None:
            masks["green"] = mf

    if "gray" in colors:
        m = cv2.inRange(hsv, np.array([10, 0, 90]), np.array([100, 50, 160]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filt = []
        for c in contours:
            if not any((pt[0][0] < 60 and pt[0][1] > 300) for pt in c):
                if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                    filt.append(c)
        if filt:
            c = max(filt, key=cv2.contourArea)
            mf = np.zeros_like(m)
            cv2.drawContours(mf, [c], -1, 255, -1)
            masks["gray"] = mf

    # --- YELLOW (updated per your pseudocode) ---
    if "yellow" in colors:
        my = cv2.inRange(hsv, np.array([18, 185, 160]), np.array([30, 255, 255]))
        my = cv2.morphologyEx(my, cv2.MORPH_OPEN, kernel)
        my = cv2.morphologyEx(my, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(my, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

        # Compute centroids
        cents: List[Tuple[int,int]] = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cents.append((cx, cy))

        def contour_to_mask(c):
            mf = np.zeros_like(my)
            cv2.drawContours(mf, [c], -1, 255, -1)
            return mf

        if frame_idx < YELLOW_TRACK_START:
            # label-by-order (at most 2)
            for i, c in enumerate(contours[:2]):
                masks[f"yellow_{i+1}"] = contour_to_mask(c)
            # seed only if we have exactly two centroids
            if len(cents) == 2:
                prev_two_yellow = (cents[0], cents[1])

        else:
            # frame >= 101
            if len(cents) == 2:
                if prev_two_yellow is not None:
                    # Take ONE current mask (choose any deterministic one; use index 0)
                    # Check if its centroid is closer to prev yellow1 or yellow2
                    (py1x, py1y), (py2x, py2y) = prev_two_yellow
                    (c0x, c0y), (c1x, c1y) = cents[0], cents[1]

                    d0_y1 = math.hypot(c0x - py1x, c0y - py1y)
                    d0_y2 = math.hypot(c0x - py2x, c0y - py2y)

                    if d0_y1 <= d0_y2:
                        # current contour 0 → yellow_1, the other → yellow_2
                        y1_idx, y2_idx = 0, 1
                    else:
                        # current contour 0 is closer to prev yellow_2
                        y1_idx, y2_idx = 1, 0

                    masks["yellow_1"] = contour_to_mask(contours[y1_idx])
                    masks["yellow_2"] = contour_to_mask(contours[y2_idx])

                    # update prev by the assigned centroids (yellow_1 first, then yellow_2)
                    prev_two_yellow = (cents[y1_idx], cents[y2_idx])
                else:
                    # fallback seed then save by order
                    prev_two_yellow = (cents[0], cents[1])
                    for i, c in enumerate(contours[:2]):
                        masks[f"yellow_{i+1}"] = contour_to_mask(c)

            elif len(cents) == 1 and prev_two_yellow is not None:
                # pick label by which previous centroid is closer
                (py1x, py1y), (py2x, py2y) = prev_two_yellow
                (cx, cy) = cents[0]
                d_y1 = math.hypot(cx - py1x, cy - py1y)
                d_y2 = math.hypot(cx - py2x, cy - py2y)

                if d_y1 <= d_y2:
                    label = "yellow_1"
                    prev_two_yellow = ((cx, cy), (py2x, py2y))
                else:
                    label = "yellow_2"
                    prev_two_yellow = ((py1x, py1y), (cx, cy))

                # draw only that label’s mask (use the first/only contour)
                masks[label] = contour_to_mask(contours[0])

            # else: 0 yellow found → no masks and keep prev_two_yellow as-is

    if "gold" in colors:
        m = cv2.inRange(hsv, np.array([18, 100, 120]), np.array([23, 190, 190]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        if contours:
            c = max(contours, key=cv2.contourArea)
            mf = np.zeros_like(m)
            cv2.drawContours(mf, [c], -1, 255, -1)
            masks["gold"] = mf

    return masks, prev_two_yellow

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return x,y,w,h

def com_from_bbox(xywh: Tuple[int,int,int,int]) -> Tuple[int,int]:
    x,y,w,h = xywh
    return x + w//2, y + h//2

# --------------- Depth helpers -------------------------
def depth_to_mm(depth_raw: np.ndarray, depth_units: float):
    return (depth_raw.astype(np.float32) * depth_units * 1000.0)

def safe_read_depth(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path): return None
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None: return None
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    if d.dtype != np.uint16:
        d = d.astype(np.uint16)
    return d

def compute_z_stats_mm(depth_mm: np.ndarray, mask: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    valid = (mask > 0) & (depth_mm > 0)
    if not np.any(valid):
        return None, None, None
    vals = depth_mm[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None, None
    return float(np.min(vals)), float(np.max(vals)), float(np.median(vals))

# --------------- Untouched state machine ---------------
def depth_ok_for(obj_name: str, z_mm: float) -> bool:
    # match your rule-of-thumb from the ref:
    if obj_name.startswith("yellow"):
        return 720 <= z_mm <= 760
    else:
        return 680 <= z_mm <= 760

class UntouchedState:
    """Holds per-object interval state for online updates."""
    def __init__(self):
        self.interval_active = False
        self.bbox = None           # (xmin,ymin,xmax,ymax) fixed during active interval
        self.x_start = None
        self.last_good = None
        # Fixed-bucket checking state
        self.check_start = None    # int or None
        self.check_end = None      # int or None  (inclusive)
        self.saw_any_bad = False   # True if at least one evaluable bad seen inside bucket
        self.ref_wh = None         # (w_ref, h_ref)

def update_untouched_states(
    states: Dict[str, UntouchedState],
    obj_order: List[str],
    fnum: int,
    fps: int,
    start_time: float,
    p: int,
    q: int,
    xy_dict: Dict[str, Optional[Tuple[int,int]]],
    z_dict: Dict[str, Optional[float]],
    untouched_out: Dict[str, List[List[int]]],
    checking_out: Dict[str, List[List[int]]]
):
    start_frame = int(start_time * fps)

    for obj in obj_order:
        st = states[obj]
        xy = xy_dict.get(obj)   # (cx, cy) or None
        z  = z_dict.get(obj)    # float or None

        # ---------------- ACTIVE INTERVAL ----------------
        if st.interval_active:
            # If a checking bucket is open, manage the fixed window first
            if st.check_start is not None and st.check_end is not None:
                # We are inside/around a fixed checking bucket
                if fnum <= st.check_end:
                    # Still inside the bucket
                    if xy is not None and z is not None:
                        cx, cy = xy
                        xmin, ymin, xmax, ymax = st.bbox
                        inside = (xmin <= cx <= xmax) and (ymin <= cy <= ymax)
                        z_ok   = depth_ok_for(obj, z)

                        if inside and z_ok:
                            # Immediate recovery: close checking, continue untouched
                            checking_out[obj].append([st.check_start, fnum - 1])
                            st.check_start = None
                            st.check_end = None
                            st.saw_any_bad = False
                            st.last_good = fnum
                        else:
                            # evaluable & bad → mark bad inside bucket
                            st.saw_any_bad = True
                            # keep waiting until bucket ends
                    else:
                        # unevaluable inside bucket → do nothing
                        pass

                else:
                    # fnum just moved PAST the bucket end → finalize decision
                    if st.saw_any_bad:
                        # End untouched interval (no recovery happened)
                        end_frame = st.last_good if st.last_good is not None else (st.x_start - 1)
                        if end_frame is not None and st.x_start is not None and end_frame >= st.x_start:
                            untouched_out[obj].append([st.x_start-p, end_frame])
                        # record checking span up to fixed end
                        checking_out[obj].append([st.check_start, st.check_end])
                        # reset all
                        st.interval_active = False
                        st.bbox = None
                        st.x_start = None
                        st.last_good = None
                        st.check_start = None
                        st.check_end = None
                        st.saw_any_bad = False
                    else:
                        # No evaluable bad at all in bucket → keep untouched ongoing
                        checking_out[obj].append([st.check_start, st.check_end])
                        st.check_start = None
                        st.check_end = None
                        st.saw_any_bad = False
                        # We can extend last_good through this frame (benign)
                        st.last_good = fnum

                # After handling bucket, move to next object
                continue

            # No open bucket → evaluate current frame
            if xy is None or z is None:
                # Unevaluable while untouched → treat as still fine; extend last_good
                st.last_good = fnum
                continue

            # Evaluable frame while untouched
            cx, cy = xy
            xmin, ymin, xmax, ymax = st.bbox
            inside = (xmin <= cx <= xmax) and (ymin <= cy <= ymax)
            z_ok   = depth_ok_for(obj, z)

            if inside and z_ok:
                # Keep untouched going
                st.last_good = fnum
            else:
                # First bad evaluable → open fixed checking bucket
                st.check_start = fnum
                st.check_end = fnum + q - 1  # inclusive
                st.saw_any_bad = True  # this first frame is a bad-evaluable

            continue
        # -------------- END ACTIVE INTERVAL --------------

        # Not active → (unchanged) wait to start after warmup handled in main loop
        if fnum < start_frame:
            continue
        if xy is None:
            continue
        if st.ref_wh is None:
            continue
        # (start logic is handled in the main loop's warmup)
        continue

# --- Helper for semi-transparent drawing ---
def _draw_transparent_circle(img_bgr, center, radius, color_bgr, alpha=0.45, outline=True):
    """Blend a filled circle with transparency onto img_bgr."""
    overlay = img_bgr.copy()
    cv2.circle(overlay, center, radius, color_bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, dst=img_bgr)
    if outline:
        cv2.circle(img_bgr, center, radius, color_bgr, 1)

# ----------------- Video Renderer -----------------------
def render_overlay_video(
    col_dir: str,
    mask_root: str,
    crop_box: Optional[Tuple[int,int,int,int]],
    untouched_out: Dict[str, List[List[int]]],
    checking_out: Dict[str, List[List[int]]],
    output_dir: str,
    video_name: str,
    fps: int,
    no_progress: bool = False
):
    """
    Renders a full-frame mp4 with:
     - masks overlaid at crop_box location (masks are ROI-sized images saved under mask_root/<obj>/frame_XXXX.png)
     - object-state circles (solid=untouched, hollow=moving, semi-transparent=checking)
     - frame number text
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, video_name)

    # color map (BGR)
    COLOR_MAP = {
        "red":      (0, 0, 255),
        "green":    (0, 255, 0),
        "gray":     (255, 255, 255),
        "yellow_1": (0, 255, 255),
        "yellow_2": (0, 128, 255),
        "gold":     (128, 0, 255),
    }
    objects = list(COLOR_MAP.keys())

    frame_files = sorted([f for f in os.listdir(col_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))], key=natural_key)
    if not frame_files:
        raise RuntimeError("No frames found to render video")

    # use first frame to get full-frame size
    sample = cv2.imread(os.path.join(col_dir, frame_files[0]))
    if sample is None:
        raise RuntimeError("Cannot read first frame for video sizing")
    full_h, full_w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (full_w, full_h))

    # circle legend placement (full-frame coords)
    circle_radius = 8
    start_x = 20
    spacing = 50
    baseline_y = 40  # y for legend circles
    font = cv2.FONT_HERSHEY_SIMPLEX

    it = frame_files if no_progress else tqdm(frame_files, desc="Rendering overlay video", unit="frame")
    for fname in it:
        frame_idx = parse_frame_num(fname)
        frame_path = os.path.join(col_dir, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        overlay = frame.copy()

        # --- Overlay masks (place them at crop box position) ---
        if crop_box:
            x_min, y_min, x_max, y_max = crop_box
            crop_w, crop_h = x_max - x_min, y_max - y_min

            # For each object attempt to load mask image from mask_root/<obj>/<fname>
            for obj in objects:
                mpath = os.path.join(mask_root, obj, fname)
                if not os.path.exists(mpath):
                    continue
                mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                # If mask shape mismatches crop size, attempt to fallback crop or skip
                if mask.shape[:2] != (crop_h, crop_w):
                    # If mask is full-frame we can crop
                    if mask.shape[0] >= y_max and mask.shape[1] >= x_max:
                        # crop to ROI
                        mask = mask[y_min:y_max, x_min:x_max]
                    else:
                        # warn and skip
                        tqdm.write(f"[warn] mask size mismatch {mpath}: {mask.shape} vs {(crop_h, crop_w)} -- skipping")
                        continue

                color = COLOR_MAP.get(obj, (255,255,255))
                # create colored fill image for the ROI
                colored_roi = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                colored_roi[mask > 0] = color

                # Put ROI onto overlay using alpha blending
                roi = overlay[y_min:y_max, x_min:x_max]
                blended = cv2.addWeighted(roi, 1.0, colored_roi, 0.4, 0)
                # draw contour (outline)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    cv2.drawContours(blended, [c], -1, color, thickness=2)
                overlay[y_min:y_max, x_min:x_max] = blended

        # --- Draw frame number ---
        frame_label = f"Frame: {frame_idx}" if frame_idx is not None else fname
        cv2.putText(overlay, frame_label, (10, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Draw per-object state circles --- (use untouched_out/checking_out spans)
        for i, obj in enumerate(objects):
            cx = start_x + i * spacing
            cy = baseline_y
            color = COLOR_MAP[obj]

            # Determine state for this frame
            is_untouched = any(a <= frame_idx <= b for (a,b) in untouched_out.get(obj, [])) if frame_idx is not None else False
            is_checking  = any(a <= frame_idx <= b for (a,b) in checking_out.get(obj, [])) if frame_idx is not None else False

            if is_checking:
                _draw_transparent_circle(overlay, (cx, cy), circle_radius, color, alpha=0.45, outline=True)
            elif is_untouched:
                cv2.circle(overlay, (cx, cy), circle_radius, color, -1)  # solid = untouched
            else:
                cv2.circle(overlay, (cx, cy), circle_radius, color, 2)   # empty = moving

            # Optional: label below circle
            cv2.putText(overlay, obj, (cx - 24, cy + 24),
                        font, 0.4, color, 1, cv2.LINE_AA)

        out.write(overlay)

    out.release()
    print(f"✅ Video saved → {output_path}")

# ----------------- Main Script -------------------------
def main():
    ap = argparse.ArgumentParser(description="Real-time style objects+COM(x,y,z)+untouched in one pass, with final overlay video")
    ap.add_argument("--col_dir", required=True, help="Directory of color frames (frame_*.png)")
    ap.add_argument("--dep_dir", required=True, help="Directory of depth frames (.tiff/.tif/.png)")
    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--colors", nargs="+",
                    choices=["red","green","gray","yellow","gold"],
                    default=["red","green","gray","yellow","gold"])
    ap.add_argument("--start_time", type=float, required=True, help="Seconds from which to allow untouched starts")
    ap.add_argument("--p", type=int, required=True, help="Window length to START untouched (lookback/warmup)")
    ap.add_argument("--q", type=int, required=True, help="Window length to END untouched (consecutive bad)")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--ref_frame", type=int, default=100, help="Reference frame index for per-object W×H")
    ap.add_argument("--depth_units", type=float, default=DEFAULT_DEPTH_UNITS, help="Meters per depth unit")
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_root = os.path.join(args.out_dir, "mask")
    os.makedirs(mask_root, exist_ok=True)

    # Collect color files (sorted)
    color_files = [f for f in os.listdir(args.col_dir) if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]
    color_files.sort(key=natural_key)
    if not color_files:
        raise RuntimeError("No frames in --col_dir")

    # 1) ArUco crop detection
    crop_box = scan_crop_box(args.col_dir, color_files, args.no_progress)
    # Save mask.txt in the short format requested
    mask_txt_path = os.path.join(mask_root, "mask.txt")
    with open(mask_txt_path, "w") as f:
        if crop_box is None:
            f.write("No crop region detected.\n")
        else:
            x_min,y_min,x_max,y_max = crop_box
            f.write(f"x_min={x_min}\n")
            f.write(f"y_min={y_min}\n")
            f.write(f"x_max={x_max}\n")
            f.write(f"y_max={y_max}\n")

    def crop_img(img: np.ndarray) -> np.ndarray:
        if crop_box is None:
            return img
        x_min,y_min,x_max,y_max = crop_box
        return img[y_min:y_max, x_min:x_max]

    # Prepare CSV writers
    xy_csv_path = os.path.join(args.out_dir, "xy_com.csv")
    depths_csv_path = os.path.join(args.out_dir, "depths.csv")

    xy_colors_columns = ["red","green","gray","yellow_1","yellow_2","gold"]
    # expand yellow in mask saving even if user omitted it
    active_objs = []
    for c in args.colors:
        if c == "yellow":
            active_objs += ["yellow_1","yellow_2"]
        else:
            active_objs.append(c)

    with open(xy_csv_path, "w", newline="") as fxy, open(depths_csv_path, "w", newline="") as fz:
        # xy_com.csv header
        xy_writer = csv.DictWriter(fxy, fieldnames=["filename"] + xy_colors_columns)
        xy_writer.writeheader()
        # depths.csv header
        depth_cols = ["frame"]
        for k in xy_colors_columns:
            depth_cols += [f"{k}_min_mm", f"{k}_max_mm", f"{k}_com_mm"]
        depth_cols += ["mean_com_mm"]
        z_writer = csv.DictWriter(fz, fieldnames=depth_cols)
        z_writer.writeheader()

        # Untouched state
        obj_order = xy_colors_columns
        states = {obj: UntouchedState() for obj in obj_order}
        untouched_out = {obj: [] for obj in obj_order}
        checking_out = {obj: [] for obj in obj_order}

        # Warmup counters for starting intervals without lookback buffer (see note in update_untouched_states)
        warmup_good_count = {obj: 0 for obj in obj_order}

        # Tracking caches
        prev_two_yellow = None
        prev_gray_accept_xy = None  # (cx,cy)
        prev_gray_accept_frame = None

        # Process frames
        it = color_files if args.no_progress else tqdm(color_files, desc="Processing frames", unit="frame")
        for fname in it:
            fnum = parse_frame_num(fname)
            if fnum is None:
                continue

            # Load frames
            color_full = cv2.imread(os.path.join(args.col_dir, fname))
            if color_full is None:
                continue
            # Crop for detection/tracking
            color = crop_img(color_full.copy())
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

            # Build masks & track yellow
            masks, prev_two_yellow = compute_masks(color, hsv, args.colors, fnum, prev_two_yellow)

            # Save masks (ROI-sized)
            for obj, m in masks.items():
                save_dir = os.path.join(mask_root, obj)
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, fname), m)

            # Build bboxes and COMs for this frame (in cropped coordinates)
            frame_xy: Dict[str, Optional[Tuple[int,int,int,int]]] = {k: None for k in xy_colors_columns}
            for obj in xy_colors_columns:
                if obj not in masks:
                    continue
                bb = bbox_from_mask(masks[obj])
                if bb is not None:
                    frame_xy[obj] = bb

            # Capture reference W×H at ref_frame to seed untouched bbox sizes
            if fnum == args.ref_frame:
                for obj in obj_order:
                    bb = frame_xy.get(obj)
                    if bb is None:
                        continue
                    x,y,w,h = bb
                    states[obj].ref_wh = (float(w), float(h))

            # ---- xy_com.csv row (with gray jump suppression after frame 100) ----
            row_xy = {"filename": fname}
            for obj in xy_colors_columns:
                bb = frame_xy.get(obj)
                if bb is None:
                    row_xy[obj] = ""
                    continue
                cx, cy = com_from_bbox(bb)
                x,y,w,h = bb
                val = f"({cx},{cy},{w},{h})"

                if obj == "gray":
                    if fnum >= GRAY_JUMP_START and prev_gray_accept_xy is not None:
                        jump = math.hypot(cx - prev_gray_accept_xy[0], cy - prev_gray_accept_xy[1])
                        if jump > GRAY_JUMP_THRESH:
                            # suppress gray this frame
                            row_xy[obj] = ""
                        else:
                            row_xy[obj] = val
                            prev_gray_accept_xy = (cx, cy)
                            prev_gray_accept_frame = fnum
                    else:
                        row_xy[obj] = val
                        prev_gray_accept_xy = (cx, cy)
                        prev_gray_accept_frame = fnum
                else:
                    row_xy[obj] = val
            xy_writer.writerow(row_xy)

            # ---- depths.csv row ----
            # find matching depth
            base = os.path.splitext(fname)[0]
            depth_path = None
            for ext in (".tiff",".tif",".png",".jpg",".jpeg"):
                cand = os.path.join(args.dep_dir, base + ext)
                if os.path.exists(cand):
                    depth_path = cand
                    break
            if depth_path is None:
                # write empty row (frame only)
                row_z = {"frame": base}
                for obj in xy_colors_columns:
                    row_z[f"{obj}_min_mm"] = ""
                    row_z[f"{obj}_max_mm"] = ""
                    row_z[f"{obj}_com_mm"] = ""
                row_z["mean_com_mm"] = ""
                z_writer.writerow(row_z)
                continue

            depth_raw = safe_read_depth(depth_path)
            if depth_raw is None:
                continue
            if crop_box is not None:
                x_min,y_min,x_max,y_max = crop_box
                depth_raw = depth_raw[y_min:y_max, x_min:x_max]
            depth_mm = depth_to_mm(depth_raw, args.depth_units)

            row_z = {"frame": base}
            com_list = []
            # Build current per-object (cx,cy) for untouched update, and z for untouched
            xy_curr_for_state: Dict[str, Optional[Tuple[int,int]]] = {}
            z_curr_for_state: Dict[str, Optional[float]] = {}

            for obj in xy_colors_columns:
                # mask read from in-memory if exists else zeros
                m = masks.get(obj, None)
                h,w = color.shape[:2]
                if m is None:
                    mask_bin = np.zeros((h,w), dtype=np.uint8)
                else:
                    mask_bin = m

                zmin,zmax,zcom = compute_z_stats_mm(depth_mm, mask_bin)
                row_z[f"{obj}_min_mm"] = "" if zmin is None else f"{zmin:.3f}"
                row_z[f"{obj}_max_mm"] = "" if zmax is None else f"{zmax:.3f}"
                row_z[f"{obj}_com_mm"] = "" if zcom is None else f"{zcom:.3f}"
                if zcom is not None:
                    com_list.append(zcom)

                # states input (COM in cropped coordinates)
                bb = frame_xy.get(obj)
                xy_curr_for_state[obj] = None if bb is None else com_from_bbox(bb)
                z_curr_for_state[obj] = zcom

            row_z["mean_com_mm"] = "" if not com_list else f"{np.mean(com_list):.3f}"
            z_writer.writerow(row_z)

            # ---- Untouched update (online) ----
            # Warmup: we approximate your p-lookback by requiring p//2 consecutive good-evaluable frames from now.
            # A "good-evaluable" frame = has xy+z, inside its candidate ref-size bbox, and depth_ok.
            for obj in obj_order:
                st = states[obj]
                xy = xy_curr_for_state[obj]
                zc = z_curr_for_state[obj]
                if st.ref_wh is None:
                    continue
                if xy is None or zc is None:
                    warmup_good_count[obj] = 0
                    continue
                cx, cy = xy
                w_ref, h_ref = st.ref_wh
                cand_xmin = cx - w_ref / 2.0
                cand_xmax = cx + w_ref / 2.0
                cand_ymin = cy - h_ref / 2.0
                cand_ymax = cy + h_ref / 2.0
                inside = (cand_xmin <= cx <= cand_xmax) and (cand_ymin <= cy <= cand_ymax)
                z_ok = depth_ok_for(obj, zc)
                if inside and z_ok:
                    warmup_good_count[obj] += 1
                else:
                    warmup_good_count[obj] = 0

                if (not st.interval_active) and (fnum >= int(args.start_time*args.fps)) and (warmup_good_count[obj] >= max(1, args.p//2)):
                    # START interval now; freeze bbox at current xy with ref size (cropped coords)
                    st.interval_active = True
                    st.bbox = (cand_xmin, cand_ymin, cand_xmax, cand_ymax)
                    st.x_start = fnum
                    st.last_good = fnum
                    st.check_start = None
                    st.consec_bad = 0

            # Now use the strict evaluator for active intervals (bad/good/q)
            update_untouched_states(
                states=states,
                obj_order=obj_order,
                fnum=fnum,
                fps=args.fps,
                start_time=args.start_time,
                p=args.p,
                q=args.q,
                xy_dict=xy_curr_for_state,
                z_dict=z_curr_for_state,
                untouched_out=untouched_out,
                checking_out=checking_out
            )

        # Finalize: close any open intervals
        last_frame = parse_frame_num(color_files[-1]) or 0
        for obj, st in states.items():
            if st.interval_active:
                # If a fixed bucket is still open at the file end, clamp and decide
                if st.check_start is not None and st.check_end is not None:
                    # Clamp the bucket end to the video end
                    bucket_end = min(st.check_end, last_frame)
                    if st.saw_any_bad:
                        end_frame = st.last_good if st.last_good is not None else (st.x_start - 1 if st.x_start else bucket_end)
                        if st.x_start is not None and end_frame is not None and end_frame >= st.x_start:
                            untouched_out[obj].append([st.x_start-args.p, end_frame])
                        checking_out[obj].append([st.check_start, bucket_end])
                        # reset interval
                        st.interval_active = False
                        st.bbox = None
                        st.x_start = None
                        st.last_good = None
                    else:
                        # benign bucket → keep untouched through last video frame
                        checking_out[obj].append([st.check_start, bucket_end])
                        # extend last_good as benign
                        st.last_good = last_frame

                    st.check_start = None
                    st.check_end = None
                    st.saw_any_bad = False

                # Commit any still-open untouched interval
                if st.interval_active:
                    end_frame = st.last_good if st.last_good is not None else (st.x_start - 1 if st.x_start else last_frame)
                    if st.x_start is not None and end_frame is not None and end_frame >= st.x_start:
                        untouched_out[obj].append([st.x_start-args.p, end_frame])

    # Write untouched_intervals_xyz.txt
    out_txt = os.path.join(args.out_dir, "untouched_intervals_xyz.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("📌 Untouched intervals (confirmed):\n")
        for obj in xy_colors_columns:
            spans = untouched_out.get(obj, [])
            if not spans:
                f.write(f"{obj}: None\n")
            else:
                parts = [f"[{a}, {b}]" for a,b in spans]
                f.write(f"{obj}: {', '.join(parts)}\n")
    print(f"\n📝 Untouched intervals saved → {out_txt}")
    print(f"✅ xy_com.csv → {xy_csv_path}")
    print(f"✅ depths.csv → {depths_csv_path}")
    print(f"✅ masks     → {mask_root}")
    if crop_box:
        print(f"✅ crop box  → {mask_txt_path}")

    # Render final overlay video (full frame)
    video_name = "untouched_overlay.mp4"
    render_overlay_video(
        col_dir=args.col_dir,
        mask_root=mask_root,
        crop_box=crop_box,
        untouched_out=untouched_out,
        checking_out=checking_out,
        output_dir=args.out_dir,
        video_name=video_name,
        fps=args.fps,
        no_progress=args.no_progress
    )

if __name__ == "__main__":
    main()
