#!/usr/bin/env python3
import os
import cv2
import ast
import pandas as pd
import argparse
import numpy as np
from collections import deque
from tqdm import tqdm  # progress bars

# ---------- Utility Functions ----------

def parse_crop_box(mask_txt_path):
    crop = {}
    with open(mask_txt_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=')
                crop[k.strip()] = int(v.strip())
    return crop['x_min'], crop['y_min'], crop['x_max'], crop['y_max']


def print_untouched_intervals(untouched):
    print("\n📌 Untouched intervals per object:")
    for obj, intervals in untouched.items():
        if not intervals:
            print(f"{obj}: None")
            continue
        ranges = [f"[{start}, {end}]" for start, end in intervals]
        print(f"{obj}: {', '.join(ranges)}")


def load_coms_and_bboxes_from_csv(csv_path, ref_frame):
    df = pd.read_csv(csv_path)
    coms = {obj: {} for obj in df.columns if obj != "filename"}
    bboxes = {}

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Loading COMs & reference bboxes",
                       unit="row"):
        fname = row["filename"]
        frame_idx = int(''.join(filter(str.isdigit, fname)))

        for obj in coms.keys():
            val = row[obj]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                cx, cy, w, h = ast.literal_eval(val)
            except Exception:
                continue

            coms[obj][frame_idx] = (cx, cy)

            # capture a reference width/height at ref_frame for this object
            if frame_idx == ref_frame and obj not in bboxes:
                x_min = cx - w / 2
                x_max = cx + w / 2
                y_min = cy - h / 2
                y_max = cy + h / 2
                bboxes[obj] = (x_min, y_min, x_max, y_max)

    # --- Fill missing frames with None so all objects have a contiguous range ---
    all_frames = [int(''.join(filter(str.isdigit, f))) for f in df["filename"]]
    min_frame, max_frame = min(all_frames), max(all_frames)
    for obj in coms.keys():
        for f in range(min_frame, max_frame + 1):
            if f not in coms[obj]:
                coms[obj][f] = None

    return bboxes, coms


def load_depth_data(depth_csv):
    """
    Loads z-coordinate (depth) COMs from the provided CSV file.
    Produces a dict like: {'red': {frame_idx: depth_val_mm, ...}, ...}
    """
    df = pd.read_csv(depth_csv)
    depth_data = {}

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Loading depth COMs",
                       unit="row"):
        frame_idx = int(''.join(filter(str.isdigit, str(row["frame"]))))

        for col in df.columns:
            if col.endswith("_com_mm"):
                obj = col.replace("_com_mm", "")
                if obj not in depth_data:
                    depth_data[obj] = {}
                val = row[col]
                if not pd.isna(val):
                    depth_data[obj][frame_idx] = float(val)
    return depth_data


# ---------- New: Frame-major streaming analysis ----------

def analyze_untouched_objects_framewise(coms, bboxes, start_time, fps, p, q, depth_data):
    """
    Streaming/Realtime version (frame-major pass):

    • For each frame (ascending), evaluate every object and update a small per-object state machine.
    • Start an untouched interval at frame f IF the previous `window` frames (f-window..f-1) are
      all inside a bbox (center = COM at f, size = reference W×H) AND depth-ok.
        - Missing COM in lookback is treated as "inside" but still requires depth to be present;
          if depth missing, lookback fails (same as your original).
    • While an interval is active, keep a FIXED bbox (f's center, ref W×H).
      - Good evaluable frame (COM+depth, inside & depth-ok) extends the interval and closes any open checking span.
      - Bad evaluable frame (COM+depth, outside OR depth-bad) increases a consecutive-bad counter.
        When it reaches `window`, the interval ends at the last good frame, and the checking span closes at the
        decision frame.
      - Frames with missing COM or missing depth are ignored (do not extend, do not increment/reset counters).
    • Returns two dicts: untouched spans and checking spans per object.
    """
    # Build global frame range
    any_obj = next(iter(coms))
    frames_sorted = sorted(coms[any_obj].keys())
    min_frame, max_frame = frames_sorted[0], frames_sorted[-1]
    start_frame = int(start_time * fps)

    # State per object
    state = {}
    untouched = {obj: [] for obj in coms.keys()}
    checking = {obj: [] for obj in coms.keys()}

    for obj, com_dict in coms.items():
        if obj not in bboxes:
            continue
        x_min, y_min, x_max, y_max = bboxes[obj]
        ref_w = x_max - x_min
        ref_h = y_max - y_min

        state[obj] = dict(
            interval_active=False,
            bbox=None,                 # (xmin, ymin, xmax, ymax) for the ACTIVE interval
            x_start=None,
            last_good=None,
            check_start=None,
            consec_bad=0
        )

    def depth_ok_for(obj_name, z):
        if obj_name.startswith("yellow"):
            return 720 <= z <= 760
        else:
            return 680 <= z <= 760

    # ---- Main frame-major loop ----
    pbar = tqdm(range(min_frame, max_frame + 1), desc="Analyzing (frame-major)", unit="frame")
    for f in pbar:
        for obj, com_dict in coms.items():
            # If we don't have a reference bbox (no size), we can't evaluate this object at all.
            if obj not in bboxes:
                continue

            st = state[obj]
            # ---------- CASE A: interval active → extend/close ----------
            if st["interval_active"]:
                # Only evaluate if COM and depth exist at f (evaluable frame)
                xy = com_dict.get(f, None)
                z_available = (obj in depth_data and f in depth_data[obj])

                if xy is None or not z_available:
                    # ignore this frame in the forward check
                    continue

                fx, fy = xy
                z = depth_data[obj][f]
                xmin, ymin, xmax, ymax = st["bbox"]
                inside = (xmin <= fx <= xmax) and (ymin <= fy <= ymax)
                z_ok = depth_ok_for(obj, z)

                if inside and z_ok:
                    # good frame: extend last_good, close any checking span
                    st["last_good"] = f
                    if st["check_start"] is not None:
                        checking[obj].append([st["check_start"], f - 1])
                        st["check_start"] = None
                    st["consec_bad"] = 0
                else:
                    # bad evaluable frame: start/extend checking
                    if st["check_start"] is None:
                        st["check_start"] = f
                    st["consec_bad"] += 1
                    if st["consec_bad"] >= q:
                        # Before finalizing, do a lookahead check
                        lookahead_frames = range(f + 1, f + q)
                        detected_inside_found = False

                        for fa in lookahead_frames:
                            if fa > max_frame:
                                break
                            xy_fa = com_dict.get(fa, None)
                            if xy_fa is None:
                                continue  # only consider frames where COM exists

                            fx_a, fy_a = xy_fa
                            if obj in depth_data and fa in depth_data[obj]:
                                z_a = depth_data[obj][fa]
                                if depth_ok_for(obj, z_a):
                                    xmin, ymin, xmax, ymax = st["bbox"]
                                    inside = (xmin <= fx_a <= xmax) and (ymin <= fy_a <= ymax)
                                    if inside:
                                        detected_inside_found = True
                                        break

                        if detected_inside_found:
                            # treat as outlier — do NOT end interval
                            st["consec_bad"] = 0
                            st["check_start"] = None
                            continue  # keep interval active

                        # else, all detected COMs were outside bbox → end interval normally
                        end_frame = st["last_good"] if st["last_good"] is not None else (st["x_start"] - 1)
                        if end_frame >= st["x_start"]:
                            untouched[obj].append([st["x_start"], end_frame])
                        checking[obj].append([st["check_start"], f])
                        # reset state
                        st["interval_active"] = False
                        st["bbox"] = None
                        st["x_start"] = None
                        st["last_good"] = None
                        st["check_start"] = None
                        st["consec_bad"] = 0

                continue  # done with active interval

            # ---------- CASE B: interval NOT active → test for START (only when f >= start_frame) ----------
            if f < start_frame:
                continue

            xy_f = com_dict.get(f, None)
            if xy_f is None:
                # cannot seed a bbox without a COM at f
                continue

            cx, cy = xy_f
            # Candidate FIXED bbox (if we start now), centered at COM[f] with reference W×H
            cand_xmin = cx - ref_w / 2.0
            cand_xmax = cx + ref_w / 2.0
            cand_ymin = cy - ref_h / 2.0
            cand_ymax = cy + ref_h / 2.0

            # Lookback: previous `window` frames must be evaluable & OK
            lookback_frames = list(range(f - p, f))
            inside_ok = True
            observed = 0
            for pf in lookback_frames:
                # We treat the lookback strictly by the original rules:
                # - Missing COM is "inside" by position, BUT we still require depth; if depth missing → fail.
                # - If COM present, it must lie inside cand bbox.
                # - Depth must be present and within the allowed range.
                # If pf < min_frame, fail (we can't satisfy a full lookback before the timeline).
                if pf < min_frame:
                    inside_ok = False
                    break

                depth_avail = (obj in depth_data and pf in depth_data[obj])
                if not depth_avail:
                    inside_ok = False
                    break
                z_pf = depth_data[obj][pf]
                z_ok = depth_ok_for(obj, z_pf)
                if not z_ok:
                    inside_ok = False
                    break

                xy_pf = com_dict.get(pf, None)
                if xy_pf is not None:
                    px, py = xy_pf
                    if not (cand_xmin <= px <= cand_xmax and cand_ymin <= py <= cand_ymax):
                        inside_ok = False
                        break

                observed += 1  # count frames we checked

            if inside_ok and observed == p:
                # Start interval. x_start matches your previous behavior:
                # max(start_frame, first lookback frame) if exist, else f.
                x_start = max(start_frame, lookback_frames[0]) if lookback_frames else f
                state[obj].update(
                    interval_active=True,
                    bbox=(cand_xmin, cand_ymin, cand_xmax, cand_ymax),
                    x_start=x_start,
                    last_good=f,          # current frame itself is good by definition (starts extension)
                    check_start=None,
                    consec_bad=0
                )
                # Note: we don't append anything now; we’ll finalize when it ends.

    # ---- Finalize: if any interval remains open at the end, close it safely ----
    for obj, st in state.items():
        if st["interval_active"]:
            # If a checking window is open but never reached p-bad, we close it at max_frame- if it existed
            if st["check_start"] is not None and st["last_good"] is not None and st["check_start"] <= st["last_good"]:
                checking[obj].append([st["check_start"], st["last_good"]])
            # Commit the untouched span up to last_good (if any)
            end_frame = st["last_good"] if st["last_good"] is not None else (st["x_start"] - 1)
            if end_frame >= st["x_start"]:
                untouched[obj].append([st["x_start"], end_frame])

    return untouched, checking


# --- Helper for semi-transparent drawing ---
def _draw_transparent_circle(img_bgr, center, radius, color_bgr, alpha=0.45, outline=True):
    """Blend a filled circle with transparency onto img_bgr."""
    overlay = img_bgr.copy()
    cv2.circle(overlay, center, radius, color_bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, dst=img_bgr)
    if outline:
        cv2.circle(img_bgr, center, radius, color_bgr, 1)


def make_untouched_video(in_dir, crop_box, untouched, output_dir, video_name, fps, coms, mask_dir=None, checking=None):
    """
    Creates a video with:
    - Cropped frames
    - Overlayed masks from mask_dir (ROI-sized; if mismatch, fallback-crop using mask_txt_path bounds)
    - Solid (untouched) / semi-transparent (checking) / empty (moving) circles
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, video_name)

    x_min, y_min, x_max, y_max = crop_box
    frame_files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not frame_files:
        raise RuntimeError("No frames found in input directory")

    sample = cv2.imread(os.path.join(in_dir, frame_files[0]))
    crop_w, crop_h = x_max - x_min, y_max - y_min
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

    # ---------- Single source of truth for colors (OpenCV BGR) ----------
    COLOR_MAP = {
        "red":      (0, 255, 0),    # red object → green
        "green":    (0, 0, 255),    # green object → red
        "gray":     (0, 255, 255),  # gray object → yellow
        "yellow_1": (255, 0, 0),    # 1st yellow → blue
        "yellow_2": (0, 128, 255),  # 2nd yellow → orange
        "gold":     (128, 0, 255),  # golden object → violet
    }
    objects = list(COLOR_MAP.keys())

    circle_radius = 6
    start_x = 20
    spacing = 40
    baseline_y = 50  # dots just below frame number

    for fname in tqdm(frame_files, desc="Rendering video frames", unit="frame"):
        frame_idx = int(''.join(filter(str.isdigit, fname)))
        frame_path = os.path.join(in_dir, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Crop frame
        x_min, y_min, x_max, y_max = crop_box
        cropped = frame[y_min:y_max, x_min:x_max].copy()
        overlay = cropped.copy()

        # --- Overlay masks (contours + semi-transparent fill + outline) ---
        if mask_dir:
            overlay_img = overlay.copy()
            for obj in objects:
                mpath = os.path.join(mask_dir, obj, fname)
                if not os.path.exists(mpath):
                    continue
                mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                # If mask is full-frame by mistake, crop it to the ROI as a fallback
                if mask.shape[:2] != overlay.shape[:2]:
                    if mask.shape[0] >= y_max and mask.shape[1] >= x_max:
                        mask = mask[y_min:y_max, x_min:x_max]
                    else:
                        print(f"[warn] Mask size mismatch for {obj} @ {fname}: {mask.shape} != {overlay.shape[:2]}")
                        continue

                color = COLOR_MAP.get(obj, (255, 255, 255))

                # Contours (largest only)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                contours_to_draw = [c]

                # Semi-transparent fill
                fill = np.zeros_like(overlay, dtype=np.uint8)
                cv2.drawContours(fill, contours_to_draw, -1, color, thickness=cv2.FILLED)
                overlay_img = cv2.addWeighted(overlay_img, 1.0, fill, 0.4, 0)

                # Outline
                cv2.drawContours(overlay_img, contours_to_draw, -1, color, thickness=2)

            overlay = overlay_img

        # --- Draw frame number ---
        cv2.putText(overlay, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # --- Draw per-object state circles ---
        for i, obj in enumerate(objects):
            cx = start_x + i * spacing
            cy = baseline_y
            color = COLOR_MAP[obj]

            obj_untouched = untouched.get(obj, [])
            obj_checking  = (checking or {}).get(obj, [])

            is_untouched = any(x <= frame_idx <= y for (x, y) in obj_untouched)
            is_checking  = any(x <= frame_idx <= y for (x, y) in obj_checking)

            if is_checking:
                _draw_transparent_circle(overlay, (cx, cy), circle_radius, color, alpha=0.45, outline=True)
            elif is_untouched:
                cv2.circle(overlay, (cx, cy), circle_radius, color, -1)  # solid = untouched
            else:
                cv2.circle(overlay, (cx, cy), circle_radius, color, 2)   # empty = moving

            # Optional: label below circle
            cv2.putText(overlay, obj, (cx - 20, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        out.write(overlay)

    out.release()
    print(f"✅ Video saved → {output_path}")


# ---------- Pipeline ----------

def untouched_pipeline(in_dir, csv_path, mask_txt_path, output_dir, start_time,
                       ref_frame, p, q, fps, video_name, depth_csv, mask_dir=None):
    crop_box = parse_crop_box(mask_txt_path)
    bboxes, coms = load_coms_and_bboxes_from_csv(csv_path, ref_frame)
    depth_data = load_depth_data(depth_csv)
    # NEW: frame-major analysis
    untouched, checking = analyze_untouched_objects_framewise(coms, bboxes, start_time, fps, p, q, depth_data)
    make_untouched_video(in_dir, crop_box, untouched, output_dir, video_name, fps, coms, mask_dir=mask_dir, checking=checking)
    return untouched, checking


# ---------- CLI Entry Point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect untouched objects in video frames (with depth filtering and visual markers).")
    parser.add_argument("--in_dir", required=True, help="Input frames directory")
    parser.add_argument("--csv_path", required=True, help="CSV file with object positions")
    parser.add_argument("--mask_txt_path", required=True, help="Mask crop box file")
    parser.add_argument("--output_dir", required=True, help="Output directory for video")
    parser.add_argument("--depth_csv", required=True, help="CSV file containing object depth (z-coordinate) data in mm")
    parser.add_argument("--mask", help="Directory containing binary object masks (optional)")
    parser.add_argument("--start_time", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--ref_frame", type=int, default=100, help="Reference frame index")
    parser.add_argument("--p", type=int, default=10,
                    help="Number of previous frames to check for starting an untouched interval")
    parser.add_argument("--q", type=int, default=10,
                    help="Number of next frames to check for confirming the end of an untouched interval")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--video_name", default="untouched_video_xyz.mp4", help="Output video name")

    args = parser.parse_args()

    untouched_map, checking_map = untouched_pipeline(
        in_dir=args.in_dir,
        csv_path=args.csv_path,
        mask_txt_path=args.mask_txt_path,
        output_dir=args.output_dir,
        start_time=args.start_time,
        ref_frame=args.ref_frame,
        p=args.p,
        q=args.q,
        fps=args.fps,
        video_name=args.video_name,
        depth_csv=args.depth_csv,
        mask_dir=args.mask
    )

    # Save untouched intervals to file (only confirmed untouched, no checking)
    output_txt_path = os.path.join(args.output_dir, "untouched_intervals_xyz.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("📌 Untouched intervals (confirmed):\n")
        for obj, intervals in untouched_map.items():
            if not intervals:
                f.write(f"{obj}: None\n")
                continue
            ranges = [f"[{start}, {end}]" for start, end in intervals]
            f.write(f"{obj}: {', '.join(ranges)}\n")

    print(f"\n📝 Untouched intervals saved to: {output_txt_path}")
