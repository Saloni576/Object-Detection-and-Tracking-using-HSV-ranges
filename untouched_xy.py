import os
import cv2
import ast
import pandas as pd
import argparse
import numpy as np

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

    for _, row in df.iterrows():
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

            if frame_idx == ref_frame and obj not in bboxes:
                x_min = cx - w / 2
                x_max = cx + w / 2
                y_min = cy - h / 2
                y_max = cy + h / 2
                bboxes[obj] = (x_min, y_min, x_max, y_max)

    # --- Fill missing frames with None ---
    all_frames = [int(''.join(filter(str.isdigit, f))) for f in df["filename"]]
    min_frame, max_frame = min(all_frames), max(all_frames)
    for obj in coms.keys():
        for f in range(min_frame, max_frame + 1):
            if f not in coms[obj]:
                coms[obj][f] = None

    return bboxes, coms


def analyze_untouched_objects_dynamic(coms, bboxes, start_time, fps, window):
    """
    Detects 'untouched' intervals for each object by checking:
      - If, in the previous `window` frames, the object's COM stayed within its bbox
        (built dynamically using COM at current frame + width/height from reference),
      - No depth check.
    """
    untouched = {}
    start_frame = int(start_time * fps)

    for obj, com_dict in coms.items():
        if obj not in bboxes:
            continue

        x_min, y_min, x_max, y_max = bboxes[obj]
        width = x_max - x_min
        height = y_max - y_min

        frames = sorted(com_dict.keys())
        untouched[obj] = []

        i = 0
        while i < len(frames):
            frame = frames[i]
            if frame < start_frame or frame not in com_dict:
                i += 1
                continue

            if com_dict[frame] is None:
                i += 1
                continue  # skip this frame for starting untouched interval
            cx, cy = com_dict[frame]
            bbox_xmin = cx - width / 2
            bbox_xmax = cx + width / 2
            bbox_ymin = cy - height / 2
            bbox_ymax = cy + height / 2

            # --- Check previous `window` frames ---
            lookback_indices = [f for f in frames if frame - window <= f < frame]
            inside = True
            for prev_f in lookback_indices:
                px_py = com_dict[prev_f]
                if px_py is None:  # missing COM, treat as inside
                    continue
                px, py = px_py
                if not (bbox_xmin <= px <= bbox_xmax and bbox_ymin <= py <= bbox_ymax):
                    inside = False
                    break

            if inside and len(lookback_indices) >= window:
                x_start = max(start_frame, lookback_indices[0])
                y_end = frame

                # --- Extend forward until object leaves bbox ---
                j = i + 1
                while j < len(frames):
                    fnum = frames[j]
                    fx_fy = com_dict[fnum]
                    if fx_fy is None:  # treat missing COM as still inside
                        y_end = fnum
                        j += 1
                        continue

                    fx, fy = fx_fy
                    inside_box = bbox_xmin <= fx <= bbox_xmax and bbox_ymin <= fy <= bbox_ymax

                    if inside_box:
                        y_end = fnum
                        j += 1
                    else:
                        break

                untouched[obj].append([x_start, y_end])
                i = j + window
            else:
                i += 1

    return untouched


def make_untouched_video(in_dir, crop_box, untouched, output_dir, video_name, fps, coms, mask_dir=None):
    """
    Creates a video with:
    - Cropped frames
    - Overlayed masks from mask_dir (optional)
    - Solid/empty circles indicating untouched/moving status
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, video_name)

    x_min, y_min, x_max, y_max = crop_box
    frame_files = sorted([f for f in os.listdir(in_dir) if f.endswith(('.png', '.jpg'))])

    if not frame_files:
        raise RuntimeError("No frames found in input directory")

    sample = cv2.imread(os.path.join(in_dir, frame_files[0]))
    crop_w, crop_h = x_max - x_min, y_max - y_min
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

    # Color mapping for circles
    circle_colors = {
        "red": (0, 255, 0),       # red → green
        "green": (0, 0, 255),     # green → red
        "gray": (0, 255, 255),    # gray → yellow
        "yellow_1": (255, 0, 0),  # yellow_1 → blue
        "yellow_2": (0, 128, 255),# yellow_2 → orange
        "gold": (128, 0, 255)     # gold → violet
    }

    objects = list(circle_colors.keys())
    circle_radius = 8
    start_x = 20
    spacing = 55
    baseline_y = 50  # dots just below frame number

    for fname in frame_files:
        frame_idx = int(''.join(filter(str.isdigit, fname)))
        frame_path = os.path.join(in_dir, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Crop frame
        cropped = frame[y_min:y_max, x_min:x_max].copy()
        overlay = cropped.copy()

        # --- Overlay masks if mask_dir is provided ---
        if mask_dir:
            for obj, color in circle_colors.items():
                mask_obj_dir = os.path.join(mask_dir, obj)
                mask_file = os.path.join(mask_obj_dir, fname)
                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        colored_mask = np.zeros_like(cropped)
                        colored_mask[mask > 0] = color
                        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        # --- Draw frame number ---
        cv2.putText(overlay, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # --- Draw untouched/moving circles below frame number ---
        for i, obj in enumerate(objects):
            cx = start_x + i * spacing
            cy = baseline_y
            color = circle_colors[obj]

            is_untouched = any(x <= frame_idx <= y for (x, y) in untouched.get(obj, []))

            if is_untouched:
                cv2.circle(overlay, (cx, cy), circle_radius, color, -1)  # solid = untouched
            else:
                cv2.circle(overlay, (cx, cy), circle_radius, color, 2)   # empty = moving

            # Optional: label below circle
            cv2.putText(overlay, obj, (cx - 20, cy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        out.write(overlay)

    out.release()
    print(f"✅ Video saved → {output_path}")


# ---------- Pipeline ----------

def untouched_pipeline(in_dir, csv_path, mask_txt_path, output_dir, start_time,
                       ref_frame, window, fps, video_name):
    crop_box = parse_crop_box(mask_txt_path)
    bboxes, coms = load_coms_and_bboxes_from_csv(csv_path, ref_frame)
    untouched = analyze_untouched_objects_dynamic(coms, bboxes, start_time, fps, window)
    make_untouched_video(in_dir, crop_box, untouched, output_dir, video_name, fps, coms, mask_dir=args.mask)
    return untouched


# ---------- CLI Entry Point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect untouched objects in video frames (2D COM only).")
    parser.add_argument("--in_dir", required=True, help="Input frames directory")
    parser.add_argument("--csv_path", required=True, help="CSV file with object positions")
    parser.add_argument("--mask_txt_path", required=True, help="Mask crop box file")
    parser.add_argument("--output_dir", required=True, help="Output directory for video")
    parser.add_argument("--mask", help="Directory containing binary object masks (optional)")
    parser.add_argument("--start_time", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--ref_frame", type=int, default=100, help="Reference frame index")
    parser.add_argument("--window", type=int, default=10, help="Number of frames to check for untouched")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--video_name", default="untouched_video_xy.mp4", help="Output video name")

    args = parser.parse_args()

    untouched_list = untouched_pipeline(
        in_dir=args.in_dir,
        csv_path=args.csv_path,
        mask_txt_path=args.mask_txt_path,
        output_dir=args.output_dir,
        start_time=args.start_time,
        ref_frame=args.ref_frame,
        window=args.window,
        fps=args.fps,
        video_name=args.video_name
    )

    print_untouched_intervals(untouched_list)