import cv2
import numpy as np
import argparse
import os
import re
from collections import Counter, defaultdict
import csv
from scipy.spatial import distance as dist
import math
import cv2.aruco as aruco

# ----------------- Argument Parser -----------------
parser = argparse.ArgumentParser(description="Detect objects and save masks")
parser.add_argument("--in_dir", help="Path to folder containing input images (PNG)")
parser.add_argument("--mask_obj_dir", help="Path to output directory for masked overlays")
parser.add_argument("--mask_dir", help="Path to output directory for binary masks only")
parser.add_argument("--log", help="Path to output log file (txt)")
parser.add_argument(
    "--colors",
    nargs="+",
    choices=["red", "green", "yellow", "gray", "skin", "gold"],
    default=["red", "green", "yellow", "gray", "gold"],
    help="Colors to detect (default: all)"
)
parser.add_argument(
    "--num",
    type=int,
    default=None,
    help="If set, also log which frames have exactly this many detections"
)
parser.add_argument(
    "--csv",
    help="Path to output CSV file with bounding box areas"
)
args = parser.parse_args()

input_folder = args.in_dir.rstrip("/\\")
mask_obj_dir = args.mask_obj_dir.rstrip("/\\")
mask_dir = args.mask_dir.rstrip("/\\")

os.makedirs(mask_obj_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# ===== Helpers for frame sorting =====
INT_RE = re.compile(r"(\d+)")

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(INT_RE, s)]

# Get sorted list of PNGs
all_pngs = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]
all_pngs.sort(key=natural_key)

# Counter for frames by number of detections
frame_counts = Counter()
frames_with_num = defaultdict(list)

# Collect results for CSV
csv_rows = []

# ====== YELLOW TRACKING STATE ======
prev_two_yellow_centroids = None
start_tracking_frame = 101  # tracking starts here

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

global_crop_box = None  # cached bounding box

def detect_and_cache_crop_box(img: np.ndarray):
    """Detect 3-4 ArUco markers, compute bounding box, and cache it."""
    global global_crop_box
    if global_crop_box is not None:
        return global_crop_box

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 3 or len(ids) > 4:
        return None

    selected_points = []
    for i, corner in enumerate(corners):
        marker_id = ids[i][0]
        pts = corner.reshape((4, 2))
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
    global_crop_box = (x_min, y_min, x_max, y_max)
    print(f"✅ Cached ArUco crop box: {global_crop_box}")
    return global_crop_box


def crop_image(img: np.ndarray) -> np.ndarray:
    """Crop color frame according to detected/cached ArUco box."""
    box = detect_and_cache_crop_box(img)
    if box is None:
        return img
    x_min, y_min, x_max, y_max = box
    return img[y_min:y_max, x_min:x_max]

def union_significant_contours(binary_mask: np.ndarray, min_area: int = 65, area_frac_keep: float = 0.35):
    """
    Keep all contours whose area >= area_frac_keep * largest_area (after min_area filter),
    and return a single 0/255 mask that is the union of those contours.
    Returns None if nothing valid.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # remove tiny noise
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

# ======== Detect crop region from ArUco markers (100→50, else 101→end) ========
global_crop_box = None
print("🔍 Searching for ArUco markers to define crop region...")

def _parse_frame_num(fname: str):
    m = re.search(r"frame_(\d+)\.png$", fname, re.IGNORECASE)
    return int(m.group(1)) if m else None

# Prepare list [(idx, fname, num)] restricted to files that match the pattern
annot = [(i, f, _parse_frame_num(f)) for i, f in enumerate(all_pngs)]
annot = [t for t in annot if t[2] is not None]

# Phase 1: check frames 100 ↓ to 50 (inclusive)
phase1 = [t for t in annot if 50 <= t[2] <= 100]
phase1.sort(key=lambda x: x[2], reverse=True)  # 100, 99, …, 50

for _, fname, num in phase1:
    test_img = cv2.imread(os.path.join(input_folder, fname))
    if test_img is None:
        continue
    box = detect_and_cache_crop_box(test_img)  # still requires exactly 3 or 4 markers
    if box is not None:
        print(f"✅ Crop region detected from {fname}: {box}")
        break

# Phase 2: if not found, check frames 101 ↑ to the end
if global_crop_box is None:
    phase2 = [t for t in annot if t[2] >= 101]
    phase2.sort(key=lambda x: x[2])  # 101, 102, …
    for _, fname, num in phase2:
        test_img = cv2.imread(os.path.join(input_folder, fname))
        if test_img is None:
            continue
        box = detect_and_cache_crop_box(test_img)
        if box is not None:
            print(f"✅ Crop region detected from {fname}: {box}")
            break

if global_crop_box is None:
    print("⚠️ No ArUco markers detected in 100→50 or 101→end — proceeding without cropping.")
else:
    print(f"📦 Using cached crop region: {global_crop_box}")

# --- Save crop box details to mask.txt ---
mask_txt_path = os.path.join(mask_dir, "mask.txt")
with open(mask_txt_path, "w") as f:
    if global_crop_box is None:
        f.write("No crop region detected.\n")
    else:
        x_min, y_min, x_max, y_max = global_crop_box
        f.write(f"Crop Box Coordinates:\n")
        f.write(f"x_min={x_min}\n")
        f.write(f"y_min={y_min}\n")
        f.write(f"x_max={x_max}\n")
        f.write(f"y_max={y_max}\n")
print(f"📝 Crop region info saved to {mask_txt_path}")

# ======== Now begin normal processing loop ========
for fname in all_pngs:
    image_path = os.path.join(input_folder, fname)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {fname}, could not load.")
        continue

    # Apply cached crop region if available
    image = crop_image(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    # ---- Masks storage ----
    masks = {}

    # --- Red (largest object) ---
    if "red" in args.colors:
        mask_red1 = cv2.inRange(hsv, np.array([0, 120, 110]), np.array([10, 240, 235]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 120, 110]), np.array([180, 240, 235]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        # NEW: keep union of significant contours (handles split halves)
        mask_red_final = union_significant_contours(mask_red, min_area=80, area_frac_keep=0.35)
        if mask_red_final is not None:
            masks["red"] = mask_red_final

    # --- Green (largest object) ---
    if "green" in args.colors:
        mask_green = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([70, 150, 190]))
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        # NEW: keep union of significant contours (handles split halves)
        mask_green_final = union_significant_contours(mask_green, min_area=80, area_frac_keep=0.35)
        if mask_green_final is not None:
            masks["green"] = mask_green_final

    # --- Gray (largest object) ---
    if "gray" in args.colors:
        # mask_gray = cv2.inRange(hsv, np.array([10, 0, 90]), np.array([100, 80, 160]))
        mask_gray = cv2.inRange(hsv, np.array([10, 0, 90]), np.array([100, 50, 160]))
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out contours with points in forbidden area
        filtered_contours = []
        for c in contours:
            if not any((pt[0][0] < 60 and pt[0][1] > 300) for pt in c):
                filtered_contours.append(c)

        # --- Ignore very small contours ---
        min_area = 65  # tune this depending on your image size
        filtered_contours = [c for c in filtered_contours if cv2.contourArea(c) > min_area]

        if filtered_contours:
            c = max(filtered_contours, key=cv2.contourArea)
            mask_gray_final = np.zeros_like(mask_gray)
            cv2.drawContours(mask_gray_final, [c], -1, 255, -1)
            masks["gray"] = mask_gray_final

    # --- Yellow (tracking logic) ---
    if "yellow" in args.colors:
        frame_idx = int(fname.split('_')[-1].split('.')[0])
        mask_yellow = cv2.inRange(hsv, np.array([18, 185, 160]), np.array([30, 255, 255]))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 65]

        yellow_centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                yellow_centroids.append((cx, cy))

        overlay_debug = image.copy()

        if frame_idx < start_tracking_frame:
            # before 101, just label by order
            for i, c in enumerate(contours[:2]):
                mask_final = np.zeros_like(mask_yellow)
                cv2.drawContours(mask_final, [c], -1, 255, -1)
                masks[f"yellow_{i+1}"] = mask_final
            if len(yellow_centroids) == 2:
                prev_two_yellow_centroids = (yellow_centroids[0], yellow_centroids[1])
        else:
            if len(yellow_centroids) == 2:
                # Case 1
                if prev_two_yellow_centroids is not None:
                    (x1, y1), (x2, y2) = prev_two_yellow_centroids
                    (x3, y3), (x4, y4) = yellow_centroids
                    d13 = euclidean_dist((x1, y1), (x3, y3))
                    d14 = euclidean_dist((x1, y1), (x4, y4))
                    if d13 < d14:
                        yellow1, yellow2 = (x3, y3), (x4, y4)
                    else:
                        yellow1, yellow2 = (x4, y4), (x3, y3)
                    prev_two_yellow_centroids = (yellow1, yellow2)
                else:
                    yellow1, yellow2 = yellow_centroids[0], yellow_centroids[1]
                    prev_two_yellow_centroids = (yellow1, yellow2)

                # draw masks
                for i, c in enumerate(contours[:2]):
                    mask_final = np.zeros_like(mask_yellow)
                    cv2.drawContours(mask_final, [c], -1, 255, -1)
                    masks[f"yellow_{i+1}"] = mask_final

                # --- Debug: draw centroids and lines ---
                cv2.circle(overlay_debug, yellow1, 5, (255, 0, 0), -1)
                cv2.circle(overlay_debug, yellow2, 5, (255, 0, 255), -1)
                cv2.line(overlay_debug, prev_two_yellow_centroids[0], yellow1, (255, 255, 0), 2)
                cv2.line(overlay_debug, prev_two_yellow_centroids[1], yellow2, (255, 255, 0), 2)

            elif len(yellow_centroids) == 1 and prev_two_yellow_centroids is not None:
                # Case 2
                (x1, y1), (x2, y2) = prev_two_yellow_centroids
                (x5, y5) = yellow_centroids[0]
                d1 = euclidean_dist((x1, y1), (x5, y5))
                d2 = euclidean_dist((x2, y2), (x5, y5))
                if d1 < d2:
                    label = "yellow_1"
                    prev_two_yellow_centroids = ((x5, y5), (x2, y2))
                    cv2.circle(overlay_debug, (x5, y5), 5, (255, 0, 0), -1)
                else:
                    label = "yellow_2"
                    prev_two_yellow_centroids = ((x1, y1), (x5, y5))
                    cv2.circle(overlay_debug, (x5, y5), 5, (255, 0, 255), -1)

                mask_final = np.zeros_like(mask_yellow)
                cv2.drawContours(mask_final, [contours[0]], -1, 255, -1)
                masks[label] = mask_final

            # else no yellow detected

    # --- Gold (largest object) ---
    if "gold" in args.colors:
        mask_gold = cv2.inRange(hsv, np.array([18, 100, 120]), np.array([23, 190, 190]))
        # mask_gold = cv2.inRange(hsv, np.array([15, 120, 140]), np.array([25, 190, 190]))
        mask_gold = cv2.morphologyEx(mask_gold, cv2.MORPH_OPEN, kernel)
        mask_gold = cv2.morphologyEx(mask_gold, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            min_area = 65  # adjust as needed
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if contours:
            c = max(contours, key=cv2.contourArea)
            mask_gold_final = np.zeros_like(mask_gold)
            cv2.drawContours(mask_gold_final, [c], -1, 255, -1)
            masks["gold"] = mask_gold_final

    # --- Skin (top 3 objects) ---
    if "skin" in args.colors:
        mask_skin = cv2.inRange(hsv, np.array([15, 80, 220]), np.array([20, 140, 260]))
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            for i, c in enumerate(contours):
                mask_skin_final = np.zeros_like(mask_skin)
                cv2.drawContours(mask_skin_final, [c], -1, 255, -1)
                masks[f"skin_{i+1}"] = mask_skin_final

    # ---- Overlay masks on original image ----
    num_detections = 0

    # Overlay color mapping (BGR)
    color_map = {
        "red": (0, 255, 0),       # red object → green
        "green": (0, 0, 255),     # green object → red
        "gray": (0, 255, 255),    # gray object → yellow
        "yellow_1": (255, 0, 0),    # 1st yellow → blue
        "yellow_2": (255, 0, 255),   # 2nd yellow → magenta
        "gold": (128, 0, 255),        # golden object → violet
        "skin_1": (0, 128, 255),      # 1st skin → orange
        "skin_2": (0, 255, 128),      # 2nd skin → spring green
        "skin_3": (255, 255, 0),      # 3rd skin → cyan-yellow
        "white": (255, 128, 64)       # white object → warm coral/orange-pink
    }

    overlay_img = image.copy()

    for col, mask in masks.items():
        color = color_map.get(col, (255, 255, 255))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Take largest contour for reporting
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # Save row for CSV
        csv_rows.append({
            "filename": fname,
            "color": col,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": area
        })

        # Semi-transparent fill
        fill = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(fill, contours, -1, color, thickness=cv2.FILLED)
        overlay_img = cv2.addWeighted(overlay_img, 1.0, fill, 0.4, 0)

        # Draw contour outline
        cv2.drawContours(overlay_img, contours, -1, color, thickness=2)

        # Save binary mask
        save_mask_dir = os.path.join(mask_dir, col)
        os.makedirs(save_mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_mask_dir, fname), mask)

        num_detections += 1

    # Save final overlay image with all objects
    save_obj_dir = os.path.join(mask_obj_dir)
    os.makedirs(save_obj_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_obj_dir, fname), overlay_img)

    frame_counts[num_detections] += 1
    if args.num is not None and num_detections == args.num:
        frames_with_num[args.num].append(fname)

    print(f"Processed {fname} | Detections: {num_detections}")

# --- Write summary log safely ---
if args.log:
    # Ensure log directory exists
    log_dir = os.path.dirname(args.log)
    if log_dir:  # only create if a folder path is given
        os.makedirs(log_dir, exist_ok=True)

    # Write log file
    with open(args.log, "w") as f:
        f.write("Detection Summary (frames vs. number of objects):\n")
        for num_objs in sorted(frame_counts.keys()):
            f.write(f"{num_objs} objects: {frame_counts[num_objs]} frames\n")
        if args.num is not None:
            f.write(f"\nFrames with exactly {args.num} objects detected:\n")
            for frame in frames_with_num[args.num]:
                f.write(frame + "\n")

    print(f"\n✅ Log written to {args.log}")
else:
    print("\n⚠️ No --log path provided, skipping log file.")

# --- Save CSV in (filename, red, green, gray, yellow_1, yellow_2, gold) format ---
if args.csv:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # Define desired order of color columns
    color_columns = ["red", "green", "gray", "yellow_1", "yellow_2", "gold"]

    # Group results by filename
    grouped = {}
    for row in csv_rows:
        fname = row["filename"]
        color = row["color"]
        if fname not in grouped:
            grouped[fname] = {}
        # Compute center
        cx = row["x"] + row["width"] // 2
        cy = row["y"] + row["height"] // 2
        grouped[fname][color] = f"({cx},{cy},{row['width']},{row['height']})"

    # Write new CSV
    with open(args.csv, "w", newline="") as csvfile:
        fieldnames = ["filename"] + color_columns
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fname in sorted(grouped.keys()):
            row_data = {"filename": fname}
            for col in color_columns:
                row_data[col] = grouped[fname].get(col, "")
            writer.writerow(row_data)

    print(f"✅ CSV written to {args.csv} with compact bounding box format.")
