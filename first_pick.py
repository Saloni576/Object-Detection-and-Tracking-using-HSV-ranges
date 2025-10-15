#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import re
import pandas as pd
import cv2.aruco as aruco

# ----------------- Argument Parser -----------------
parser = argparse.ArgumentParser(description="Detect objects from masks, log first XY + Z movement for all colors")
parser.add_argument("--in_dir", help="Path to folder containing input images (PNG)")
parser.add_argument("--mask_dir", help="Path to folder containing mask subfolders")
parser.add_argument("--out_dir", help="Path to output directory for results")
parser.add_argument("--log", help="Path to output log file (txt)")
parser.add_argument("--csv", help="Path to CSV file with z-values")
parser.add_argument("--xy_scale", type=float, default=1.0,
                    help="Fraction of reference box used to detect XY movement (default=0.5)")
args = parser.parse_args()

input_folder = args.in_dir.rstrip("/\\")
mask_dir = args.mask_dir.rstrip("/\\")
output_dir = args.out_dir.rstrip("/\\")
scale = args.xy_scale
os.makedirs(output_dir, exist_ok=True)

FRAMES_THRESHOLD = 60  # number of consecutive frames required to confirm movement


# ===== Helpers for frame sorting =====
INT_RE = re.compile(r"(\d+)")

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(INT_RE, s)]

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

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

# Get sorted list of PNGs
all_pngs = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]
all_pngs.sort(key=natural_key)

# ----------------- Mask Subdirs -----------------
mask_subdirs = {
    "red": "red",
    "green": "green",
    "gray": "gray",
    "yellow_1": "yellow_1",
    "yellow_2": "yellow_2",
    "gold": "gold"
}

color_map = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "gray": (255, 0, 0),
    "yellow_1": (0, 255, 255),
    "yellow_2": (0, 200, 200),
    "gold": (0, 215, 255)
}

# ----------------- Detect from Masks -----------------
def detect_objects_from_masks(frame_name):
    """Returns dict of bounding boxes from masks"""
    results = {}
    kernel = np.ones((3, 3), np.uint8)

    for cname, subdir in mask_subdirs.items():
        mask_path = os.path.join(mask_dir, subdir, frame_name)
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 20:  # filter small noise
            results[cname] = [x, y, x + w, y + h]

    return results

# ----------------- Load Z CSV -----------------
df = pd.read_csv(args.csv)
z_data = {}
for _, row in df.iterrows():
    frame_str = str(row["frame"])
    if frame_str.startswith("frame_"):
        frame_idx = int(frame_str.replace("frame_", ""))
    else:
        frame_idx = int(frame_str)
    z_data[frame_idx] = {
        "red": (row.get("red_min_mm"), row.get("red_max_mm"), row.get("red_com_mm")),
        "green": (row.get("green_min_mm"), row.get("green_max_mm"), row.get("green_com_mm")),
        "gray": (row.get("gray_min_mm"), row.get("gray_max_mm"), row.get("gray_com_mm")),
        "yellow_1": (row.get("yellow_1_min_mm"), row.get("yellow_1_max_mm"), row.get("yellow_1_com_mm")),
        "yellow_2": (row.get("yellow_2_min_mm"), row.get("yellow_2_max_mm"), row.get("yellow_2_com_mm")),
        "gold": (row.get("gold_min_mm"), row.get("gold_max_mm"), row.get("gold_com_mm"))
    }

# ----------------- Step 1: Find Reference Frame -----------------
reference_idx = None
ref_objects = {}
ref_coms = {}
ref_z = {}

def find_frame(idx):
    for f in all_pngs:
        if re.search(rf"frame_0*{idx}\.png$", f):
            return f
    return None

for idx in [100, 99, 98, 97, 96]:
    fname = find_frame(idx)
    if fname is None:
        continue

    objs = detect_objects_from_masks(fname)

    if all(k in objs for k in ["red", "green", "gray", "yellow_1", "yellow_2", "gold"]):
        reference_idx = idx
        ref_objects = {c: objs[c] for c in ["red", "green", "gray", "yellow_1", "yellow_2", "gold"]}
        for c, box in ref_objects.items():
            ref_coms[c] = get_center(box)

        if reference_idx in z_data:
            for c in ["red", "green", "gray", "yellow_1", "yellow_2", "gold"]:
                ref_z[c] = (z_data[reference_idx][c][0], z_data[reference_idx][c][1])

        reference_img = cv2.imread(os.path.join(input_folder, fname))
        reference_img = crop_image(reference_img)
        for c, box in objs.items():
            x1, y1, x2, y2 = box
            cv2.rectangle(reference_img, (x1, y1), (x2, y2), color_map[c], 2)
        break

if reference_idx is None:
    raise RuntimeError("No valid reference frame found near 100.")

# Save reference image
first_mov_dir = os.path.join(output_dir, "first_mov")
os.makedirs(first_mov_dir, exist_ok=True)
ref_fname = find_frame(reference_idx)
cv2.imwrite(os.path.join(first_mov_dir, ref_fname), reference_img)

# ----------------- Step 2: Scan Forward for First Movement -----------------
colors = ["red", "green", "gray", "yellow_1", "yellow_2", "gold"]
moved_xy = {c: False for c in colors}
moved_z = {c: False for c in colors}

outside_counter = {c: 0 for c in colors}  # counts how many consecutive frames COM is outside

for fname in all_pngs:
    frame_idx = natural_key(fname)[1]
    if frame_idx <= reference_idx:
        continue

    objs = detect_objects_from_masks(fname)
    if not objs:
        continue

    image = cv2.imread(os.path.join(input_folder, fname))
    image = crop_image(image)

    for c in colors:
        if c not in objs or c not in ref_objects:
            continue

        new_com = get_center(objs[c])
        ref_com = ref_coms[c]
        ref_box = ref_objects[c]

        # --- XY Movement ---
        # --- XY Movement (with persistence check) ---
        if not moved_xy[c]:
            x1, y1, x2, y2 = ref_box
            cx, cy = ref_coms[c]

            # Compute scaled bounding box centered around the original center
            w = (x2 - x1) * scale
            h = (y2 - y1) * scale
            sx1 = int(cx - w / 2)
            sx2 = int(cx + w / 2)
            sy1 = int(cy - h / 2)
            sy2 = int(cy + h / 2)

            # Check if COM is inside or outside scaled box
            inside_scaled_box = (sx1 <= new_com[0] <= sx2 and sy1 <= new_com[1] <= sy2)

            if not inside_scaled_box:
                outside_counter[c] += 1
            else:
                outside_counter[c] = 0

            # If outside for 60 consecutive frames, mark as moved
            if outside_counter[c] >= FRAMES_THRESHOLD:
                moved_xy[c] = True
                out_img = image.copy()
                cv2.rectangle(out_img, (x1, y1), (x2, y2), color_map[c], 1)
                cv2.rectangle(out_img, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
                nx1, ny1, nx2, ny2 = objs[c]
                cv2.rectangle(out_img, (nx1, ny1), (nx2, ny2), color_map[c], 2)
                cv2.line(out_img, ref_com, new_com, color_map[c], 2)
                cv2.putText(out_img, f"XY moved (>{FRAMES_THRESHOLD} frames)",
                            (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[c], 2)
                cv2.imwrite(os.path.join(first_mov_dir, f"XY_{c}_{fname}"), out_img)
                print(f"{c} first XY movement (persistent {FRAMES_THRESHOLD} frames) at {fname}")


        # --- Z Movement ---
        if not moved_z[c] and frame_idx in z_data and c in z_data[frame_idx]:
            min_z, max_z, com_z = z_data[frame_idx][c]
            ref_min_z, ref_max_z = ref_z.get(c, (None, None))
            if com_z and ref_min_z and com_z < ref_min_z:
                moved_z[c] = True
                out_img = image.copy()
                nx1, ny1, nx2, ny2 = objs[c]
                cv2.rectangle(out_img, (nx1, ny1), (nx2, ny2), color_map[c], 2)
                cv2.putText(out_img, f"Z moved: {com_z:.1f} < {ref_min_z:.1f}",
                            (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[c], 2)
                cv2.imwrite(os.path.join(first_mov_dir, f"Z_{c}_{fname}"), out_img)
                print(f"{c} first Z movement at {fname}")

    # Stop early if all captured
    if all(moved_xy.values()) and all(moved_z.values()):
        print("✅ All XY and Z movements captured. Stopping.")
        break

print("Done. Results saved to:", first_mov_dir)
