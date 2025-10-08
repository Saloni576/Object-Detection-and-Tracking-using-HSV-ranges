#!/usr/bin/env python3
"""
Convert a folder of .png color frames into an .mp4 video.

Usage:
    python png_to_mp4.py --input /path/to/frames --output /path/to/video.mp4 --fps 30
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm   # progress bar

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PNG frames to MP4 video")
    parser.add_argument("--input", required=True, help="Path to folder containing .png frames")
    parser.add_argument("--output", required=True, help="Output video file name (e.g., video.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input)

    # Get all .png files sorted by name
    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        print(f"No .png frames found in {input_dir}")
        return

    # Read first frame to get resolution
    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape

    # Ensure output directory exists
    os.makedirs(Path(args.output).parent, exist_ok=True)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    # Write frames with progress bar
    for frame_path in tqdm(frames, desc="Writing frames", unit="frame"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Skipping unreadable frame {frame_path}")
            continue
        out.write(frame)

    out.release()
    print(f"✅ Video saved to {args.output}")

if __name__ == "__main__":
    main()