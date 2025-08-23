import sys
from pathlib import Path

# --- add ../inc to sys.path so we can import local modules like analyze ---
# Resolve this file's directory; fallback to CWD if __file__ is not available
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

INC_DIR = (HERE / ".." / "inc").resolve()
if str(INC_DIR) not in sys.path:
    sys.path.insert(0, str(INC_DIR))

# Now we can import from analyze.py inside ../inc
from analyze import process_channel, compose_rgb, intensity_rgb
from info_czi import parse_czi_metadata_from_file

import cv2
from aicspylibczi import CziFile
import numpy as np
from typing import List

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main(czi_path: str, channels: List[int], z_index: int, out_png: str = "result.png") -> None:
    # Read CZI
    czi = CziFile(czi_path)
    image_array, _ = czi.read_image()  # shape (0, 0, 0, ch, z, h, w)

    # Process selected channels
    results = []
    for ch in channels:
        results.append(process_channel(image_array, ch, z_index))

    # Expect exactly two channels: [dead, live] like original code
    if len(results) != 2:
        raise ValueError("Expected exactly two channels in `channels` for composition (dead, live).")

    dead = results[0]
    live = results[1]

    rgb_image = compose_rgb(
        dead_brightest=dead["brightest_u8"],
        live_brightest=live["brightest_u8"],
        dead_count=dead["num_skeletons"],
        live_count=live["num_skeletons"],
    )

    cv2.imwrite(out_png, rgb_image)

    # --- print proportions instead of raw counts ---
    dead_count = dead["num_skeletons"]
    live_count = live["num_skeletons"]
    total = dead_count + live_count

    if total > 0:
        dead_prop = dead_count / total
        live_prop = live_count / total
        ratio = dead_count / (live_count if live_count > 0 else 1e-9)
        print(f"Dead: {dead_prop:.2%} ({dead_count}) | Live: {live_prop:.2%} ({live_count}) | Dead/Live ratio: {ratio:.3f}")
    else:
        print("Dead: 0.00% (0) | Live: 0.00% (0) | Dead/Live ratio: N/A")

    percentage, _, _ = intensity_rgb(rgb_image)
    print(percentage)

    metadata = parse_czi_metadata_from_file(czi_path)
    print(metadata)

if __name__ == "__main__":
    # Example usage, keep your original path and channel order (dead first, live second)
    czi_path = "/home/nguyen/Biofilms_git/BioFilmsCZI/Biofilms/image/Bi-Bag-P-P1-3h-4.czi"
    selected_channels = [0, 1]  # [dead, live]
    selected_z = 0
    main(czi_path, selected_channels, selected_z, out_png="result.png")
