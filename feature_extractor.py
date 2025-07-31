import numpy as np
import cv2
import os
import re

FE_FILE = "disposable FEs.txt"
FEs = []

# === Load FEs from File ===
def load_feature_extractors():
    global FEs
    if FEs:  # Already loaded
        return FEs

    if not os.path.exists(FE_FILE):
        print(f"❌ FE file '{FE_FILE}' not found.")
        return []

    with open(FE_FILE, 'r') as f:
        content = f.read()

    blocks = content.split("Evaluations: 1000")
    for block in blocks[1:]:
        if "Best:" in block:
            best_start = block.index("Best:") + len("Best:")
            numbers_str = block[best_start:].strip().replace("\n", " ").split()
            numbers = []

            for token in numbers_str:
                try:
                    numbers.append(float(token))
                except ValueError:
                    break

            if len(numbers) >= 74:
                fe = {
                    "x": numbers[0:24],
                    "y": numbers[24:48],
                    "thresholds": numbers[48:72],
                    "radius_w": int(numbers[72]),
                    "radius_h": int(numbers[73])
                }
                FEs.append(fe)
    return FEs

# === Patch-Based Feature Extraction ===
def extract_feature_vector(image, fe_index=0):
    """
    Extracts a patch-based feature vector from a grayscale image using the selected FE.
    """
    if image is None:
        return []

    fes = load_feature_extractors()
    if not fes:
        print("❌ No FEs loaded.")
        return []

    fe = fes[fe_index % len(fes)]
    x_coords = fe["x"]
    y_coords = fe["y"]
    thresholds = fe["thresholds"]
    rw, rh = fe["radius_w"], fe["radius_h"]

    h, w = image.shape
    fv = []

    for x, y, t in zip(x_coords, y_coords, thresholds):
        if t <= 0.4999:
            fv.append(0.0)
            continue

        x, y = int(round(x)), int(round(y))
        x1, y1 = max(0, x - rw), max(0, y - rh)
        x2, y2 = min(w, x + rw), min(h, y + rh)

        patch = image[y1:y2, x1:x2]
        avg = float(np.mean(patch)) if patch.size > 0 else 0.0
        fv.append(avg)

    return fv
