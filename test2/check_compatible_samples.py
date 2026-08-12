import json
from pathlib import Path

import numpy as np


ASLNOW_ROOT = Path("data/aslnow")

REFERENCE_LETTER = "B"

TARGET_LETTERS = ["O", "K"]

TOP_N = 5


# =========================================================
# Load
# =========================================================

def load_landmarks(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = np.array(
        [[p["x"], p["y"]] for p in data],
        dtype=np.float32
    )

    if points.shape != (21, 2):
        raise ValueError(
            f"Unexpected shape: {points.shape}"
        )

    return points


# =========================================================
# Normalize a hand
# =========================================================

def normalize_hand(points):

    points = points.copy()

    wrist = points[0].copy()

    # Move wrist to origin
    points -= wrist

    # Scale using maximum distance from wrist
    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = distances.max()

    if scale > 0:
        points /= scale

    return points


# =========================================================
# Compare two normalized hands
# =========================================================

def shape_distance(a, b):

    return np.mean(
        np.linalg.norm(
            a - b,
            axis=1
        )
    )


# =========================================================
# Find samples
# =========================================================

def get_samples(letter):

    folder = ASLNOW_ROOT / letter

    files = sorted(
        folder.glob("*.json")
    )

    samples = []

    for path in files:

        points = load_landmarks(path)

        normalized = normalize_hand(points)

        samples.append(
            {
                "path": path,
                "points": points,
                "normalized": normalized
            }
        )

    return samples


# =========================================================
# Main
# =========================================================

reference_samples = get_samples(
    REFERENCE_LETTER
)

if not reference_samples:
    raise ValueError(
        f"No samples found for {REFERENCE_LETTER}"
    )


# For now use the first reference sample.
reference = reference_samples[0]

print("=" * 60)
print(
    f"REFERENCE: {REFERENCE_LETTER}"
)
print(
    f"Sample: {reference['path'].name}"
)
print("=" * 60)


for letter in TARGET_LETTERS:

    samples = get_samples(letter)

    results = []

    for sample in samples:

        distance = shape_distance(
            reference["normalized"],
            sample["normalized"]
        )

        results.append(
            (
                distance,
                sample["path"].name
            )
        )

    results.sort(
        key=lambda x: x[0]
    )

    print()
    print(
        f"Closest {letter} samples:"
    )

    for distance, filename in results[:TOP_N]:

        print(
            f"  {distance:.6f}  {filename}"
        )