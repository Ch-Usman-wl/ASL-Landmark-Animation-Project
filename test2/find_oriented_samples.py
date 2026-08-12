import json
from pathlib import Path

import numpy as np


ASLNOW_ROOT = Path("data/aslnow")

TARGET_LETTERS = ["B", "O", "K", "J"]

# Use the orientation group we want consistently
TARGET_ORIENTATION = "B"


def load_landmarks(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return np.array(
        [[p["x"], p["y"]] for p in data],
        dtype=np.float32
    )


def orientation_score(points):

    wrist = points[0]

    index_mcp = points[5]

    pinky_mcp = points[17]

    a = index_mcp - wrist
    b = pinky_mcp - wrist

    cross = (
        a[0] * b[1]
        - a[1] * b[0]
    )

    return cross


def classify_orientation(points):

    score = orientation_score(points)

    if score > 0:
        return "A"

    if score < 0:
        return "B"

    return "UNKNOWN"


for letter in TARGET_LETTERS:

    folder = ASLNOW_ROOT / letter

    print()
    print("=" * 60)
    print(f"LETTER: {letter}")
    print(f"TARGET ORIENTATION: {TARGET_ORIENTATION}")
    print("=" * 60)

    matches = []

    for path in sorted(folder.glob("*.json")):

        points = load_landmarks(path)

        orientation = classify_orientation(points)

        if orientation == TARGET_ORIENTATION:

            matches.append(path.name)

    print(
        f"Found {len(matches)} compatible samples."
    )

    for filename in matches[:5]:

        print(filename)