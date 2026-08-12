import json
from pathlib import Path
import numpy as np


ASLNOW_ROOT = Path("data/aslnow")


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

    # Vectors from wrist
    a = index_mcp - wrist
    b = pinky_mcp - wrist

    # 2D cross product
    cross = (
        a[0] * b[1]
        - a[1] * b[0]
    )

    return cross


def classify(score):

    if score > 0:
        return "A"

    elif score < 0:
        return "B"

    return "UNKNOWN"


def inspect_letter(letter):

    folder = ASLNOW_ROOT / letter

    files = sorted(folder.glob("*.json"))

    counts = {
        "A": 0,
        "B": 0,
        "UNKNOWN": 0
    }

    print()
    print("=" * 50)
    print(f"LETTER: {letter}")
    print("=" * 50)

    for path in files:

        points = load_landmarks(path)

        score = orientation_score(points)
        group = classify(score)

        counts[group] += 1

    print("Total samples:", len(files))
    print("Orientation A:", counts["A"])
    print("Orientation B:", counts["B"])
    print("Unknown:", counts["UNKNOWN"])


if __name__ == "__main__":

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

        if (ASLNOW_ROOT / letter).exists():

            inspect_letter(letter)