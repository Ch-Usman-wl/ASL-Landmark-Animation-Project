import json
import random
from pathlib import Path

import numpy as np


ASLNOW_PATH = Path("data/aslnow")


def load_letter(letter, sample_index=0):
    """
    Load one ASLNow static fingerspelling sample.

    Returns:
        numpy array with shape (21, 2)
    """

    letter = letter.upper()

    letter_dir = ASLNOW_PATH / letter

    if not letter_dir.exists():
        raise ValueError(f"Letter folder not found: {letter}")

    files = sorted(letter_dir.glob("*.json"))

    if not files:
        raise ValueError(f"No JSON samples found for letter: {letter}")

    if sample_index >= len(files):
        raise IndexError(
            f"{letter} only has {len(files)} samples."
        )

    with open(files[sample_index], "r", encoding="utf-8") as f:
        data = json.load(f)

    points = np.array(
        [[p["x"], p["y"]] for p in data],
        dtype=np.float32
    )

    if points.shape != (21, 2):
        raise ValueError(
            f"Expected (21, 2), got {points.shape}"
        )

    return points


if __name__ == "__main__":

    letter = "A"

    points = load_letter(letter)

    print(f"Letter: {letter}")
    print("Shape:", points.shape)
    print("First landmark:", points[0])