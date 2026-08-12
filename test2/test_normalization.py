import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ASLNOW_ROOT = Path("data/aslnow")


# These are the exact samples we are currently using
SAMPLES = {
    "B": "001248b2-200f-4f5f-88ab-477ecd5964a0.json",
    "O": "084e9078-230c-43c4-a586-01f692b94367.json",
    "K": "04a661ae-b63a-4532-9a9d-874b14bb1a39.json",
    "J": "0ce5ef7b-55cf-4f3e-93b8-5796bc66aacb.json",
}


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def load_letter(letter, filename):

    path = ASLNOW_ROOT / letter / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return np.array(
        [[p["x"], p["y"]] for p in data],
        dtype=np.float32
    )


def normalize_hand(points, reference_size):

    points = points.copy()

    # -----------------------------------------------------
    # 1. Put wrist at origin
    # -----------------------------------------------------

    wrist = points[0].copy()

    points -= wrist

    # -----------------------------------------------------
    # 2. Measure current hand size
    # -----------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )

    current_size = distances.max()

    if current_size == 0:
        return points

    # -----------------------------------------------------
    # 3. Scale to reference hand size
    # -----------------------------------------------------

    points *= reference_size / current_size

    return points


def draw_hand(ax, points, title):

    for start, end in HAND_CONNECTIONS:

        ax.plot(
            [points[start, 0], points[end, 0]],
            [points[start, 1], points[end, 1]],
            linewidth=2
        )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=45
    )

    ax.set_title(title)

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0.8, -0.8)

    ax.set_aspect("equal")
    ax.grid(True)


# =========================================================
# Load raw samples
# =========================================================

raw = {}

for letter, filename in SAMPLES.items():

    raw[letter] = load_letter(
        letter,
        filename
    )


# =========================================================
# Use B as the reference scale
# =========================================================

b_distances = np.linalg.norm(
    raw["B"] - raw["B"][0],
    axis=1
)

reference_size = b_distances.max()

print("Reference B size:", reference_size)


# =========================================================
# Normalize
# =========================================================

normalized = {}

for letter, points in raw.items():

    normalized[letter] = normalize_hand(
        points,
        reference_size
    )


# =========================================================
# Print sizes
# =========================================================

print()

for letter in SAMPLES:

    distances = np.linalg.norm(
        normalized[letter],
        axis=1
    )

    print(
        f"{letter}: "
        f"size={distances.max():.6f}"
    )


# =========================================================
# Display
# =========================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4)
)

for ax, letter in zip(
    axes,
    ["B", "O", "K", "J"]
):

    draw_hand(
        ax,
        normalized[letter],
        f"{letter} — normalized"
    )


plt.tight_layout()
plt.show()