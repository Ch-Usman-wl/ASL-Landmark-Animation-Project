import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ASLNOW_ROOT = Path("data/aslnow")

REFERENCE = {
    "letter": "B",
    "file": "001248b2-200f-4f5f-88ab-477ecd5964a0.json"
}

TARGETS = [
    {
        "letter": "O",
        "file": "032cb5c5-7520-4579-b515-e8a08725e5c7.json"
    },
    {
        "letter": "K",
        "file": "e8cd36b1-286a-4c7d-85c9-c7a950f60f17.json"
    }
]


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9),
    (9, 13),
    (13, 17),
]


def load_landmarks(letter, filename):

    path = ASLNOW_ROOT / letter / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return np.array(
        [[p["x"], p["y"]] for p in data],
        dtype=np.float32
    )


def normalize_hand(points):

    points = points.copy()

    wrist = points[0].copy()

    points -= wrist

    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = distances.max()

    if scale > 0:
        points /= scale

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

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(1.15, -1.15)

    ax.set_aspect("equal")

    ax.grid(True)


# =========================================================
# Load
# =========================================================

reference_points = load_landmarks(
    REFERENCE["letter"],
    REFERENCE["file"]
)

reference_points = normalize_hand(
    reference_points
)


target_points = []

for target in TARGETS:

    points = load_landmarks(
        target["letter"],
        target["file"]
    )

    points = normalize_hand(points)

    target_points.append(
        (target["letter"], points)
    )


# =========================================================
# Plot
# =========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 5)
)


draw_hand(
    axes[0],
    reference_points,
    f"B\n{REFERENCE['file'][:8]}"
)


for ax, (letter, points) in zip(
    axes[1:],
    target_points
):

    draw_hand(
        ax,
        points,
        f"{letter}"
    )


plt.tight_layout()

plt.show()