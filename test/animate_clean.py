import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# SETTINGS
# ============================================================

FILE = "clean_1002.npy"

FPS = 15


# ============================================================
# LOAD
# ============================================================

sequence = np.load(FILE)

print(f"Frames: {len(sequence)}")
print(f"Landmarks: {sequence.shape[1]}")


# ============================================================
# HAND LANDMARKS
# ============================================================

# Our hand landmarks are 91:133
hands = sequence[:, 91:133, :]


# ============================================================
# ANIMATION
# ============================================================

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(0, 1)
ax.set_ylim(1, 0)

ax.set_aspect("equal")

ax.set_title("Cleaned WAVE")

scatter = ax.scatter([], [])


def update(frame):

    ax.clear()

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")

    points = hands[frame]

    valid = np.any(points != 0, axis=1)

    points = points[valid]

    if len(points) > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=30
        )

    ax.set_title(
        f"Cleaned WAVE — Frame {frame + 1}/{len(hands)}"
    )


animation = FuncAnimation(
    fig,
    update,
    frames=len(hands),
    interval=1000 / FPS,
    repeat=True
)

plt.show()