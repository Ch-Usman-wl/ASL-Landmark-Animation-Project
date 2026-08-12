import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================================================
# SETTINGS
# =========================================================

HDF5_PATH = "data/primarymath/val.hdf5"

SAMPLE_A = "1002"   # WAVE
SAMPLE_B = "1027"   # BOOK

TRANSITION_FRAMES = 8

FPS = 20

OUTPUT = "WAVE_BOOK_final.mp4"


# =========================================================
# HAND CONNECTIONS
# =========================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17),
]


# =========================================================
# LOAD SAMPLE
# =========================================================

def load_sample(sample_id):

    with h5py.File(HDF5_PATH, "r") as f:

        data = f[sample_id]["data"][:]

    # Original:
    # (frames, 2, 135)

    # Convert:
    # (frames, 135, 2)

    return data.transpose(0, 2, 1)


# =========================================================
# FIND ACTIVE REGION
# =========================================================

def find_active_region(points):

    # Hand landmarks
    hands = points[:, 91:133, :]

    # A landmark is present if it isn't (0,0)
    valid = np.any(hands != 0, axis=2)

    landmark_count = valid.sum(axis=1)

    active = landmark_count > 0

    if not np.any(active):
        return 0, len(points)

    first = np.where(active)[0][0]
    last = np.where(active)[0][-1]

    return first, last + 1


# =========================================================
# TRIM SAMPLE
# =========================================================

def trim_sample(points):

    start, end = find_active_region(points)

    return points[start:end]


# =========================================================
# TRANSITION
# =========================================================

def make_transition(a, b, frames):

    # Last valid frame of A
    start = a[-1]

    # First valid frame of B
    end = b[0]

    transition = []

    for i in range(1, frames + 1):

        t = i / (frames + 1)

        frame = start * (1 - t) + end * t

        transition.append(frame)

    return np.array(transition)


# =========================================================
# BUILD FINAL COMPOSITION
# =========================================================

wave = load_sample(SAMPLE_A)
book = load_sample(SAMPLE_B)

print("Original:")
print(f"{SAMPLE_A}: {wave.shape}")
print(f"{SAMPLE_B}: {book.shape}")


wave_clean = trim_sample(wave)
book_clean = trim_sample(book)

print()
print("After trimming:")
print(f"{SAMPLE_A}: {wave_clean.shape}")
print(f"{SAMPLE_B}: {book_clean.shape}")


transition = make_transition(
    wave_clean,
    book_clean,
    TRANSITION_FRAMES
)

print()
print(f"Transition: {transition.shape}")


combined = np.concatenate(
    [
        wave_clean,
        transition,
        book_clean
    ],
    axis=0
)

print()
print("Final composition:")
print(f"Combined: {combined.shape}")


# =========================================================
# EXTRACT HANDS
# =========================================================

hand_a = combined[:, 91:112, :]
hand_b = combined[:, 112:133, :]


# =========================================================
# FIND PLOT BOUNDS
# =========================================================

all_hands = combined[:, 91:133, :]

valid = np.any(all_hands != 0, axis=2)

valid_points = all_hands[valid]

xmin = valid_points[:, 0].min()
xmax = valid_points[:, 0].max()

ymin = valid_points[:, 1].min()
ymax = valid_points[:, 1].max()

padding = 0.08


# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(9, 8))

ax.set_xlim(
    xmin - padding,
    xmax + padding
)

ax.set_ylim(
    ymax + padding,
    ymin - padding
)

ax.set_aspect("equal")

ax.set_xlabel("X")
ax.set_ylabel("Y")


# =========================================================
# CREATE HAND ARTISTS
# =========================================================

def create_hand_artists():

    lines = []

    for connection in HAND_CONNECTIONS:

        line, = ax.plot(
            [],
            [],
            linewidth=2
        )

        lines.append(line)

    scatter = ax.scatter(
        [],
        [],
        s=45
    )

    return lines, scatter


lines_a, scatter_a = create_hand_artists()
lines_b, scatter_b = create_hand_artists()


# =========================================================
# UPDATE HAND
# =========================================================

def update_hand(current_hand, lines, scatter):

    valid = np.any(current_hand != 0, axis=1)

    visible_points = current_hand[valid]

    if len(visible_points) > 0:

        scatter.set_offsets(visible_points)

    else:

        scatter.set_offsets(
            np.empty((0, 2))
        )

    for line, (start, end) in zip(
        lines,
        HAND_CONNECTIONS
    ):

        if valid[start] and valid[end]:

            line.set_data(
                [
                    current_hand[start, 0],
                    current_hand[end, 0]
                ],
                [
                    current_hand[start, 1],
                    current_hand[end, 1]
                ]
            )

        else:

            line.set_data([], [])


# =========================================================
# ANIMATION UPDATE
# =========================================================

def update(frame):

    update_hand(
        hand_a[frame],
        lines_a,
        scatter_a
    )

    update_hand(
        hand_b[frame],
        lines_b,
        scatter_b
    )

    ax.set_title(
        f"WAVE → BOOK   Frame {frame + 1}/{len(combined)}"
    )


# =========================================================
# CREATE ANIMATION
# =========================================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(combined),
    interval=1000 / FPS,
    blit=False
)


# =========================================================
# SAVE
# =========================================================

print()
print(f"Rendering frames: {len(combined)}")
print(f"Saving: {OUTPUT}")

animation.save(
    OUTPUT,
    fps=FPS,
    writer="ffmpeg"
)

print("Done!")

plt.show()