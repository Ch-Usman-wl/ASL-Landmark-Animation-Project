import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


HDF5_PATH = "data/primarymath/val.hdf5"
#SAMPLE_ID = "1027"
SAMPLE_ID = "1177"

# ---------------------------------------------------------
# Standard 21-landmark hand connections
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

with h5py.File(HDF5_PATH, "r") as f:

    data = f[SAMPLE_ID]["data"][:]


# Original:
# (frames, 2, 135)
#
# Convert to:
# (frames, 135, 2)

points = data.transpose(0, 2, 1)

num_frames = points.shape[0]

print("Frames:", num_frames)


# ---------------------------------------------------------
# Extract the two hands
# ---------------------------------------------------------

hand_a = points[:, 91:112, :]
hand_b = points[:, 112:133, :]

print("Hand A shape:", hand_a.shape)
print("Hand B shape:", hand_b.shape)


# ---------------------------------------------------------
# Find plotting bounds
# Ignore zero/missing landmarks
# ---------------------------------------------------------

all_hands = points[:, 91:133, :]

valid = np.any(all_hands != 0, axis=2)

valid_points = all_hands[valid]

xmin = valid_points[:, 0].min()
xmax = valid_points[:, 0].max()

ymin = valid_points[:, 1].min()
ymax = valid_points[:, 1].max()

padding = 0.08


# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Create artists for one hand
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Update one hand
# ---------------------------------------------------------

def update_hand(current_hand, lines, scatter):

    # A landmark is valid if it isn't (0, 0)
    valid = np.any(current_hand != 0, axis=1)

    # Only show valid landmarks
    visible_points = current_hand[valid]

    if len(visible_points) > 0:
        scatter.set_offsets(visible_points)
    else:
        scatter.set_offsets(np.empty((0, 2)))


    # Update bones
    for line, (start, end) in zip(
        lines,
        HAND_CONNECTIONS
    ):

        start_valid = valid[start]
        end_valid = valid[end]

        if start_valid and end_valid:

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

            # Hide the bone if either landmark is missing
            line.set_data([], [])


# ---------------------------------------------------------
# Animation update
# ---------------------------------------------------------

def update(frame):

    current_a = hand_a[frame]
    current_b = hand_b[frame]

    update_hand(
        current_a,
        lines_a,
        scatter_a
    )

    update_hand(
        current_b,
        lines_b,
        scatter_b
    )

    # This will now update because blit=False
    ax.set_title(
        f"ASL BOOK — Frame {frame + 1}/{num_frames}"
    )


# ---------------------------------------------------------
# Create animation
# ---------------------------------------------------------

animation = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,
    blit=False
)


plt.show()