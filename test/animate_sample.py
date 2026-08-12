import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HDF5_PATH = "data/primarymath/val.hdf5"
SAMPLE_ID = "1002"

with h5py.File(HDF5_PATH, "r") as f:

    data = f[SAMPLE_ID]["data"][:]

# Dataset format:
# (frames, 2, landmarks)
#
# Convert to:
# (frames, landmarks, 2)

points = data.transpose(0, 2, 1)

num_frames = points.shape[0]
num_landmarks = points.shape[1]

print("Frames:", num_frames)
print("Landmarks:", num_landmarks)


# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(
    points[:, :, 0].min() - 0.05,
    points[:, :, 0].max() + 0.05
)

ax.set_ylim(
    points[:, :, 1].max() + 0.05,
    points[:, :, 1].min() - 0.05
)

ax.set_aspect("equal")

ax.set_title("PrimaryMath — HELLO")

ax.set_xlabel("X")
ax.set_ylabel("Y")


# Initial frame
scatter = ax.scatter(
    points[0, :, 0],
    points[0, :, 1],
    s=25
)


# ---------------------------------------------------------
# Animation update
# ---------------------------------------------------------

def update(frame):

    scatter.set_offsets(
        points[frame]
    )

    ax.set_title(
        f"PrimaryMath — HELLO — Frame {frame + 1}/{num_frames}"
    )

    return scatter,


animation = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,
    blit=True
)

plt.show()