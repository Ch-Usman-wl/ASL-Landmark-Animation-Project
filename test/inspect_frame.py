import h5py
import matplotlib.pyplot as plt

HDF5_PATH = "data/primarymath/val.hdf5"
SAMPLE_ID = "1002"
FRAME_ID = 30

with h5py.File(HDF5_PATH, "r") as f:
    data = f[SAMPLE_ID]["data"][:]

# (T, 2, 135) -> (T, 135, 2)
points = data.transpose(0, 2, 1)

frame = points[FRAME_ID]

fig, ax = plt.subplots(figsize=(12, 10))

scatter = ax.scatter(
    frame[:, 0],
    frame[:, 1],
    s=45
)

# Label every landmark
for i, (x, y) in enumerate(frame):
    ax.text(
        x + 0.005,
        y + 0.005,
        str(i),
        fontsize=8
    )

ax.set_xlim(
    frame[:, 0].min() - 0.05,
    frame[:, 0].max() + 0.05
)

ax.set_ylim(
    frame[:, 1].max() + 0.05,
    frame[:, 1].min() - 0.05
)

ax.set_aspect("equal")

ax.set_title(
    f"PrimaryMath sample {SAMPLE_ID} — "
    f"frame {FRAME_ID}"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")


def onclick(event):

    if event.inaxes != ax:
        return

    x = event.xdata
    y = event.ydata

    # Find nearest landmark
    distances = (
        (frame[:, 0] - x) ** 2
        +
        (frame[:, 1] - y) ** 2
    )

    index = distances.argmin()

    print(
        f"Landmark {index}: "
        f"x={frame[index, 0]:.6f}, "
        f"y={frame[index, 1]:.6f}"
    )


fig.canvas.mpl_connect(
    "button_press_event",
    onclick
)

plt.show()