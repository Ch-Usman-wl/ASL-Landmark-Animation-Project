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

fig, ax = plt.subplots(figsize=(10, 8))

# Hand A: 91-111
hand_a = frame[91:112]

ax.scatter(
    hand_a[:, 0],
    hand_a[:, 1],
    s=70,
    label="Hand A: 91-111"
)

for i in range(91, 112):
    ax.text(
        frame[i, 0] + 0.005,
        frame[i, 1] + 0.005,
        str(i),
        fontsize=10
    )

# Hand B: 112-132
hand_b = frame[112:133]

ax.scatter(
    hand_b[:, 0],
    hand_b[:, 1],
    s=70,
    label="Hand B: 112-132"
)

for i in range(112, 133):
    ax.text(
        frame[i, 0] + 0.005,
        frame[i, 1] + 0.005,
        str(i),
        fontsize=10
    )

ax.set_xlim(
    min(hand_a[:, 0].min(), hand_b[:, 0].min()) - 0.08,
    max(hand_a[:, 0].max(), hand_b[:, 0].max()) + 0.08
)

ax.set_ylim(
    max(hand_a[:, 1].max(), hand_b[:, 1].max()) + 0.08,
    min(hand_a[:, 1].min(), hand_b[:, 1].min()) - 0.08
)

ax.set_aspect("equal")

ax.set_title(
    f"PrimaryMath {SAMPLE_ID} — frame {FRAME_ID} — Hands"
)

ax.legend()

plt.show()