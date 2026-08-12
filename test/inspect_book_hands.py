import h5py
import matplotlib.pyplot as plt

HDF5_PATH = "data/primarymath/val.hdf5"
SAMPLE_ID = "1027"
FRAME_ID = 55

with h5py.File(HDF5_PATH, "r") as f:
    data = f[SAMPLE_ID]["data"][:]

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
        fontsize=9
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
        fontsize=9
    )

# Limits based only on the hands
all_hands = frame[91:133]

ax.set_xlim(
    all_hands[:, 0].min() - 0.08,
    all_hands[:, 0].max() + 0.08
)

ax.set_ylim(
    all_hands[:, 1].max() + 0.08,
    all_hands[:, 1].min() - 0.08
)

ax.set_aspect("equal")

ax.set_title(
    f"BOOK — sample {SAMPLE_ID} — frame {FRAME_ID}"
)

ax.legend()

plt.show()