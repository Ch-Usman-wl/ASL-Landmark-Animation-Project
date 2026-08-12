import h5py
import json
import numpy as np

HDF5_PATH = "data/primarymath/val.hdf5"
LABELS_PATH = "data/primarymath/labels.json"

TARGET = "BOOK"

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

label_id = labels["label_to_id"][TARGET]

print(f"{TARGET} = class {label_id}")

with h5py.File(HDF5_PATH, "r") as f:

    found = 0

    for sample_id in f.keys():

        group = f[sample_id]

        label = int(group["label"][()])

        if label != label_id:
            continue

        data = group["data"][:]

        # (T, 2, 135)
        #
        # Hand A = 91-111
        # Hand B = 112-132

        hand_a = data[:, :, 91:112]
        hand_b = data[:, :, 112:133]

        # Calculate average distance from (0,0)
        # across all frames/landmarks.
        activity_a = np.sqrt(
            (hand_a ** 2).sum(axis=1)
        ).mean()

        activity_b = np.sqrt(
            (hand_b ** 2).sum(axis=1)
        ).mean()

        print(
            f"Sample {sample_id}: "
            f"frames={data.shape[0]}, "
            f"handA={activity_a:.4f}, "
            f"handB={activity_b:.4f}"
        )

        found += 1

        # Only show first 10
        if found >= 10:
            break

print(f"\nFound {found} BOOK samples.")