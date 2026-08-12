import h5py
import json

HDF5_PATH = "data/primarymath/val.hdf5"
LABELS_PATH = "data/primarymath/labels.json"

# Load labels.json
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

print("labels.json type:", type(labels))

if isinstance(labels, dict):
    print("labels.json keys:", list(labels.keys()))

print("\nFirst few entries from labels.json:")
if isinstance(labels, dict):
    for i, (key, value) in enumerate(labels.items()):
        print(" ", key, ":", value)
        if i >= 10:
            break

print("\n" + "=" * 60)

# Inspect HDF5
with h5py.File(HDF5_PATH, "r") as f:

    print("Number of samples:", len(f))

    for sample_id in list(f.keys())[:10]:

        group = f[sample_id]

        label = group["label"][()]

        if isinstance(label, bytes):
            label = label.decode("utf-8")

        print(
            f"Sample {sample_id}: "
            f"label={label!r}, "
            f"shape={group['data'].shape}"
        )