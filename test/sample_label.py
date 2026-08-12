import h5py
import json

# ============================================================
# CHANGE THIS
# ============================================================

SAMPLE_ID = 1177


# ============================================================
# FILES
# ============================================================

VAL_FILE = "data/primarymath/val.hdf5"
LABELS_FILE = "data/primarymath/labels.json"


# ============================================================
# LOOK UP SAMPLE
# ============================================================

with h5py.File(VAL_FILE, "r") as f:

    # Change "labels" only if your existing HDF5 uses
    # a different name for the label dataset.
    labels = f["labels"]

    class_id = int(labels[SAMPLE_ID])


# ============================================================
# CLASS ID -> WORD
# ============================================================

with open(LABELS_FILE, "r", encoding="utf-8") as f:
    label_data = json.load(f)

word = label_data["id_to_label"][str(class_id)]


# ============================================================
# RESULT
# ============================================================

print("=" * 40)
print(f"Sample ID : {SAMPLE_ID}")
print(f"Class ID  : {class_id}")
print(f"ASL       : {word}")
print("=" * 40)