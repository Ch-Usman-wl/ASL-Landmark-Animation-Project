import h5py
import json

HDF5_PATH = "data/primarymath/val.hdf5"
LABELS_PATH = "data/primarymath/labels.json"


# ---------------------------------------------------------
# Load labels without assuming a particular JSON structure
# ---------------------------------------------------------

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_data = json.load(f)

print("labels.json structure:")
print(type(labels_data))

if isinstance(labels_data, dict):
    print("Top-level keys:", list(labels_data.keys()))

print()


# ---------------------------------------------------------
# HDF5 inspection
# ---------------------------------------------------------

with h5py.File(HDF5_PATH, "r") as f:

    print("=" * 70)
    print("HDF5 ROOT")
    print("=" * 70)

    print("Keys:", list(f.keys()))
    print("Number of samples:", len(f))

    # Inspect first 5 samples
    for sample_id in list(f.keys())[:5]:

        group = f[sample_id]

        print("\n" + "=" * 70)
        print("SAMPLE:", sample_id)
        print("=" * 70)

        print("Group keys:", list(group.keys()))

        # Read label
        label = group["label"][()]

        # Decode bytes if necessary
        if isinstance(label, bytes):
            label = label.decode("utf-8")

        print("Label:", label)

        # Read video name if available
        if "video_name" in group:
            video_name = group["video_name"][()]

            if isinstance(video_name, bytes):
                video_name = video_name.decode("utf-8")

            print("Video:", video_name)

        # Read landmark data
        data = group["data"][:]

        print("Data shape:", data.shape)
        print("Data dtype:", data.dtype)

        # Expected: (frames, 2, 135)
        if len(data.shape) == 3:
            frames, coordinates, landmarks = data.shape

            print("Frames:", frames)
            print("Coordinates:", coordinates)
            print("Landmarks:", landmarks)

            print("\nFirst frame shape:", data[0].shape)

            print("\nFirst 10 landmarks:")
            for i in range(min(10, landmarks)):
                x = data[0, 0, i]
                y = data[0, 1, i]

                print(
                    f"  {i:3d}: "
                    f"x={x:.6f}, "
                    f"y={y:.6f}"
                )

            print("\nLast 10 landmarks:")
            for i in range(max(0, landmarks - 10), landmarks):
                x = data[0, 0, i]
                y = data[0, 1, i]

                print(
                    f"  {i:3d}: "
                    f"x={x:.6f}, "
                    f"y={y:.6f}"
                )