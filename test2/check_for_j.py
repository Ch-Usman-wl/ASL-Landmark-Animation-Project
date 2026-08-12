import h5py
import numpy as np

HDF5_PATH = "data/primarymath/val.hdf5"

with h5py.File(HDF5_PATH, "r") as f:

    matches = []

    for sample_id in f.keys():

        sample = f[sample_id]

        if "label" not in sample:
            continue

        label = sample["label"][()]

        if isinstance(label, bytes):
            label = label.decode("utf-8")

        # Print anything that is literally J
        if str(label).strip().upper() == "J":
            matches.append(sample_id)

    print("J matches:", len(matches))

    for sample_id in matches[:20]:

        sample = f[sample_id]

        print()
        print("Sample:", sample_id)
        print("Label:", sample["label"][()])

        if "video_name" in sample:
            video = sample["video_name"][()]

            if isinstance(video, bytes):
                video = video.decode("utf-8")

            print("Video:", video)

        print(
            "Shape:",
            sample["data"].shape
        )