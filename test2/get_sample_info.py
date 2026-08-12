import h5py

HDF5_PATH = "data/primarymath/val.hdf5"

# Change this whenever you want to inspect another sample
SAMPLE_ID = "1177"

with h5py.File(HDF5_PATH, "r") as f:

    sample = f[SAMPLE_ID]

    label = sample["label"][()]
    video_name = sample["video_name"][()]

    # Decode bytes if necessary
    if isinstance(label, bytes):
        label = label.decode()

    if isinstance(video_name, bytes):
        video_name = video_name.decode()

    print("Sample:", SAMPLE_ID)
    print("Label:", label)
    print("Video:", video_name)