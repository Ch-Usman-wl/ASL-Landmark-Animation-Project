import h5py

HDF5_PATH = "data/primarymath/val.hdf5"

with h5py.File(HDF5_PATH, "r") as f:

    print("Top-level keys:")
    print(list(f.keys()))

    # Inspect one sample
    sample_id = "1027"

    sample = f[sample_id]

    print()
    print("Sample:", sample_id)
    print("Keys:", list(sample.keys()))

    for key in sample.keys():

        obj = sample[key]

        print(
            f"{key}:",
            type(obj),
            getattr(obj, "shape", None)
        )

        if hasattr(obj, "attrs"):
            print("  attrs:", dict(obj.attrs))