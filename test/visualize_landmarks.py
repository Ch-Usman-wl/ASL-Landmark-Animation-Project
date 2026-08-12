import h5py
import matplotlib.pyplot as plt

HDF5_PATH = "data/primarymath/val.hdf5"

with h5py.File(HDF5_PATH, "r") as f:

    # Use first sample
    sample_id = list(f.keys())[0]

    data = f[sample_id]["data"][:]

    # First frame
    frame = data[0]

    # (2, 135)
    x = frame[0]
    y = frame[1]

    fig, ax = plt.subplots(figsize=(14, 10))

    # --------------------------------------------------
    # Plot groups
    # --------------------------------------------------

    groups = [
        ("Pose", 0, 33, "blue"),
        ("Hand 1", 33, 54, "red"),
        ("Hand 2", 54, 75, "green"),
        ("Face", 75, 135, "purple"),
    ]

    for name, start, end, color in groups:

        ax.scatter(
            x[start:end],
            y[start:end],
            label=f"{name} ({start}-{end-1})",
            s=35
        )

        # Label hand points
        if "Hand" in name:
            for i in range(start, end):
                ax.text(
                    x[i],
                    y[i],
                    str(i),
                    fontsize=8
                )

    ax.invert_yaxis()

    ax.set_aspect("equal")
    ax.set_title(
        f"PrimaryMath sample {sample_id} - frame 0"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.legend()

    plt.show()