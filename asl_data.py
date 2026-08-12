import h5py
import numpy as np


HDF5_PATH = "data/primarymath/val.hdf5"


def load_sample(sample_id):
    """
    Load one PrimaryMath sample.

    Returns:
        numpy array with shape:
        (frames, 135, 2)
    """

    with h5py.File(HDF5_PATH, "r") as f:
        data = f[str(sample_id)]["data"][:]

    # Original:
    # (frames, 2, 135)
    #
    # We want:
    # (frames, 135, 2)

    return data.transpose(0, 2, 1)


def get_hands(sequence):
    """
    Extract the two hand landmark sets.

    Returns:
        hand_a: (frames, 21, 2)
        hand_b: (frames, 21, 2)
    """

    hand_a = sequence[:, 91:112, :]
    hand_b = sequence[:, 112:133, :]

    return hand_a, hand_b