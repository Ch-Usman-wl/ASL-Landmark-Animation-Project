import h5py
import numpy as np

HDF5_PATH = "data/primarymath/val.hdf5"

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20) # pinky
]


def hand_score(points):
    """
    Measure how 'hand-like' a 21-point block is.

    We calculate the average length of the standard
    MediaPipe hand connections relative to the overall
    size of the point cloud.
    """

    # points shape: (T, 21, 2)

    scores = []

    for frame in points:

        # Ignore frames where everything is zero
        if np.max(np.abs(frame)) == 0:
            continue

        # Overall hand scale
        scale = np.max(
            np.linalg.norm(
                frame[:, None, :] - frame[None, :, :],
                axis=-1
            )
        )

        if scale == 0:
            continue

        lengths = []

        for a, b in HAND_CONNECTIONS:
            length = np.linalg.norm(frame[a] - frame[b])
            lengths.append(length)

        # Normalized average connection length
        scores.append(np.mean(lengths) / scale)

    if not scores:
        return None

    return np.mean(scores)


with h5py.File(HDF5_PATH, "r") as f:

    # Gather every possible contiguous 21-point block.
    #
    # We exclude the very beginning initially because
    # pose occupies the first part of the representation.
    candidates = []

    for start in range(0, 115):
        end = start + 21

        scores = []

        for sample_id in list(f.keys())[:200]:

            data = f[sample_id]["data"][:]

            # Convert:
            # (T, 2, 135)
            #
            # into:
            # (T, 135, 2)
            points = np.transpose(data, (0, 2, 1))

            block = points[:, start:end, :]

            score = hand_score(block)

            if score is not None:
                scores.append(score)

        if scores:
            candidates.append(
                (
                    start,
                    start + 20,
                    np.mean(scores)
                )
            )

    # Print candidates
    print("\nPotential 21-landmark blocks:\n")

    for start, end, score in candidates:
        print(
            f"{start:3d}-{end:3d} "
            f"score={score:.6f}"
        )