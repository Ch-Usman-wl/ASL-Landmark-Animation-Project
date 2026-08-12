import h5py
import numpy as np

HDF5_PATH = "data/primarymath/val.hdf5"

SAMPLE_A = "1002"   # WAVE
SAMPLE_B = "1027"   # BOOK

TRANSITION_FRAMES = 8


# ---------------------------------------------------------
# Load sample
# ---------------------------------------------------------

def load_sample(sample_id):
    with h5py.File(HDF5_PATH, "r") as f:
        data = f[sample_id]["data"][:]

    # HDF5:
    # (frames, 2, 135)
    #
    # Convert:
    # (frames, 135, 2)

    return data.transpose(0, 2, 1)


# ---------------------------------------------------------
# Determine whether a frame contains meaningful landmarks
# ---------------------------------------------------------

def landmark_count(frame):
    """
    Count landmarks that are not (0, 0).
    """
    return np.count_nonzero(
        np.any(frame != 0, axis=1)
    )


# ---------------------------------------------------------
# Trim inactive beginning/end
# ---------------------------------------------------------

def trim_sample(points, threshold=10):

    counts = np.array([
        landmark_count(frame)
        for frame in points
    ])

    active = counts >= threshold

    if not np.any(active):
        raise ValueError("No active frames found.")

    first = np.argmax(active)

    last = len(active) - 1 - np.argmax(active[::-1])

    return points[first:last + 1]


# ---------------------------------------------------------
# Get a safe frame for transition
# ---------------------------------------------------------

def last_valid_frame(points):

    for frame in reversed(points):
        if np.any(frame != 0):
            return frame.copy()

    return np.zeros_like(points[0])


def first_valid_frame(points):

    for frame in points:
        if np.any(frame != 0):
            return frame.copy()

    return np.zeros_like(points[0])


# ---------------------------------------------------------
# Create transition
# ---------------------------------------------------------

def make_transition(a, b, num_frames):

    start = last_valid_frame(a)
    end = first_valid_frame(b)

    transition = []

    for i in range(1, num_frames + 1):

        t = i / (num_frames + 1)

        frame = (1 - t) * start + t * end

        transition.append(frame)

    return np.array(transition)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

a = load_sample(SAMPLE_A)
b = load_sample(SAMPLE_B)

print("Original:")
print(f"{SAMPLE_A}: {a.shape}")
print(f"{SAMPLE_B}: {b.shape}")


# Trim in memory

a_trimmed = trim_sample(a)
b_trimmed = trim_sample(b)

print()
print("After trimming:")
print(f"{SAMPLE_A}: {a_trimmed.shape}")
print(f"{SAMPLE_B}: {b_trimmed.shape}")


# Transition

transition = make_transition(
    a_trimmed,
    b_trimmed,
    TRANSITION_FRAMES
)

print()
print("Transition:", transition.shape)


# Compose

combined = np.concatenate(
    [
        a_trimmed,
        transition,
        b_trimmed
    ],
    axis=0
)

print()
print("Final composition:")
print("Combined:", combined.shape)