# from asl_data import load_sample
# import numpy as np


# def trim_empty_padding(sequence, min_landmarks=10):
#     """
#     Remove only clearly empty frames from the beginning and end.

#     We do NOT use motion here.
#     We do NOT remove frames from the middle.
#     """

#     hands = sequence[:, 91:133, :]

#     # A landmark exists if it isn't (0, 0)
#     valid = np.any(hands != 0, axis=2)

#     # Number of detected hand landmarks per frame
#     counts = valid.sum(axis=1)

#     # A frame is considered to contain a hand if enough
#     # landmarks are present.
#     active = counts >= min_landmarks

#     indices = np.where(active)[0]

#     if len(indices) == 0:
#         return sequence

#     first = indices[0]
#     last = indices[-1]

#     return sequence[first:last + 1]


# def inspect_sample(sample_id):
#     sequence = load_sample(sample_id)

#     cleaned = trim_empty_padding(sequence)

#     print("=" * 60)
#     print(f"Sample: {sample_id}")
#     print(f"Original frames: {len(sequence)}")
#     print(f"Cleaned frames:  {len(cleaned)}")

#     print(
#         f"Removed: "
#         f"{len(sequence) - len(cleaned)} frames"
#     )

#     return sequence, cleaned


# if __name__ == "__main__":

#     inspect_sample("1002")
#     inspect_sample("1027")
#     inspect_sample("1377")
#     inspect_sample("1177")
#     inspect_sample("67")



from asl_data import load_sample
import numpy as np


def trim_empty_padding(sequence, min_landmarks=10):

    hands = sequence[:, 91:133, :]

    valid = np.any(hands != 0, axis=2)
    counts = valid.sum(axis=1)

    active = counts >= min_landmarks

    indices = np.where(active)[0]

    if len(indices) == 0:
        return sequence

    first = indices[0]
    last = indices[-1]

    return sequence[first:last + 1]


if __name__ == "__main__":

    SAMPLE_ID = "1002"

    sequence = load_sample(SAMPLE_ID)

    cleaned = trim_empty_padding(sequence)

    print("=" * 60)
    print(f"Sample: {SAMPLE_ID}")
    print(f"Original frames: {len(sequence)}")
    print(f"Cleaned frames:  {len(cleaned)}")
    print(f"Removed: {len(sequence) - len(cleaned)} frames")

    # Save cleaned landmarks
    np.save(
        f"clean_{SAMPLE_ID}.npy",
        cleaned
    )

    print(f"Saved: clean_{SAMPLE_ID}.npy")