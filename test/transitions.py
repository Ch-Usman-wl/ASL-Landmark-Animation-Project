import numpy as np


def interpolate_transition(
    sequence_a,
    sequence_b,
    num_frames=15
):
    """
    Create smooth transition frames between
    the final frame of sequence_a and the
    first frame of sequence_b.

    Returns:
        transition frames with shape:
        (num_frames, 135, 2)
    """

    start = sequence_a[-1]
    end = sequence_b[0]

    transition = []

    for i in range(1, num_frames + 1):

        t = i / (num_frames + 1)

        frame = (
            (1 - t) * start
            + t * end
        )

        transition.append(frame)

    return np.array(transition)