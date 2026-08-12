import numpy as np


def make_transition(sequence_a, sequence_b, frames=15):

    start = sequence_a[-1].copy()
    end = sequence_b[0].copy()

    transition = []

    for i in range(1, frames + 1):

        t = i / (frames + 1)

        pose = np.zeros_like(start)

        for landmark in range(start.shape[0]):

            start_valid = np.any(start[landmark] != 0)
            end_valid = np.any(end[landmark] != 0)

            if start_valid and end_valid:
                # Normal interpolation
                pose[landmark] = (
                    (1 - t) * start[landmark]
                    + t * end[landmark]
                )

            elif start_valid:
                # Landmark disappears
                pose[landmark] = start[landmark]

            elif end_valid:
                # Landmark appears
                pose[landmark] = end[landmark]

            else:
                # Missing in both
                pose[landmark] = 0

        transition.append(pose)

    return np.array(transition)