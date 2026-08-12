import numpy as np

from asl_data import load_sample
from transitions import make_transition
from renderer import render_animation


def trim_empty_padding(sequence, min_landmarks=10):
    """
    Remove obvious empty frames from the beginning and end.

    Does not remove frames containing partial/real landmark data.
    """

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


def compose_two(sequence_a, sequence_b, transition_frames=8):
    """
    Trim two sequences, create a transition, and combine them.
    """

    sequence_a = trim_empty_padding(sequence_a)
    sequence_b = trim_empty_padding(sequence_b)

    transition = make_transition(
        sequence_a,
        sequence_b,
        frames=transition_frames
    )

    combined = np.concatenate(
        [
            sequence_a,
            transition,
            sequence_b
        ],
        axis=0
    )

    return combined


def compose_sequence(sequences, transition_frames=8):
    """
    Compose multiple landmark sequences into one sequence.

    Each sequence is trimmed in memory before being joined.
    """

    if not sequences:
        raise ValueError("No sequences provided.")

    cleaned = [
        trim_empty_padding(sequence)
        for sequence in sequences
    ]

    result = cleaned[0]

    for next_sequence in cleaned[1:]:

        transition = make_transition(
            result,
            next_sequence,
            frames=transition_frames
        )

        result = np.concatenate(
            [
                result,
                transition,
                next_sequence
            ],
            axis=0
        )

    return result


if __name__ == "__main__":

    wave = load_sample("1002")
    book = load_sample("1027")

    combined = compose_sequence(
        [wave, book],
        transition_frames=8
    )

    print("Combined:", combined.shape)

    render_animation(
        combined,
        title="WAVE → BOOK"
    )