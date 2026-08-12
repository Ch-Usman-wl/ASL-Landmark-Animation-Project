import json
import os
import sys

import numpy as np

from asl_data import load_sample
from transitions import make_transition
from renderer import render_animation

from test_fingerspell import (
    load_letter,
    find_representative_sample
)


# =========================================================
# CONFIGURATION
# =========================================================

VOCAB_PATH = "word_samples.json"

OUTPUT_DIR = "output"

TRANSITION_FRAMES = 8

# Preferred duration for a single temporal sign.
TARGET_SIGN_FRAMES = 60


# =========================================================
# LOAD VOCABULARY
# =========================================================

def load_vocabulary():

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# NORMALIZE WORD
# =========================================================

def normalize_word(word):

    return word.strip().upper()


# =========================================================
# SELECT TEMPORAL SAMPLE
# =========================================================

def select_temporal_sample(
    word,
    sample_ids
):
    """
    Select the temporal sample whose frame count
    is closest to TARGET_SIGN_FRAMES.

    This helps avoid samples that contain repeated
    or unusually long performances.
    """

    candidates = []

    print(
        f"  Checking {len(sample_ids)} "
        f"samples for {word}..."
    )

    for sample_id in sample_ids:

        try:

            sequence = load_sample(
                sample_id
            )

            frame_count = sequence.shape[0]

            distance = abs(
                frame_count
                - TARGET_SIGN_FRAMES
            )

            candidates.append(
                (
                    distance,
                    frame_count,
                    sample_id
                )
            )

        except Exception as e:

            print(
                f"  Skipping sample "
                f"{sample_id}: {e}"
            )

    if not candidates:

        raise ValueError(
            f"No usable samples found "
            f"for {word}."
        )

    # Closest to 60 frames wins.
    candidates.sort(
        key=lambda x: x[0]
    )

    distance, frame_count, sample_id = (
        candidates[0]
    )

    print(
        f"  Selected sample: {sample_id}"
    )

    print(
        f"  Frames: {frame_count}"
    )

    print(
        f"  Distance from target: "
        f"{distance}"
    )

    return sample_id


# =========================================================
# FINGERSPELL WORD
# =========================================================

def embed_fingerspell_pose(hand_pose):
    """
    Convert a 21-landmark fingerspelling hand pose into
    the standard 135-landmark project format.

    The fingerspelling representatives currently selected
    are Orientation B, so the hand is stored in 112:133.
    """

    if hand_pose.shape != (21, 2):
        raise ValueError(
            f"Expected fingerspelling pose "
            f"(21, 2), got {hand_pose.shape}"
        )

    pose = np.zeros(
        (135, 2),
        dtype=hand_pose.dtype
    )

    # Orientation B / second hand slot.
    pose[112:133] = hand_pose

    return pose


def fingerspell_word(word):

    letters = []

    for letter in word:

        print(
            f"  Fingerspelling: {letter}"
        )

        filename = find_representative_sample(
            letter
        )

        hand_pose = load_letter(
            letter,
            filename
        )

        # Convert:
        #
        #     (21, 2)
        #
        # into:
        #
        #     (135, 2)
        #
        pose = embed_fingerspell_pose(
            hand_pose
        )

        letters.append(
            pose
        )

    if not letters:
        raise ValueError(
            f"Cannot fingerspell empty word: {word}"
        )

    frames_per_letter = 12

    result = np.repeat(
        letters[0][np.newaxis, :, :],
        frames_per_letter,
        axis=0
    )

    for pose in letters[1:]:

        next_part = np.repeat(
            pose[np.newaxis, :, :],
            frames_per_letter,
            axis=0
        )

        transition = make_transition(
            result,
            next_part,
            frames=TRANSITION_FRAMES
        )

        result = np.concatenate(
            [
                result,
                transition,
                next_part
            ],
            axis=0
        )

    return result
# =========================================================
# LOAD WORD
# =========================================================

def load_word(
    word,
    vocabulary
):

    normalized = normalize_word(
        word
    )

    # -----------------------------------------------------
    # Temporal sign exists
    # -----------------------------------------------------

    if normalized in vocabulary:

        sample_ids = vocabulary[
            normalized
        ]

        # Compatibility with an older word_samples.json
        # that may contain only one sample ID.
        if isinstance(
            sample_ids,
            str
        ):
            sample_ids = [
                sample_ids
            ]

        sample_id = select_temporal_sample(
            normalized,
            sample_ids
        )

        print()
        print(
            f"  Found sign: "
            f"{normalized}"
        )

        sequence = load_sample(
            sample_id
        )

        print(
            f"  Sign sequence: "
            f"{sequence.shape}"
        )

        return sequence

    # -----------------------------------------------------
    # No temporal sign
    # -----------------------------------------------------

    print()
    print(
        f"  No temporal sign found for "
        f"{normalized}."
    )

    print(
        "  Falling back to fingerspelling."
    )

    return fingerspell_word(
        normalized
    )


# =========================================================
# JOIN SEQUENCES
# =========================================================

def join_sequences(sequences):

    if not sequences:

        raise ValueError(
            "No sequences to join."
        )

    result = sequences[0]

    for next_sequence in sequences[1:]:

        transition = make_transition(
            result,
            next_sequence,
            frames=TRANSITION_FRAMES
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


# =========================================================
# BUILD SENTENCE
# =========================================================

def build_sentence(text):

    vocabulary = load_vocabulary()

    words = text.strip().split()

    if not words:

        raise ValueError(
            "No text provided."
        )

    print(
        "=" * 60
    )

    print(
        f"INPUT: {text}"
    )

    print(
        "=" * 60
    )

    sequences = []

    for word in words:

        print()
        print(
            "-" * 60
        )

        print(
            f"WORD: {word}"
        )

        print(
            "-" * 60
        )

        sequence = load_word(
            word,
            vocabulary
        )

        sequences.append(
            sequence
        )

    final_sequence = join_sequences(
        sequences
    )

    print()
    print(
        "=" * 60
    )

    print(
        f"FINAL SEQUENCE: "
        f"{final_sequence.shape}"
    )

    print(
        "=" * 60
    )

    return final_sequence


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        text = " ".join(
            sys.argv[1:]
        )

    else:

        text = input(
            "Enter text: "
        ).strip()

    if not text:

        raise ValueError(
            "No text entered."
        )

    sequence = build_sentence(
        text
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    safe_name = "_".join(
        text.upper().split()
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}.gif"
    )

    print()
    print(
        f"Output: {output_file}"
    )

    render_animation(
        sequence,
        title=f"ASL — {text.upper()}",
        output_file=output_file
    )