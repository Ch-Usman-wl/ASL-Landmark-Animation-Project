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

TARGET_SIGN_FRAMES = 60

# Used only when the sentence contains no temporal signs.
DEFAULT_FINGERSPELL_SCALE = 0.8

# Additional adjustments for letters whose hand shape
# previously appeared disproportionately large.
#
# These multiply the sentence-specific scale.
LETTER_SCALE_ADJUSTMENTS = {
    "M": 0.90,
    "N": 0.90,
    "E": 0.90,
    "O": 0.90,
}

FINGERSPELL_SIZE_BOOST = 1.0
FINGERSPELL_POSITION_STABILITY = 0.70

UNSUPPORTED_LETTER_FRAMES = 12

UNSUPPORTED_LETTERS = {
    "X",
    "Y",
    "Z",
}

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
# ESTIMATE HAND SIZE
# =========================================================

def estimate_hand_size(data):
    """
    Estimate hand size.

    Accepts either:

        Single pose:
            (135, 2)

        Sequence:
            (frames, 135, 2)

    Uses wrist (0) -> middle MCP (9).
    """

    data = np.asarray(data)

    # -----------------------------------------------------
    # Single pose
    # -----------------------------------------------------

    if data.ndim == 2:

        if data.shape != (135, 2):
            raise ValueError(
                f"Expected pose shape "
                f"(135, 2), got {data.shape}"
            )

        sequence = data[np.newaxis, :, :]

    # -----------------------------------------------------
    # Sequence
    # -----------------------------------------------------

    elif data.ndim == 3:

        if data.shape[1:] != (135, 2):
            raise ValueError(
                f"Expected sequence shape "
                f"(frames, 135, 2), got {data.shape}"
            )

        sequence = data

    else:

        raise ValueError(
            f"Unexpected landmark dimensions: "
            f"{data.shape}"
        )

    # -----------------------------------------------------
    # Extract hands
    # -----------------------------------------------------

    hand_a = sequence[:, 91:112, :]
    hand_b = sequence[:, 112:133, :]

    measurements = []

    for hand in (hand_a, hand_b):

        wrist = hand[:, 0, :]
        middle_mcp = hand[:, 9, :]

        valid_wrist = np.any(
            wrist != 0,
            axis=1
        )

        valid_mcp = np.any(
            middle_mcp != 0,
            axis=1
        )

        valid = (
            valid_wrist
            & valid_mcp
        )

        if np.any(valid):

            distances = np.linalg.norm(
                middle_mcp[valid]
                - wrist[valid],
                axis=1
            )

            distances = distances[
                distances > 1e-6
            ]

            if len(distances) > 0:

                measurements.extend(
                    distances.tolist()
                )

    if not measurements:
        return None

    return float(
        np.median(measurements)
    )
    
    
# =========================================================
# SELECT TEMPORAL SAMPLE
# =========================================================

def select_temporal_sample(
    word,
    sample_ids
):

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
# SCALE A FINGERSPELLING HAND
# =========================================================

def scale_hand_pose(
    hand_pose,
    scale
):
    """
    Scale a 21-landmark hand around its wrist.

    The wrist itself remains fixed.
    """

    if hand_pose.shape != (21, 2):

        raise ValueError(
            f"Expected hand pose "
            f"(21, 2), got {hand_pose.shape}"
        )

    result = hand_pose.copy()

    wrist = hand_pose[0].copy()

    for landmark in range(21):

        if not np.any(
            hand_pose[landmark] != 0
        ):
            continue

        result[landmark] = (
            wrist
            + (
                hand_pose[landmark]
                - wrist
            ) * scale
        )

    return result



def stabilize_hand_position(
    hand_pose,
    anchor_wrist,
    stability=FINGERSPELL_POSITION_STABILITY
):
    """
    Softly pull a fingerspelling hand toward an anchor position.

    stability:
        0.0 = no stabilization
        1.0 = completely anchored

    The hand is translated as a whole, so its shape is
    not changed.
    """

    if hand_pose.shape != (21, 2):
        raise ValueError(
            f"Expected hand pose (21, 2), "
            f"got {hand_pose.shape}"
        )

    current_wrist = hand_pose[0].copy()

    # Where we would like the wrist to move.
    target_wrist = (
        (1.0 - stability) * current_wrist
        + stability * anchor_wrist
    )

    offset = target_wrist - current_wrist

    result = hand_pose.copy()

    valid = np.any(
        result != 0,
        axis=1
    )

    result[valid] += offset

    return result


# =========================================================
# EMBED FINGERSPELLING POSE
# =========================================================

def embed_fingerspell_pose(
    hand_pose
):
    """
    Convert a 21-landmark fingerspelling pose
    into the project's standard 135-landmark format.

    Current fingerspelling representatives are
    Orientation B, so they occupy landmarks 112:133.
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

    pose[112:133] = hand_pose

    return pose


# =========================================================
# FINGERSPELL WORD
# =========================================================

def fingerspell_word(
    word,
    target_hand_size
):
    """
    Create a fingerspelling sequence.

    X/Y/Z currently have no reliable landmark data.

    For those letters:
        - hold a nearby hand pose
        - display the letter as an overlay
        - remove the overlay before transition
    """

    letters = []
    overlay_ranges = []

    anchor_wrist = None

    # -----------------------------------------------------
    # First load all available normal letters.
    #
    # This also lets an unsupported first letter borrow
    # the next available hand pose.
    # -----------------------------------------------------

    prepared_letters = []

    for letter in word:

        if letter in UNSUPPORTED_LETTERS:

            prepared_letters.append(
                {
                    "letter": letter,
                    "unsupported": True,
                    "pose": None
                }
            )

            continue

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

        # -------------------------------------------------
        # Determine adaptive scale
        # -------------------------------------------------

        source_size = estimate_hand_size(
            embed_fingerspell_pose(
                hand_pose
            )
        )

        if (
            target_hand_size is not None
            and source_size is not None
        ):

            scale = (
                target_hand_size
                / source_size
            )

            print(
                f"  Source hand size: "
                f"{source_size:.4f}"
            )

            print(
                f"  Target hand size: "
                f"{target_hand_size:.4f}"
            )

            print(
                f"  Base scale: "
                f"{scale:.4f}"
            )

        else:

            scale = (
                DEFAULT_FINGERSPELL_SCALE
            )

            print(
                f"  Using default scale: "
                f"{scale:.4f}"
            )

        # -------------------------------------------------
        # Per-letter adjustment
        # -------------------------------------------------

        adjustment = (
            LETTER_SCALE_ADJUSTMENTS.get(
                letter,
                1.0
            )
        )

        final_scale = (
            scale
            * adjustment
            * FINGERSPELL_SIZE_BOOST
        )

        if adjustment != 1.0:

            print(
                f"  {letter} adjustment: "
                f"{adjustment:.2f}"
            )

        print(
            f"  Final scale: "
            f"{final_scale:.4f}"
        )

        # -------------------------------------------------
        # Scale around wrist
        # -------------------------------------------------

        hand_pose = scale_hand_pose(
            hand_pose,
            final_scale
        )

        # -------------------------------------------------
        # Soft positional stabilization
        # -------------------------------------------------

        if anchor_wrist is None:

            anchor_wrist = (
                hand_pose[0].copy()
            )

        else:

            hand_pose = (
                stabilize_hand_position(
                    hand_pose,
                    anchor_wrist
                )
            )

        prepared_letters.append(
            {
                "letter": letter,
                "unsupported": False,
                "pose": hand_pose
            }
        )

    # -----------------------------------------------------
    # Find a usable fallback pose for unsupported letters.
    #
    # Prefer:
    #   previous supported letter
    #
    # Otherwise:
    #   next supported letter
    # -----------------------------------------------------

    last_supported_pose = None

    for item in prepared_letters:

        if item["pose"] is not None:

            last_supported_pose = (
                item["pose"]
            )

        else:

            if last_supported_pose is not None:

                item["pose"] = (
                    last_supported_pose.copy()
                )

    # Handle unsupported letters at the beginning.
    next_supported_pose = None

    for item in reversed(
        prepared_letters
    ):

        if item["pose"] is not None:

            next_supported_pose = (
                item["pose"]
            )

        else:

            if next_supported_pose is not None:

                item["pose"] = (
                    next_supported_pose.copy()
                )

    # -----------------------------------------------------
    # Verify that every unsupported letter has a pose.
    # -----------------------------------------------------

    for item in prepared_letters:

        if item["pose"] is None:

            raise ValueError(
                f"Cannot display unsupported "
                f"letter {item['letter']} "
                f"because the word contains "
                f"no usable hand pose."
            )

    # -----------------------------------------------------
    # Build sequence
    # -----------------------------------------------------

    result = None

    current_frame = 0

    previous_pose = None

    for item in prepared_letters:

        letter = item["letter"]

        hand_pose = item["pose"]
        
        # -------------------------------------------------
        # Convert 21-landmark hand to full 135-landmark pose
        # BEFORE transitions are created.
        # -------------------------------------------------

        full_pose = embed_fingerspell_pose(
            hand_pose
        )

        # -------------------------------------------------
        # Unsupported letter
        # -------------------------------------------------

        if item["unsupported"]:

            # Hold the fallback hand pose.
            hold_frames = (
                UNSUPPORTED_LETTER_FRAMES
            )

            hold = np.repeat(
                full_pose[
                    np.newaxis,
                    :,
                    :
                ],
                hold_frames,
                axis=0
            )

            start_frame = current_frame

            end_frame = (
                start_frame
                + hold_frames
            )

            overlay_ranges.append(
                {
                    "start": start_frame,
                    "end": end_frame,
                    "text": letter
                }
            )

        # -------------------------------------------------
        # Normal letter
        # -------------------------------------------------

        else:

            hold_frames = 12

            hold = np.repeat(
                full_pose[
                    np.newaxis,
                    :,
                    :
                ],
                hold_frames,
                axis=0
            )

        # -------------------------------------------------
        # First letter
        # -------------------------------------------------

        if result is None:

            result = hold

            current_frame += (
                hold.shape[0]
            )

            previous_pose = hand_pose

            continue

        # -------------------------------------------------
        # Transition
        #
        # IMPORTANT:
        #
        # The overlay only covers the hold frames above.
        # It does NOT cover this transition.
        # -------------------------------------------------

        transition = make_transition(
            result,
            hold,
            frames=TRANSITION_FRAMES
        )

        result = np.concatenate(
            [
                result,
                transition,
                hold
            ],
            axis=0
        )

        current_frame = (
            result.shape[0]
        )

        previous_pose = hand_pose

    return (
        result,
        overlay_ranges
    )
    
    
    
# =========================================================
# LOAD WORD
# =========================================================

def load_word(
    word,
    vocabulary,
    target_hand_size
):

    normalized = normalize_word(
        word
    )

    # -----------------------------------------------------
    # Temporal sign
    # -----------------------------------------------------

    if normalized in vocabulary:

        sample_ids = vocabulary[
            normalized
        ]

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

        return sequence, True, []

    # -----------------------------------------------------
    # Fingerspelling fallback
    # -----------------------------------------------------

    print()
    print(
        f"  No temporal sign found for "
        f"{normalized}."
    )

    print(
        "  Falling back to fingerspelling."
    )

    sequence, overlays = fingerspell_word(
        normalized,
        target_hand_size
    )

    return sequence, False, overlays


# =========================================================
# JOIN SEQUENCES
# =========================================================

def join_sequences(
    sequences
):

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

def build_sentence(
    text
):

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

    # -----------------------------------------------------
    # First pass:
    #
    # Load all temporal signs.
    # We need them before generating fingerspelling so
    # fingerspelling can be scaled to the actual signs
    # present in THIS sentence.
    # -----------------------------------------------------

    word_sequences = []

    temporal_sizes = []

    for word in words:

        normalized = normalize_word(
            word
        )

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

        if normalized in vocabulary:

            sample_ids = vocabulary[
                normalized
            ]

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

            sequence = load_sample(
                sample_id
            )

            hand_size = estimate_hand_size(
                sequence
            )

            print(
                f"  Temporal hand size: "
                f"{hand_size}"
            )

            word_sequences.append(
                (
                    word,
                    sequence,
                    True
                )
            )

            if hand_size is not None:

                temporal_sizes.append(
                    hand_size
                )

        else:

            word_sequences.append(
                (
                    word,
                    None,
                    False
                )
            )

    # -----------------------------------------------------
    # Determine sentence target size.
    #
    # Median is used so one unusually large/small sign
    # doesn't dominate the entire sentence.
    # -----------------------------------------------------

    if temporal_sizes:

        target_hand_size = float(
            np.median(
                temporal_sizes
            )
        )

        print()
        print(
            f"Sentence temporal hand sizes: "
            f"{[round(x, 4) for x in temporal_sizes]}"
        )

        print(
            f"Target fingerspelling hand size: "
            f"{target_hand_size:.4f}"
        )

    else:

        target_hand_size = None

        print()
        print(
            "No temporal signs in sentence."
        )

        print(
            f"Using default fingerspelling "
            f"scale: {DEFAULT_FINGERSPELL_SCALE}"
        )

    # -----------------------------------------------------
    # Second pass:
    #
    # Generate missing words now that we know the target
    # size.
    # -----------------------------------------------------

    #sequences = []

    final_sequence = None
    all_overlays = []

    for word, sequence, is_temporal in word_sequences:

        if is_temporal:

            current_sequence = sequence
            local_overlays = []

        else:

            current_sequence, local_overlays = (
                fingerspell_word(
                    normalize_word(word),
                    target_hand_size
                )
            )

        # -----------------------------------------------------
        # First sequence
        # -----------------------------------------------------

        if final_sequence is None:

            final_sequence = current_sequence

            for overlay in local_overlays:

                all_overlays.append(
                    {
                        "start": overlay["start"],
                        "end": overlay["end"],
                        "text": overlay["text"]
                    }
                )

            continue

        # -----------------------------------------------------
        # Current global start
        # -----------------------------------------------------

        word_start = (
            final_sequence.shape[0]
        )

        # -----------------------------------------------------
        # Transition
        # -----------------------------------------------------

        transition = make_transition(
            final_sequence,
            current_sequence,
            frames=TRANSITION_FRAMES
        )

        # -----------------------------------------------------
        # Add current word
        # -----------------------------------------------------

        transition_start = (
            word_start
        )

        current_word_start = (
            transition_start
            + TRANSITION_FRAMES
        )

        final_sequence = np.concatenate(
            [
                final_sequence,
                transition,
                current_sequence
            ],
            axis=0
        )

        # -----------------------------------------------------
        # Convert local overlay positions to global positions
        # -----------------------------------------------------

        for overlay in local_overlays:

            all_overlays.append(
                {
                    "start": (
                        current_word_start
                        + overlay["start"]
                    ),
                    "end": (
                        current_word_start
                        + overlay["end"]
                    ),
                    "text": overlay["text"]
                }
            )

    # -----------------------------------------------------
    # Combine everything.
    # -----------------------------------------------------

    # final_sequence = join_sequences(
    #     sequences
    # )

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

    return final_sequence, all_overlays


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

    sequence, all_overlays = build_sentence(
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
        output_file=output_file,
        overlays=all_overlays
    )