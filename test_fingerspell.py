import json
import os
import sys

import numpy as np

from renderer import render_animation
from transitions import make_transition


# =========================================================
# SETTINGS
# =========================================================

#WORD = "TEACHER"


WORD = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "HELLO"
)

OUTPUT_DIR = "output"

FRAMES_PER_LETTER = 12
TRANSITION_FRAMES = 8

# Letters that look better slightly smaller.
SMALL_HAND_LETTERS = {
    "M",
    "N",
    "E",
    "O",
}

FINGER_SCALE = 0.75

# Repeated-letter separation
REPEAT_RELEASE_FRAMES = 5
REPEAT_OPEN_AMOUNT = 0.10

ASLNOW_ROOT = "data/aslnow"

# Which hand slot in our 135-landmark representation
# should contain the fingerspelling hand.
#
# A = landmarks 91:112
# B = landmarks 112:133
FINGERSPELL_HAND = "A"

# Geometric orientation group.
#
# This is NOT necessarily anatomical left/right.
TARGET_ORIENTATION = "B"

# Automatically find samples for letters that don't
# already have a manually selected sample.
AUTO_SELECT_SAMPLES = True


# =========================================================
# MANUALLY SELECTED SAMPLES
# =========================================================
#
# Keep known-good samples here.
#
# These will NOT be replaced by automatic selection.
#
# B = known good B
# O = manually selected because the previous O
#     had a bad thumb
# K = known compatible K
# =========================================================

SELECTED_SAMPLES = {
    "B": "001248b2-200f-4f5f-88ab-477ecd5964a0.json",
    "O": "084e9078-230c-43c4-a586-01f692b94367.json",
    "K": "04a661ae-b63a-4532-9a9d-874b14bb1a39.json",
}


# =========================================================
# Determine geometric orientation
# =========================================================

def get_orientation(points):
    """
    Determine the geometric orientation group of a
    21-landmark hand.

    This is NOT claiming anatomical left/right.

    Returns:
        "A"
        "B"
        "UNKNOWN"
    """

    wrist = points[0]

    index_mcp = points[5]

    pinky_mcp = points[17]

    a = index_mcp - wrist
    b = pinky_mcp - wrist

    score = (
        a[0] * b[1]
        - a[1] * b[0]
    )

    if score > 0:
        return "A"

    if score < 0:
        return "B"

    return "UNKNOWN"


# =========================================================
# Find compatible ASLNow sample
# =========================================================

def find_compatible_sample(letter):
    """
    Find the first valid ASLNow sample with the desired
    geometric orientation.

    Returns:
        filename
    """

    folder = os.path.join(
        ASLNOW_ROOT,
        letter
    )

    if not os.path.isdir(folder):
        raise ValueError(
            f"No ASLNow folder found for letter "
            f"'{letter}': {folder}"
        )

    files = sorted(
        f
        for f in os.listdir(folder)
        if f.endswith(".json")
    )

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            points = np.array(
                [
                    [p["x"], p["y"]]
                    for p in data
                ],
                dtype=np.float32
            )

        except Exception:
            continue

        if points.shape != (21, 2):
            continue

        orientation = get_orientation(
            points
        )

        if orientation == TARGET_ORIENTATION:

            return filename

    raise ValueError(
        f"No compatible sample found for "
        f"letter '{letter}' with orientation "
        f"{TARGET_ORIENTATION}"
    )
    

# =========================================================
# Find representative ASLNow sample
# =========================================================
def find_representative_sample(letter):
    """
    Find the most representative sample for a letter.

    Only samples with TARGET_ORIENTATION are considered.

    The selected sample is the geometric medoid:
    the sample with the smallest average distance
    to all other compatible samples.

    Returns:
        filename
    """

    folder = os.path.join(
        ASLNOW_ROOT,
        letter
    )

    if not os.path.isdir(folder):
        raise ValueError(
            f"No ASLNow folder found for "
            f"letter '{letter}'"
        )

    files = sorted(
        f
        for f in os.listdir(folder)
        if f.endswith(".json")
    )

    candidates = []

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            points = np.array(
                [
                    [p["x"], p["y"]]
                    for p in data
                ],
                dtype=np.float32
            )

        except Exception:
            continue

        if points.shape != (21, 2):
            continue

        if get_orientation(points) != TARGET_ORIENTATION:
            continue

        # ---------------------------------------------
        # Normalize around wrist
        # ---------------------------------------------

        wrist = points[0].copy()

        normalized = (
            points - wrist
        )

        size = np.linalg.norm(
            normalized,
            axis=1
        ).max()

        if size == 0:
            continue

        normalized /= size

        candidates.append(
            (filename, normalized)
        )

    if not candidates:

        raise ValueError(
            f"No compatible samples found "
            f"for {letter}"
        )

    # Only one candidate
    if len(candidates) == 1:

        return candidates[0][0]

    # ---------------------------------------------
    # Compare every candidate to every other
    # ---------------------------------------------

    best_filename = None
    best_score = float("inf")

    for filename_a, points_a in candidates:

        distances = []

        for filename_b, points_b in candidates:

            if filename_a == filename_b:
                continue

            distance = np.mean(
                np.linalg.norm(
                    points_a - points_b,
                    axis=1
                )
            )

            distances.append(distance)

        score = np.mean(distances)

        if score < best_score:

            best_score = score
            best_filename = filename_a

    print(
        f"Representative {letter}: "
        f"{best_filename}"
    )

    print(
        f"  Candidates: {len(candidates)}"
    )

    print(
        f"  Average distance: "
        f"{best_score:.6f}"
    )

    return best_filename


# =========================================================
# Automatically fill missing samples
# =========================================================

def prepare_samples(letters):
    """
    Make sure every letter in the word has a sample.

    Manually selected samples are preserved.

    Missing letters are automatically assigned the
    first compatible ASLNow sample.
    """

    for letter in letters:

        if letter in SELECTED_SAMPLES:
            continue

        if not AUTO_SELECT_SAMPLES:

            raise ValueError(
                f"No selected sample for "
                f"letter '{letter}'."
            )

        # filename = find_compatible_sample(
        #     letter
        # )
        filename = find_representative_sample(
            letter
        )

        SELECTED_SAMPLES[letter] = filename

        print(
            f"Auto-selected {letter}: "
            f"{filename}"
        )


# =========================================================
# Scale hand around wrist
# =========================================================

def scale_hand_from_wrist(
    points,
    scale
):
    """
    Scale the hand around the wrist.

    The wrist landmark stays fixed.

    scale = 1.00 -> unchanged
    scale = 0.75 -> 25% smaller
    scale = 0.50 -> half size
    """

    points = points.copy()

    wrist = points[0].copy()

    for i in range(1, 21):

        points[i] = (
            wrist
            + (points[i] - wrist) * scale
        )

    return points


# =========================================================
# Load one exact ASLNow sample
# =========================================================

def load_letter(
    letter,
    filename
):
    """
    Load one exact ASLNow JSON sample.

    Returns:
        (21, 2)
    """

    path = os.path.join(
        ASLNOW_ROOT,
        letter,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Could not find sample:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    points = np.array(
        [
            [p["x"], p["y"]]
            for p in data
        ],
        dtype=np.float32
    )

    if points.shape != (21, 2):

        raise ValueError(
            f"Unexpected landmark shape for "
            f"{letter}: {points.shape}"
        )

    orientation = get_orientation(
        points
    )

    print(
        f"{letter}: {filename}"
    )

    print(
        f"  Orientation: {orientation}"
    )

    if orientation != TARGET_ORIENTATION:

        print(
            f"  WARNING: expected orientation "
            f"{TARGET_ORIENTATION}"
        )

    return points


# =========================================================
# Normalize hand to reference
# =========================================================

def normalize_to_reference(
    points,
    reference_points
):
    """
    Normalize one 21-landmark hand so that:

    1. Wrist is at the same position as reference.
    2. Overall hand size matches reference.
    3. Finger proportions are preserved.
    """

    points = points.copy()

    reference_points = (
        reference_points.copy()
    )

    # -----------------------------------------------------
    # Source/reference wrist
    # -----------------------------------------------------

    source_wrist = points[0].copy()

    reference_wrist = (
        reference_points[0].copy()
    )

    # -----------------------------------------------------
    # Move source wrist to origin
    # -----------------------------------------------------

    points -= source_wrist

    # -----------------------------------------------------
    # Source size
    # -----------------------------------------------------

    source_distances = np.linalg.norm(
        points,
        axis=1
    )

    source_size = (
        source_distances.max()
    )

    # -----------------------------------------------------
    # Reference size
    # -----------------------------------------------------

    reference_relative = (
        reference_points
        - reference_wrist
    )

    reference_distances = np.linalg.norm(
        reference_relative,
        axis=1
    )

    reference_size = (
        reference_distances.max()
    )

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    if source_size == 0:

        return reference_points.copy()

    if reference_size == 0:

        return points + reference_wrist

    # -----------------------------------------------------
    # Match overall size
    # -----------------------------------------------------

    scale = (
        reference_size
        / source_size
    )

    points *= scale

    # -----------------------------------------------------
    # Put wrist at reference wrist
    # -----------------------------------------------------

    points += reference_wrist

    return points


# =========================================================
# Create 135-landmark pose
# =========================================================

def make_letter_pose(
    letter,
    reference_points
):
    """
    Load an ASLNow letter, normalize it, optionally
    scale selected letters, and put it into the
    135-landmark representation.

    Returns:
        (135, 2)
    """

    filename = (
        SELECTED_SAMPLES[letter]
    )

    landmarks = load_letter(
        letter,
        filename
    )

    # -----------------------------------------------------
    # Normalize position and size
    # -----------------------------------------------------

    landmarks = normalize_to_reference(
        landmarks,
        reference_points
    )

    # -----------------------------------------------------
    # Make M/N/E/O slightly smaller
    # -----------------------------------------------------

    if (
        letter.upper()
        in SMALL_HAND_LETTERS
    ):

        landmarks = scale_hand_from_wrist(
            landmarks,
            FINGER_SCALE
        )

    # -----------------------------------------------------
    # Empty 135-landmark pose
    # -----------------------------------------------------

    pose = np.zeros(
        (135, 2),
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Insert fingerspelling hand
    # -----------------------------------------------------

    if FINGERSPELL_HAND == "A":

        pose[91:112] = landmarks

    elif FINGERSPELL_HAND == "B":

        pose[112:133] = landmarks

    else:

        raise ValueError(
            "FINGERSPELL_HAND must be "
            "'A' or 'B'"
        )

    return pose


# =========================================================
# Hold a pose
# =========================================================

def hold_pose(
    pose,
    frames
):
    """
    Repeat a pose for a number of frames.
    """

    return np.repeat(
        pose[np.newaxis, :, :],
        frames,
        axis=0
    )


# =========================================================
# Repeated-letter release
# =========================================================

def make_repeat_release(
    pose,
    frames=None
):
    """
    Create a small opening motion between
    two identical letters.

    The wrist remains fixed.
    """

    if frames is None:

        frames = (
            REPEAT_RELEASE_FRAMES
        )

    # -----------------------------------------------------
    # Select fingerspelling hand slot
    # -----------------------------------------------------

    if FINGERSPELL_HAND == "A":

        start = 91
        end = 112

    elif FINGERSPELL_HAND == "B":

        start = 112
        end = 133

    else:

        raise ValueError(
            "FINGERSPELL_HAND must be "
            "'A' or 'B'"
        )

    hand = pose[start:end]

    wrist = hand[0]

    release_frames = []

    # -----------------------------------------------------
    # Finger landmarks
    # -----------------------------------------------------

    finger_landmarks = [

        # Thumb
        2, 3, 4,

        # Index
        6, 7, 8,

        # Middle
        10, 11, 12,

        # Ring
        14, 15, 16,

        # Pinky
        18, 19, 20,
    ]

    # -----------------------------------------------------
    # Generate release frames
    # -----------------------------------------------------

    for i in range(
        1,
        frames + 1
    ):

        t = (
            i
            / (frames + 1)
        )

        new_pose = pose.copy()

        new_hand = hand.copy()

        for index in finger_landmarks:

            direction = (
                hand[index]
                - wrist
            )

            new_hand[index] = (
                hand[index]
                + direction
                * REPEAT_OPEN_AMOUNT
                * t
            )

        new_pose[start:end] = (
            new_hand
        )

        release_frames.append(
            new_pose
        )

    return np.array(
        release_frames,
        dtype=np.float32
    )



# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    letters = WORD.upper()

    # -----------------------------------------------------
    # Check word
    # -----------------------------------------------------

    if not letters:

        raise ValueError(
            "WORD is empty."
        )

    print()
    print("=" * 60)
    print(
        f"Preparing word: {letters}"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # Automatically find missing samples
    #
    # Existing manually selected samples are kept.
    # -----------------------------------------------------

    prepare_samples(
        letters
    )

    # -----------------------------------------------------
    # Load B as reference
    #
    # B remains our reference hand for:
    # - wrist position
    # - overall scale
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("Loading reference B")
    print("=" * 60)

    reference_b = load_letter(
        "B",
        SELECTED_SAMPLES["B"]
    )

    # -----------------------------------------------------
    # Create each unique letter pose
    # -----------------------------------------------------

    poses = {}

    for letter in letters:

        if letter in poses:
            continue

        print()
        print("=" * 60)
        print(
            f"Loading letter: {letter}"
        )
        print("=" * 60)

        poses[letter] = make_letter_pose(
            letter,
            reference_b
        )

        print(
            f"Pose shape: "
            f"{poses[letter].shape}"
        )

    # -----------------------------------------------------
    # Build final animation
    # -----------------------------------------------------

    final_parts = []

    for i, letter in enumerate(letters):

        current_pose = poses[
            letter
        ]

        # -------------------------------------------------
        # Hold current letter
        # -------------------------------------------------

        final_parts.append(
            hold_pose(
                current_pose,
                FRAMES_PER_LETTER
            )
        )
        

        # -------------------------------------------------
        # Last letter
        # -------------------------------------------------

        if i + 1 >= len(letters):

            break

        next_letter = letters[
            i + 1
        ]

        # =================================================
        # REPEATED LETTER
        # =================================================

        if letter == next_letter:

            print()
            print(
                f"Repeated letter: "
                f"{letter}{next_letter}"
            )

            # -------------------------------------------------
            # Slight opening/release
            # -------------------------------------------------

            release = (
                make_repeat_release(
                    current_pose,
                    REPEAT_RELEASE_FRAMES
                )
            )

            final_parts.append(
                release
            )

            # -------------------------------------------------
            # Transition back into same letter
            # -------------------------------------------------

            next_pose = poses[
                next_letter
            ]

            transition = make_transition(
                release,
                hold_pose(
                    next_pose,
                    1
                ),
                frames=TRANSITION_FRAMES
            )

            final_parts.append(
                transition
            )

        # =================================================
        # DIFFERENT LETTER
        # =================================================

        else:

            transition = make_transition(
                current_pose[
                    np.newaxis,
                    :,
                    :
                ],
                poses[next_letter][
                    np.newaxis,
                    :,
                    :
                ],
                frames=TRANSITION_FRAMES
            )

            final_parts.append(
                transition
            )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    result = np.concatenate(
        final_parts,
        axis=0
    )

    # -----------------------------------------------------
    # Information
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"WORD: {WORD.upper()}"
    )
    print(
        f"Letters: {len(letters)}"
    )
    print(
        f"Frames per letter: "
        f"{FRAMES_PER_LETTER}"
    )
    print(
        f"Transition frames: "
        f"{TRANSITION_FRAMES}"
    )
    print(
        f"Repeat release frames: "
        f"{REPEAT_RELEASE_FRAMES}"
    )
    print(
        f"Final sequence: "
        f"{result.shape}"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # Render
    # -----------------------------------------------------


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{WORD.upper()}.gif"
    )
    
    render_animation(
        result,
        title=(
            f"ASL Fingerspelling — "
            f"{WORD.upper()}"
        ),
        output_file=output_file
    )