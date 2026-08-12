from asl_data import load_sample
import numpy as np


def analyze_motion(sample_id):
    sequence = load_sample(sample_id)

    # Two hands = 42 landmarks
    hands = sequence[:, 91:133, :]

    # --------------------------------------------------
    # Landmark presence
    # --------------------------------------------------

    valid = np.any(hands != 0, axis=2)

    landmark_counts = valid.sum(axis=1)

    # --------------------------------------------------
    # Frame-to-frame movement
    # --------------------------------------------------

    movement = np.zeros(len(hands))

    for i in range(1, len(hands)):

        current = hands[i]
        previous = hands[i - 1]

        # Only compare landmarks that exist in BOTH frames
        valid_both = (
            np.any(current != 0, axis=1)
            &
            np.any(previous != 0, axis=1)
        )

        if np.any(valid_both):

            distances = np.linalg.norm(
                current[valid_both] -
                previous[valid_both],
                axis=1
            )

            movement[i] = distances.mean()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("=" * 60)
    print(f"Sample {sample_id}")
    print(f"Frames: {len(hands)}")

    print(
        f"Landmarks: "
        f"min={landmark_counts.min()}, "
        f"max={landmark_counts.max()}, "
        f"avg={landmark_counts.mean():.1f}"
    )

    print(
        f"Movement: "
        f"min={movement[1:].min():.5f}, "
        f"max={movement[1:].max():.5f}, "
        f"avg={movement[1:].mean():.5f}"
    )

    # --------------------------------------------------
    # Most active frames
    # --------------------------------------------------

    top = np.argsort(movement)[-10:][::-1]

    print("\nMost active frames:")

    for frame in top:

        print(
            f"  Frame {frame:3d}: "
            f"movement={movement[frame]:.5f}, "
            f"landmarks={landmark_counts[frame]}"
        )

    # --------------------------------------------------
    # Motion by chunks
    # --------------------------------------------------

    print("\nMotion by 5-frame chunks:")

    for start in range(0, len(hands), 5):

        end = min(start + 5, len(hands))

        chunk = movement[start:end]

        print(
            f"  {start:3d}-{end - 1:3d}: "
            f"avg={chunk.mean():.5f}"
        )


if __name__ == "__main__":

    analyze_motion("1002")
    analyze_motion("1027")
    analyze_motion("1377")
    analyze_motion("1177")
    analyze_motion("67")