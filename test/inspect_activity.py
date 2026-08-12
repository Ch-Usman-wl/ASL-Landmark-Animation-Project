from asl_data import load_sample
import numpy as np


def analyze_sample(sample_id):
    sequence = load_sample(sample_id)

    # Two hands: 42 landmarks total
    hands = sequence[:, 91:133, :]

    # A landmark is present if it isn't (0, 0)
    valid = np.any(hands != 0, axis=2)

    # Number of valid landmarks in each frame
    counts = valid.sum(axis=1)

    # Consider a frame "active" if at least 10 hand landmarks exist
    active = counts >= 10

    active_indices = np.where(active)[0]

    print("=" * 60)
    print(f"Sample {sample_id}")
    print(f"Total frames: {len(counts)}")

    if len(active_indices) == 0:
        print("No active frames detected.")
        return

    first = active_indices[0]
    last = active_indices[-1]

    print(f"Active region: {first} → {last}")
    print(f"Active frames: {len(active_indices)}")

    print("\nFirst 10 frames:")
    for i in range(min(10, len(counts))):
        print(
            f"  Frame {i:3d}: "
            f"{counts[i]:2d} landmarks "
            f"{'ACTIVE' if active[i] else 'inactive'}"
        )

    print("\nLast 10 frames:")
    start = max(0, len(counts) - 10)

    for i in range(start, len(counts)):
        print(
            f"  Frame {i:3d}: "
            f"{counts[i]:2d} landmarks "
            f"{'ACTIVE' if active[i] else 'inactive'}"
        )

    print("\nActivity changes:")

    previous = active[0]

    for i in range(1, len(active)):

        if active[i] != previous:

            state = "ACTIVE" if active[i] else "INACTIVE"

            print(
                f"  Frame {i}: → {state}"
            )

            previous = active[i]


if __name__ == "__main__":

    # WAVE
    analyze_sample("1002")

    # BOOK
    analyze_sample("1027")
    
    analyze_sample("1377")
    
    analyze_sample("117")
    
    analyze_sample("1177")
    
    analyze_sample("67")