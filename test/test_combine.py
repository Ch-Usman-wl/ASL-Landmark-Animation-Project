import numpy as np

from asl_data import load_sample
from transitions import interpolate_transition
from render_sequence import render_sequence


wave = load_sample("1002")
book = load_sample("1027")

print("WAVE:", wave.shape)
print("BOOK:", book.shape)


transition = interpolate_transition(
    wave,
    book,
    num_frames=15
)

print("Transition:", transition.shape)


combined = np.concatenate(
    [
        wave,
        transition,
        book
    ],
    axis=0
)

print("Combined:", combined.shape)

print("\nCoordinate ranges:")

for name, sequence in [
    ("WAVE", wave),
    ("BOOK", book),
    ("TRANSITION", transition),
    ("COMBINED", combined),
]:
    valid = np.any(sequence != 0, axis=2)
    points = sequence[valid]

    print(
        f"{name}: "
        f"X={points[:, 0].min():.3f} to {points[:, 0].max():.3f}, "
        f"Y={points[:, 1].min():.3f} to {points[:, 1].max():.3f}"
    )


render_sequence(
    combined,
    "WAVE_BOOK_transition.mp4"
)