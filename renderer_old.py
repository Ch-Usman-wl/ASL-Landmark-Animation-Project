import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np

# Standard 21-landmark hand connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17),
]


def render_animation(points, title="ASL Animation", output_file="test.gif"):
    """
    Render a landmark sequence.

    points shape:
        (frames, 135, 2)
    """

    num_frames = points.shape[0]

    print("Rendering frames:", num_frames)

    # ---------------------------------------------------------
    # Extract the two hands
    # ---------------------------------------------------------

    hand_a = points[:, 91:112, :]
    hand_b = points[:, 112:133, :]

    # ---------------------------------------------------------
    # Find plotting bounds
    # ---------------------------------------------------------

    all_hands = points[:, 91:133, :]

    valid = (all_hands != 0).any(axis=2)
    valid_points = all_hands[valid]

    if len(valid_points) == 0:
        raise ValueError("No valid hand landmarks found.")

    xmin = valid_points[:, 0].min()
    xmax = valid_points[:, 0].max()

    ymin = valid_points[:, 1].min()
    ymax = valid_points[:, 1].max()

    padding = 0.08

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.set_xlim(
        xmin - padding,
        xmax + padding
    )

    ax.set_ylim(
        ymax + padding,
        ymin - padding
    )

    ax.set_aspect("equal")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # ---------------------------------------------------------
    # Create artists for one hand
    # ---------------------------------------------------------

    def create_hand_artists():

        lines = []

        for connection in HAND_CONNECTIONS:

            line, = ax.plot(
                [],
                [],
                linewidth=2
            )

            lines.append(line)

        scatter = ax.scatter(
            [],
            [],
            s=45
        )

        return lines, scatter

    lines_a, scatter_a = create_hand_artists()
    lines_b, scatter_b = create_hand_artists()

    # ---------------------------------------------------------
    # Update one hand
    # ---------------------------------------------------------

    def update_hand(current_hand, lines, scatter):

        valid = (current_hand != 0).any(axis=1)

        visible_points = current_hand[valid]

        if len(visible_points) > 0:
            scatter.set_offsets(visible_points)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        for line, (start, end) in zip(
            lines,
            HAND_CONNECTIONS
        ):

            start_valid = valid[start]
            end_valid = valid[end]

            if start_valid and end_valid:

                line.set_data(
                    [
                        current_hand[start, 0],
                        current_hand[end, 0]
                    ],
                    [
                        current_hand[start, 1],
                        current_hand[end, 1]
                    ]
                )

            else:

                line.set_data([], [])

    # ---------------------------------------------------------
    # Animation update
    # ---------------------------------------------------------

    def update(frame):

        current_a = hand_a[frame]
        current_b = hand_b[frame]

        update_hand(
            current_a,
            lines_a,
            scatter_a
        )

        update_hand(
            current_b,
            lines_b,
            scatter_b
        )

        ax.set_title(
            f"{title} — Frame {frame + 1}/{num_frames}"
        )

    # ---------------------------------------------------------
    # Create animation
    # ---------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=50,
        blit=False
    )

    plt.show()
    
    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    #output_file = "test.gif"
    #output_file = "test.mp4"
    print(f"Saving to {output_file}...")

    animation.save(
        output_file,
        writer=PillowWriter(fps=20)
        #writer=FFMpegWriter(fps=20)
    )

    plt.close(fig)

    print("Done!")

    return animation