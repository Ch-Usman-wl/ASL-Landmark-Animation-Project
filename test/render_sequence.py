import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def render_sequence(sequence, output_file, fps=20):

    num_frames = sequence.shape[0]

    # -----------------------------------------------------
    # Hands
    # -----------------------------------------------------

    hand_a = sequence[:, 91:112, :]
    hand_b = sequence[:, 112:133, :]

    # -----------------------------------------------------
    # Find valid coordinates for plot bounds
    # -----------------------------------------------------

    all_hands = sequence[:, 91:133, :]

    valid = np.any(all_hands != 0, axis=2)
    valid_points = all_hands[valid]

    xmin = valid_points[:, 0].min()
    xmax = valid_points[:, 0].max()

    ymin = valid_points[:, 1].min()
    ymax = valid_points[:, 1].max()

    #padding = 0.08
    padding = 0.15

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(
        xmin - padding,
        xmax + padding
    )

    ax.set_ylim(
        ymax + padding,
        ymin - padding
    )
    
    ax.set_title(
        f"Combined ASL — Frame 1/{num_frames}"
    )

    ax.set_aspect("equal")

    # -----------------------------------------------------
    # Artists
    # -----------------------------------------------------

    def create_hand():

        lines = []

        for start, end in HAND_CONNECTIONS:

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

    lines_a, points_a = create_hand()
    lines_b, points_b = create_hand()

    # -----------------------------------------------------
    # Update hand
    # -----------------------------------------------------

    def update_hand(current, lines, scatter):

        valid = np.any(current != 0, axis=1)

        visible = current[valid]

        if len(visible):
            scatter.set_offsets(visible)
        else:
            scatter.set_offsets(
                np.empty((0, 2))
            )

        for line, (start, end) in zip(
            lines,
            HAND_CONNECTIONS
        ):

            if valid[start] and valid[end]:

                line.set_data(
                    [
                        current[start, 0],
                        current[end, 0]
                    ],
                    [
                        current[start, 1],
                        current[end, 1]
                    ]
                )

            else:

                line.set_data([], [])

    # -----------------------------------------------------
    # Animation
    # -----------------------------------------------------

    def update(frame):

        update_hand(
            hand_a[frame],
            lines_a,
            points_a
        )

        update_hand(
            hand_b[frame],
            lines_b,
            points_b
        )

        ax.set_title(
            f"Combined ASL — Frame "
            f"{frame + 1}/{num_frames}"
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False
    )

    print(f"Saving {output_file}...")

    animation.save(
        output_file,
        writer=FFMpegWriter(fps=fps)
    )

    plt.close(fig)

    print("Done!")