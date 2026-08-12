#import h5py
from asl_data import load_sample
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np


HDF5_PATH = "data/primarymath/val.hdf5"


HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),

    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (5, 9), (9, 13), (13, 17),
]


def render_sample(sample_id, output_file):

    print(f"Loading sample {sample_id}...")

    # # with h5py.File(HDF5_PATH, "r") as f:
    # #     data = f[str(sample_id)]["data"][:]

    # # (frames, 2, 135)
    # #       ↓
    # # (frames, 135, 2)
    # points = data.transpose(0, 2, 1)
    
    points = load_sample(sample_id)

    num_frames = points.shape[0]

    print(f"Frames: {num_frames}")

    # Two hands
    hand_a = points[:, 91:112, :]
    hand_b = points[:, 112:133, :]

    # -----------------------------------------------------
    # Find plotting bounds
    # Ignore missing (0,0) landmarks
    # -----------------------------------------------------

    all_hands = points[:, 91:133, :]

    valid = np.any(all_hands != 0, axis=2)

    valid_points = all_hands[valid]

    xmin = valid_points[:, 0].min()
    xmax = valid_points[:, 0].max()

    ymin = valid_points[:, 1].min()
    ymax = valid_points[:, 1].max()

    padding = 0.08

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

    ax.set_aspect("equal")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # -----------------------------------------------------
    # Create hand artists
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Update a hand
    # -----------------------------------------------------

    def update_hand(current_hand, lines, scatter):

        valid = np.any(
            current_hand != 0,
            axis=1
        )

        visible_points = current_hand[valid]

        if len(visible_points) > 0:
            scatter.set_offsets(visible_points)
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

    # -----------------------------------------------------
    # Animation update
    # -----------------------------------------------------

    def update(frame):

        update_hand(
            hand_a[frame],
            lines_a,
            scatter_a
        )

        update_hand(
            hand_b[frame],
            lines_b,
            scatter_b
        )

        ax.set_title(
            f"ASL Skeleton — Sample {sample_id} "
            f"— Frame {frame + 1}/{num_frames}"
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=50,
        blit=False
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print(f"Saving to {output_file}...")

    animation.save(
        output_file,
        #writer=PillowWriter(fps=20)
        writer=FFMpegWriter(fps=20)
    )

    plt.close(fig)

    print("Done!")


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    render_sample(
        "1027",
        #"BOOK_1027.gif"
        "BOOK_1027.mp4"
        
        # "1002",
        # "WAVE_1002.gif"
    )