import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from urbaning.data import Sequence
from urbaning.visualizer import CameraDataVisualizer
from urbaning.visualizer.utils import object_type2color
from urbaning.data.object_label import classes

# ── User-defined parameters ──────────────────────────────────────────────────
ROOT_FOLDER = "../CVDatasets/UrbanIng-V2X"   # relative to the working directory
SEQUENCE_NAME = "20241126_0017_crossing1_00"
START_FRAME = 0   # Index of the first frame in the 5-second window
# ─────────────────────────────────────────────────────────────────────────────

FPS = 10
DURATION_FRAMES = FPS * 5  # 50 frames = 5 seconds


def render_camera(camera, object_labels: dict) -> np.ndarray:
    vis = CameraDataVisualizer()
    vis.plot_camera_data(camera)
    if object_labels:
        vis.plot_labels(
            object_labels=object_labels,
            box_color="class_label",
            text_color=(255, 255, 255),
        )
    # Convert BGR → RGB for matplotlib
    return cv2.cvtColor(vis.result(), cv2.COLOR_BGR2RGB)


def build_legend_handles():
    return [
        mpatches.Patch(
            facecolor=np.array(object_type2color[cat][::-1]) / 255.0,
            label=cat,
        )
        for cat in classes
    ]


def main():
    sequence = Sequence(ROOT_FOLDER, SEQUENCE_NAME)
    total = len(sequence)
    end_frame = min(START_FRAME + DURATION_FRAMES, total)
    n_frames = end_frame - START_FRAME

    print(f"Sequence '{SEQUENCE_NAME}': {total} frames total")
    print(f"Visualizing frames {START_FRAME}–{end_frame - 1} "
          f"({n_frames} frames, {n_frames / FPS:.1f} s)")
    print("Controls: close window = quit | SPACE = pause/resume")

    # ── Figure layout: two camera panels + legend strip ──────────────────────
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 1], hspace=0.05, wspace=0.02)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])

    for ax in (ax_left, ax_right, ax_legend):
        ax.axis("off")

    ax_legend.legend(
        handles=build_legend_handles(),
        loc="center",
        ncol=len(classes),
        frameon=False,
        fontsize=8,
    )

    plt.ion()

    # Pause/resume state controlled via keyboard
    paused = [False]

    def on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Resolve vehicle and camera keys from the first available frame
    first_frame = sequence[START_FRAME]
    vehicle_name = next(iter(first_frame.vehicles))
    cam_left_key = f"{vehicle_name}_front_left_camera"
    cam_right_key = f"{vehicle_name}_front_right_camera"
    print(f"Using vehicle: '{vehicle_name}' | "
          f"cameras: '{cam_left_key}', '{cam_right_key}'")

    im_left = im_right = None  # image handles for in-place update

    for i in range(START_FRAME, end_frame):
        if not plt.fignum_exists(fig.number):
            break

        frame = sequence[i]
        vehicle = frame.vehicles[vehicle_name]
        cam_left = vehicle.cameras[cam_left_key]
        cam_right = vehicle.cameras[cam_right_key]

        labels = {}
        if frame.labels is not None:
            labels = frame.labels.objects_in_global_coordinates(remove_av_labels=True)

        img_left = render_camera(cam_left, labels)
        img_right = render_camera(cam_right, labels)

        elapsed = (i - START_FRAME) / FPS

        if im_left is None:
            im_left = ax_left.imshow(img_left)
            im_right = ax_right.imshow(img_right)
        else:
            im_left.set_data(img_left)
            im_right.set_data(img_right)

        ax_left.set_title(f"Front Left  |  frame {i}  |  t = {elapsed:.1f} / {n_frames / FPS:.0f} s",
                          fontsize=9)
        ax_right.set_title("Front Right", fontsize=9)

        fig.canvas.draw()
        fig.canvas.flush_events()

        # Wait 100 ms per frame (10 fps); honour pause state
        plt.pause(1.0 / FPS)
        while paused[0] and plt.fignum_exists(fig.number):
            plt.pause(0.05)

    print("Done.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
