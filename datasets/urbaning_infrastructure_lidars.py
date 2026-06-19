import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from urbaning.data import Sequence
from urbaning.data.registry import GLOBAL
from urbaning.visualizer import BEVVisualizer
from urbaning.visualizer.utils import object_type2color
from urbaning.data.object_label import classes

# ── User-defined parameters ──────────────────────────────────────────────────
ROOT_FOLDER = "../CVDatasets/UrbanIng-V2X"
SEQUENCE_NAME = "20241126_0001_crossing2_00"
START_FRAME = 100         # Index of the first frame in the 5-second window
BEV_EXTENT = 100        # Total scene extent in metres (50 m each side of centre)
BEV_SIZE = (740, 740)   # BEV render resolution (width, height) in pixels
LIDAR_COLOR_TYPE = "time_offset"    # "intensity" or "time_offset"
INCLUDE_VEHICLE_LIDARS = True       # Show individual vehicle LiDAR panels and
                                    # include vehicle point clouds in the fused BEV
# ─────────────────────────────────────────────────────────────────────────────

FPS = 10
DURATION_FRAMES = FPS * 10  # 100 frames = 10 seconds


# ── Render helpers ────────────────────────────────────────────────────────────

def render_lidar_bev(lidar, extent: float, size: tuple, lanelet_map) -> np.ndarray:
    """Render one LiDAR sensor in global BEV coordinates."""
    vis = BEVVisualizer(image_size=size, extent=extent)
    vis.set_origin(GLOBAL)
    vis.plot_point_cloud(lidar, color=LIDAR_COLOR_TYPE)
    vis.plot_lanelet_map(lanelet_map)
    return cv2.cvtColor(vis.result(), cv2.COLOR_BGR2RGB)


def render_fused_bev(sources: list, labels: dict,
                     extent: float, size: tuple, lanelet_map) -> np.ndarray:
    """Render a fused BEV from an arbitrary list of sources with GT overlay.

    sources can be a mix of Infrastructure, Vehicle, or LidarData objects.
    Each source is plotted in the same global coordinate frame.
    """
    vis = BEVVisualizer(image_size=size, extent=extent)
    vis.set_origin(GLOBAL)
    for source in sources:
        vis.plot_point_cloud(source, color=LIDAR_COLOR_TYPE)
    vis.plot_lanelet_map(lanelet_map)
    if labels:
        vis.plot_labels(
            object_labels=labels,
            box_thickness=2,
            box_color="class_label",
            text_color=(255, 255, 255),
        )
    return cv2.cvtColor(vis.result(), cv2.COLOR_BGR2RGB)


def build_legend_handles() -> list:
    return [
        mpatches.Patch(
            facecolor=np.array(object_type2color[cat][::-1]) / 255.0,
            label=cat,
        )
        for cat in classes
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sequence = Sequence(ROOT_FOLDER, SEQUENCE_NAME)
    lanelet_map = sequence.lanelet_map
    total = len(sequence)
    end_frame = min(START_FRAME + DURATION_FRAMES, total)
    n_frames = end_frame - START_FRAME

    print(f"Sequence '{SEQUENCE_NAME}': {total} frames total")
    print(f"Visualizing frames {START_FRAME}–{end_frame - 1} "
          f"({n_frames} frames, {n_frames / FPS:.1f} s)")

    # Resolve sensor names from the first frame
    first_frame = sequence[START_FRAME]

    infra_name = next(iter(first_frame.infrastructures))
    infra0 = first_frame.infrastructures[infra_name]
    lidar_names = list(infra0.lidars.keys())
    n_lidars = len(lidar_names)

    # Vehicle LiDARs: one lidar per vehicle (e.g. vehicle1_middle_lidar)
    vehicle_lidar_pairs = []   # list of (vehicle_name, lidar_name) tuples
    if INCLUDE_VEHICLE_LIDARS:
        for vname, veh in first_frame.vehicles.items():
            for lname in veh.lidars:
                vehicle_lidar_pairs.append((vname, lname))
    n_vehicles = len(vehicle_lidar_pairs)

    print(f"Infrastructure: '{infra_name}' | {n_lidars} LiDARs: {lidar_names}")
    if INCLUDE_VEHICLE_LIDARS:
        print(f"Vehicles: {[(v, l) for v, l in vehicle_lidar_pairs]}")
    print("Controls: close window = quit | SPACE = pause/resume")

    # ── Figure layout ─────────────────────────────────────────────────────────
    #
    # Without vehicles (INCLUDE_VEHICLE_LIDARS = False):
    #   Row 0: [infra_0] … [infra_N] [fused]
    #   Row 1: legend
    #
    # With vehicles (INCLUDE_VEHICLE_LIDARS = True):
    #   Row 0: [infra_0] [infra_1] … [infra_N]
    #   Row 1: [veh_0]  [veh_1]   [fused — spans remaining columns]
    #   Row 2: legend
    #
    if not INCLUDE_VEHICLE_LIDARS:
        n_cols = n_lidars + 1
        fig = plt.figure(figsize=(4 * n_cols, 5))
        gs = GridSpec(2, n_cols, figure=fig,
                      height_ratios=[10, 1], hspace=0.12, wspace=0.04)

        ax_lidars = [fig.add_subplot(gs[0, i]) for i in range(n_lidars)]
        ax_fused = fig.add_subplot(gs[0, n_lidars])
        ax_legend = fig.add_subplot(gs[1, :])
        ax_vehicles = []

    else:
        n_cols = max(n_lidars, n_vehicles + 1)  # fused needs at least 1 col
        fig = plt.figure(figsize=(4 * n_cols, 10))
        gs = GridSpec(3, n_cols, figure=fig,
                      height_ratios=[10, 10, 1], hspace=0.12, wspace=0.04)

        ax_lidars = [fig.add_subplot(gs[0, i]) for i in range(n_lidars)]
        ax_vehicles = [fig.add_subplot(gs[1, i]) for i in range(n_vehicles)]
        ax_fused = fig.add_subplot(gs[1, n_vehicles:])   # fused spans remaining cols
        ax_legend = fig.add_subplot(gs[2, :])

    # Labels and axes housekeeping
    for ax in ax_lidars + ax_vehicles + [ax_fused, ax_legend]:
        ax.axis("off")

    for ax, name in zip(ax_lidars, lidar_names):
        ax.set_title(name, fontsize=8)

    for ax, (vname, lname) in zip(ax_vehicles, vehicle_lidar_pairs):
        ax.set_title(lname, fontsize=8)

    fused_title = "Fused (infra + vehicles) + GT" if INCLUDE_VEHICLE_LIDARS \
                  else "Fused — all LiDARs + Ground Truth"
    ax_fused.set_title(fused_title, fontsize=8, fontweight="bold")

    ax_legend.legend(
        handles=build_legend_handles(),
        loc="center",
        ncol=len(classes),
        frameon=False,
        fontsize=7,
    )

    plt.ion()
    paused = [False]

    def on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]

    fig.canvas.mpl_connect("key_press_event", on_key)

    im_lidars = [None] * n_lidars
    im_vehicles = [None] * n_vehicles
    im_fused = None

    for i in range(START_FRAME, end_frame):
        if not plt.fignum_exists(fig.number):
            break

        frame = sequence[i]
        infra = frame.infrastructures[infra_name]

        labels = {}
        if frame.labels is not None:
            labels = frame.labels.objects_in_global_coordinates(remove_av_labels=True)

        # ── Infrastructure individual panels ──────────────────────────────────
        for j, lidar_name in enumerate(lidar_names):
            img = render_lidar_bev(
                infra.lidars[lidar_name], BEV_EXTENT, BEV_SIZE, lanelet_map)
            if im_lidars[j] is None:
                im_lidars[j] = ax_lidars[j].imshow(img)
            else:
                im_lidars[j].set_data(img)

        # ── Vehicle individual panels ─────────────────────────────────────────
        for j, (vname, lname) in enumerate(vehicle_lidar_pairs):
            vehicle = frame.vehicles[vname]
            img = render_lidar_bev(
                vehicle.lidars[lname], BEV_EXTENT, BEV_SIZE, lanelet_map)
            if im_vehicles[j] is None:
                im_vehicles[j] = ax_vehicles[j].imshow(img)
            else:
                im_vehicles[j].set_data(img)

        # ── Fused panel ───────────────────────────────────────────────────────
        fused_sources = [infra]
        if INCLUDE_VEHICLE_LIDARS:
            fused_sources.extend(frame.vehicles.values())
        img_fused = render_fused_bev(
            fused_sources, labels, BEV_EXTENT, BEV_SIZE, lanelet_map)
        if im_fused is None:
            im_fused = ax_fused.imshow(img_fused)
        else:
            im_fused.set_data(img_fused)

        elapsed = (i - START_FRAME) / FPS
        fig.suptitle(
            f"{infra_name}  |  Frame {i}  |  "
            f"t = {elapsed:.1f} / {n_frames / FPS:.0f} s",
            fontsize=10,
        )

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1.0 / FPS)

        while paused[0] and plt.fignum_exists(fig.number):
            plt.pause(0.01)

    print("Done.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
