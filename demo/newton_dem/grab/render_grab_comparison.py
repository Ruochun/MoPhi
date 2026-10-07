"""Render recorded and DEME GRAB motion from a shared, configurable camera."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
import newton
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import trimesh
import warp as wp

import mophi
from mophi.utils.grab.motion import GrabPlayback, playback_times
import scene_config as config

# ── Configuration ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "output/demo_grab_contact/grab"
DEFAULT_OUTPUT = ROOT / "output/render_grab_comparison"
DEFAULT_VIEW = "handle"
DEFAULT_MODE = "compare"
PATCH_COLOR_SEED = 23
PATCH_FIGURE_SIZE = (14, 7)
PATCH_FIGURE_DPI = 180
PATCH_VIEW = (25, -60)
HANDLE_CAMERA_POSITION = (-0.3, -0.7, 0.48)
HANDLE_CAMERA_TARGET = (0.0, 0.0, 0.09)
PALM_CAMERA_POSITION = (-0.48, 0.65, 0.38)
PALM_CAMERA_TARGET = (0.0, 0.0, 0.09)
LABEL_HEIGHT = 42
LABEL_FONT_SIZE = 22
LABEL_WIDTH = 640
LABEL_BACKGROUND = (15, 20, 25)
LABEL_ORIGIN = (15, 10)
PROGRESS_INTERVAL = 30


def load_render_poses(run, summary, playback, config):
    """Read exact movie-cadence poses, or explicitly interpolate legacy diagnostics."""
    path = run / "render_poses.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            return dict(archive), "saved movie-cadence DEME poses; no pose interpolation"
    # Older runs saved diagnostics less frequently than rendering. Preserve
    # their usability, but identify this approximation in the manifest.
    with np.load(run / "trajectory.npz", allow_pickle=False) as archive:
        trajectory = dict(archive)
    start = summary["start_frame"] / float(playback.hands[0]["source_fps"])
    stop = summary["stop_frame"] / float(playback.hands[0]["source_fps"])
    duration = summary["settle_time_s"] + (stop - start) / summary["playback_speed"]
    dt = float(f'{summary["dt_s"]:.7g}')
    steps = int(np.ceil(duration / dt))
    indices = sorted(
        set(np.minimum(steps, np.rint(np.arange(0, duration, 1 / config.RENDER_FPS) / dt).astype(int))) | {steps}
    )
    times = np.minimum(np.array(indices) * summary["dt_s"], trajectory["time_s"][-1])
    initial = playback.sample(start)["object"].astype(float)
    offset = np.array([-initial[:, 0].mean(), -initial[:, 1].mean(), config.REST_GAP - initial[:, 2].min()])
    return {
        "time_s": times,
        "source_time_s": np.minimum(
            stop, start + np.maximum(0, times - summary["settle_time_s"]) * summary["playback_speed"]
        ),
        "cup_com_world": np.column_stack(
            [np.interp(times, trajectory["time_s"], trajectory["cup_com_world"][:, axis]) for axis in range(3)]
        ),
        "cup_quaternion_xyzw": Slerp(trajectory["time_s"], Rotation.from_quat(trajectory["cup_quaternion_xyzw"]))(
            times
        ).as_quat(),
        "cup_vertices_principal": (playback.object["object_vertices_local"] - trajectory["source_mesh_com"])
        @ trajectory["principal_to_source"],
        "cup_faces": playback.faces["object"],
        "world_translation": offset,
        "fps": config.RENDER_FPS,
    }, "legacy diagnostic poses interpolated linearly / quaternion SLERP; contact transients may be missed"


def recorded_poses(playback, start_frame, stop_frame, speed):
    """Use the grasp scene's source interval and settling hold without a simulation."""
    fps = float(playback.hands[0]["source_fps"])
    start, stop = start_frame / fps, stop_frame / fps
    if not playback.start_time <= start < stop <= playback.end_time:
        mophi.fatal("Source frame interval is outside the prepared recordings")
    times = playback_times(0, config.SETTLE_TIME + (stop - start) / speed, config.RENDER_FPS, 1)
    initial = playback.sample(start)["object"].astype(float)
    offset = np.array([-initial[:, 0].mean(), -initial[:, 1].mean(), config.REST_GAP - initial[:, 2].min()])
    return dict(
        time_s=times,
        source_time_s=np.minimum(stop, start + np.maximum(0, times - config.SETTLE_TIME) * speed),
        world_translation=offset,
        fps=config.RENDER_FPS,
        cup_faces=playback.faces["object"],
    )


def render_patches(archive, output):
    """Optional static view of the actual collision partition, in local coordinates."""
    with np.load(archive, allow_pickle=False) as data:
        names = [name for name in ("cup", "surface") if name + "_vertices" in data]
        fig = plt.figure(figsize=PATCH_FIGURE_SIZE)
        rng = np.random.default_rng(PATCH_COLOR_SEED)
        for column, name in enumerate(names, 1):
            vertices, faces, ids = (data[name + suffix] for suffix in ("_vertices", "_faces", "_patch_ids"))
            unique, inverse = np.unique(ids, return_inverse=True)
            colors = 0.2 + 0.75 * rng.random((len(unique), 3))
            ax = fig.add_subplot(1, len(names), column, projection="3d")
            ax.add_collection3d(
                Poly3DCollection(
                    vertices[faces], facecolors=colors[inverse], edgecolors=(0, 0, 0, 0.2), linewidths=0.15
                )
            )
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            half = np.ptp(vertices, axis=0).max() * 0.55
            ax.set(
                xlim=(center[0] - half, center[0] + half),
                ylim=(center[1] - half, center[1] + half),
                zlim=(center[2] - half, center[2] + half),
                xlabel="Local X (m)",
                ylabel="Local Y (m)",
                zlabel="Local Z (m)",
                title=f"{name}: {len(unique)} patches / {len(faces)} triangles",
            )
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(*PATCH_VIEW)
        fig.tight_layout()
        fig.savefig(output / "patches.png", dpi=PATCH_FIGURE_DPI)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("compare", "recorded", "simulation"), default=DEFAULT_MODE)
    parser.add_argument("--hand", type=Path, default=config.HAND_PATH, help="Prepared hand for recorded-only viewing")
    parser.add_argument(
        "--object", type=Path, default=config.OBJECT_PATH, help="Prepared object for recorded-only viewing"
    )
    parser.add_argument("--start-frame", type=int, default=config.GRAB_START_FRAME, help="Recorded-only mode")
    parser.add_argument("--stop-frame", type=int, default=config.GRAB_STOP_FRAME, help="Recorded-only mode")
    parser.add_argument("--playback-speed", type=float, default=config.PLAYBACK_SPEED, help="Recorded-only mode")
    parser.add_argument("--interactive", action="store_true", help="Single-panel window instead of movie output")
    parser.add_argument(
        "--patches", action="store_true", help="Also save a color-coded collision-patch PNG from the run"
    )
    parser.add_argument("--view", choices=("original", "handle", "palm"), default=DEFAULT_VIEW)
    parser.add_argument("--camera-position", nargs=3, type=float)
    parser.add_argument("--camera-target", nargs=3, type=float)
    parser.add_argument("--preview-frame", type=int, help="Save only this movie frame as PNG")
    args = parser.parse_args()
    if args.interactive and args.mode == "compare":
        mophi.fatal("Interactive viewing uses --mode recorded or --mode simulation")
    if args.mode == "recorded":
        if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
            mophi.fatal("--playback-speed must be finite and positive")
        playback = GrabPlayback([args.hand], args.object)
        poses = recorded_poses(playback, args.start_frame, args.stop_frame, args.playback_speed)
        sampling = "recorded motion only; no simulation required"
    else:
        summary = json.loads((args.run_directory / "summary.json").read_text())
        if summary.get("case") != "grab" or not summary["checks"]["completed"]:
            mophi.fatal("Comparison/replay requires a completed grasp run")
        playback = GrabPlayback([summary["source_hand"]], summary["source_object"])
        poses, sampling = load_render_poses(args.run_directory, summary, playback, config)
    cameras = {
        "original": (config.CAMERA_POSITION, config.CAMERA_TARGET),
        "handle": (HANDLE_CAMERA_POSITION, HANDLE_CAMERA_TARGET),
        "palm": (PALM_CAMERA_POSITION, PALM_CAMERA_TARGET),
    }
    position, look_at = cameras[args.view]
    camera = np.asarray(args.camera_position or position)
    target = np.asarray(args.camera_target or look_at)
    direction = target - camera
    if not np.isfinite(np.r_[camera, target]).all() or np.linalg.norm(direction) == 0:
        mophi.fatal("Camera and target must be finite and distinct")
    if args.preview_frame is not None and not 0 <= args.preview_frame < len(poses["time_s"]):
        mophi.fatal("Preview frame is outside saved motion")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.patches:
        render_patches(args.run_directory / "patches.npz", args.output_dir)
    wp.init()
    model = newton.ModelBuilder(up_axis=newton.Axis.Z).finalize()
    vis = mophi.OpenGLVisualizer(
        model, width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT, headless=not args.interactive
    )
    vis.set_camera(
        wp.vec3(*camera),
        float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))),
        float(np.degrees(np.arctan2(direction[1], direction[0]))),
    )
    floor = trimesh.creation.box(extents=config.FLOOR_VISUAL_EXTENTS)
    floor.apply_translation([0, 0, -config.FLOOR_VISUAL_EXTENTS[2] / 2])
    indices = {
        name: wp.array(faces.ravel(), dtype=wp.int32)
        for name, faces in {
            "cup": poses["cup_faces"],
            "surface": playback.faces[str(playback.hands[0]["hand_side"])],
            "floor": floor.faces,
        }.items()
    }
    buffers = {}
    writers = {}
    images = []
    frames = range(len(poses["time_s"])) if args.preview_frame is None else [args.preview_frame]
    try:
        movie_names = {
            "compare": ("recorded_reference", "deme_replay", "recorded_vs_deme"),
            "recorded": ("recorded_reference",),
            "simulation": ("deme_replay",),
        }[args.mode]
        if args.preview_frame is None and not args.interactive:
            for name in movie_names:
                writers[name] = imageio.get_writer(args.output_dir / (name + ".mp4"), fps=float(poses["fps"]))
        iteration = 0
        while iteration < len(frames) and vis.is_running():
            frame_start = time.perf_counter()
            index = frames[iteration]
            sample = playback.sample(float(poses["source_time_s"][index]))
            cups = []
            if args.mode != "simulation":
                cups.append(sample["object"] + poses["world_translation"])
            if args.mode != "recorded":
                cups.append(
                    Rotation.from_quat(poses["cup_quaternion_xyzw"][index]).apply(poses["cup_vertices_principal"])
                    + poses["cup_com_world"][index]
                )
            images = []
            for cup in cups:
                vis.begin_frame(float(poses["time_s"][index]))
                if iteration == 0 and not images:
                    mophi.log_orientation_and_scale_reference(
                        vis,
                        axis_origin=config.AXIS_ORIGIN,
                        axis_length=config.AXIS_LENGTH,
                        scale_bar_center=config.SCALE_BAR_CENTER,
                        marker_radius=config.SCALE_MARKER_RADIUS,
                    )
                geometry = {
                    "cup": cup,
                    "surface": sample[str(playback.hands[0]["hand_side"])] + poses["world_translation"],
                    "floor": floor.vertices,
                }
                for name, vertices in geometry.items():
                    if name not in buffers:
                        buffers[name] = wp.array(vertices, dtype=wp.vec3)
                    else:
                        buffers[name].assign(np.asarray(vertices, dtype=np.float32))
                    vis.log_mesh(name, buffers[name], indices[name], color=config.MESH_COLORS[name])
                vis.end_frame()
                images.append(vis.get_frame().numpy().copy())
            pair = Image.fromarray(np.concatenate(images, axis=1))
            draw = ImageDraw.Draw(pair)
            labels = {
                "compare": ("RECORDED GRAB HAND + CUP", "DEME: RECORDED HAND + DYNAMIC CUP"),
                "recorded": ("RECORDED GRAB HAND + CUP",),
                "simulation": ("DEME: RECORDED HAND + DYNAMIC CUP",),
            }[args.mode]
            for column, label in enumerate(labels):
                x = column * config.WINDOW_WIDTH
                draw.rectangle((x, 0, x + LABEL_WIDTH, LABEL_HEIGHT), fill=LABEL_BACKGROUND)
                draw.text((x + LABEL_ORIGIN[0], LABEL_ORIGIN[1]), label, fill="white", font_size=LABEL_FONT_SIZE)
            if writers:
                if args.mode != "simulation":
                    writers["recorded_reference"].append_data(images[0])
                if args.mode != "recorded":
                    writers["deme_replay"].append_data(images[-1])
                if args.mode == "compare":
                    writers["recorded_vs_deme"].append_data(np.asarray(pair))
            if iteration % PROGRESS_INTERVAL == 0:
                print(f"Rendered {iteration + 1}/{len(frames)}", flush=True)
            if not (args.interactive and vis.is_paused()):
                iteration += 1
            if args.interactive:
                time.sleep(max(0, 1 / float(poses["fps"]) - (time.perf_counter() - frame_start)))
        if not images:
            mophi.fatal("Viewer closed before rendering a frame")
        imageio.imwrite(args.output_dir / "last_frame.png", images[0])
        pair.save(args.output_dir / "comparison_last_frame.png")
    finally:
        for writer in writers.values():
            writer.close()
        vis.close()
    np.savez(
        args.output_dir / "frame_timing.npz",
        simulation_time_s=poses["time_s"],
        source_time_s=poses["source_time_s"],
        world_translation=poses["world_translation"],
    )
    manifest = {
        "source_simulation": str(args.run_directory.resolve()) if args.mode != "recorded" else None,
        "mode": args.mode,
        "camera_position": camera.tolist(),
        "camera_target": target.tolist(),
        "view": args.view,
        "preview_frame": args.preview_frame,
        "fps": float(poses["fps"]),
        "frames": iteration,
        "simulation_pose_sampling": sampling,
        "world_translation": poses["world_translation"].tolist(),
        "reference_motion": "Recorded hand/object; no dynamics. Same prescribed hand in both panels.",
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
