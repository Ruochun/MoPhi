"""Play prepared GRAB hand/object recordings through Newton's mesh renderer."""

import argparse
from pathlib import Path
import time

import imageio.v2 as imageio
import newton
import numpy as np
import warp as wp

import mophi
from mophi.utils.grab_motion import GrabPlayback, playback_times

# ── Configuration ────────────────────────────────────────────────────────────
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
HAND_PATHS = (
    REPOSITORY_ROOT / "data/grab/s2/mug_drink_2_left.npz",
    REPOSITORY_ROOT / "data/grab/s2/mug_drink_2_right.npz",
)
OBJECT_PATH = REPOSITORY_ROOT / "data/grab/s2/mug_drink_2_object.npz"
START_TIME_S = None  # None uses the beginning of the shared source interval.
END_TIME_S = None  # None plays through the last shared recorded pose.
RENDER_FPS = 60.0
PLAYBACK_SPEED = 1.0
REALTIME_PLAYBACK = True
MAX_FRAMES = None  # Optional finite smoke-test limit, independent of source time.
USE_OMNIVERSE_VISUALIZATION = False
HEADLESS = False
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_POSITION = (1.15, -2.25, 2.15)
CAMERA_TARGET = (-0.35, 0.25, 1.2)
AXIS_ORIGIN = (-1.0, -0.3, 0.7)
AXIS_LENGTH = 0.2
SCALE_BAR_CENTER = (-0.25, -0.35, 0.6)
SCALE_MARKER_RADIUS = 0.012
MESH_COLORS = {"left": (0.20, 0.53, 0.74), "right": (0.91, 0.55, 0.19), "object": (0.65, 0.70, 0.75)}
SAVE_MOVIE = False
SAVE_SNAPSHOT = True
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output/demo_grab_playback"
MOVIE_FILENAME = "grab_playback.mp4"
SNAPSHOT_FILENAME = "last_frame.png"
USD_FILENAME = "grab_playback.usdc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", nargs="+", type=Path, default=HAND_PATHS)
    parser.add_argument("--object", type=Path, default=OBJECT_PATH)
    parser.add_argument("--start-time", type=float, default=START_TIME_S, help="Absolute source time in seconds")
    parser.add_argument("--end-time", type=float, default=END_TIME_S, help="Inclusive absolute source time")
    parser.add_argument("--fps", type=float, default=RENDER_FPS)
    parser.add_argument("--speed", type=float, default=PLAYBACK_SPEED)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--headless", action="store_true", default=HEADLESS)
    parser.add_argument("--save-movie", action="store_true", default=SAVE_MOVIE)
    parser.add_argument("--usd", action="store_true", default=USE_OMNIVERSE_VISUALIZATION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    for path in [*args.hands, args.object]:
        if not path.is_file():
            mophi.fatal(
                f"Missing prepared GRAB asset: {path}. See docs/how-to/grab-playback.md for conversion commands."
            )
    if args.max_frames is not None and args.max_frames < 1:
        mophi.fatal("--max-frames must be positive")
    if args.usd and args.save_movie:
        mophi.fatal("MP4 capture requires the OpenGL backend; omit --usd when using --save-movie")
    playback = GrabPlayback(args.hands, args.object)
    start = playback.start_time if args.start_time is None else args.start_time
    stop = playback.end_time if args.end_time is None else args.end_time
    if not playback.start_time <= start <= stop <= playback.end_time:
        mophi.fatal(f"Requested times must lie within {playback.start_time:.6f}..{playback.end_time:.6f} s")
    if not np.isfinite([args.fps, args.speed]).all() or args.fps <= 0 or args.speed <= 0:
        mophi.fatal("--fps and --speed must be finite and positive")
    times = playback_times(start, stop, args.fps, args.speed)
    if args.max_frames is not None:
        times = times[: args.max_frames]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    mophi.check_newton_warp_versions(mophi.REQUIRED_NEWTON_VERSION, mophi.REQUIRED_WARP_VERSION)
    # Geometry is prescribed; no physics solver or simulation timestep is needed.
    model = newton.ModelBuilder(up_axis=newton.Axis.Z).finalize()
    if args.usd:
        vis = mophi.OmniverseVisualizer(str(args.output_dir / USD_FILENAME), fps=args.fps)
        if not vis.pxr_available:
            mophi.fatal("USD export requires OpenUSD: python -m pip install usd-core")
    else:
        vis = mophi.OpenGLVisualizer(model, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, headless=args.headless)
    direction = np.asarray(CAMERA_TARGET) - CAMERA_POSITION
    vis.set_camera(
        pos=wp.vec3(*CAMERA_POSITION),
        pitch=float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))),
        yaw=float(np.degrees(np.arctan2(direction[1], direction[0]))),
    )
    topology = {name: wp.array(faces.ravel(), dtype=wp.int32) for name, faces in playback.faces.items()}
    points = {name: wp.array(vertices, dtype=wp.vec3) for name, vertices in playback.sample(start).items()}
    writer = None
    rendered = 0
    print(f"Recorded geometry playback: {start:.6f}..{stop:.6f} s, {len(times)} frames. No contact physics.")
    try:
        if args.save_movie:
            writer = imageio.get_writer(args.output_dir / MOVIE_FILENAME, fps=args.fps)
        index = 0
        while index < len(times) and vis.is_running():
            frame_start = time.perf_counter()
            source_time = float(times[index])
            paused = vis.is_paused() and not (args.headless or args.usd or args.save_movie)
            vis.begin_frame((source_time - start) / args.speed)
            if index == 0:
                mophi.log_orientation_and_scale_reference(
                    vis,
                    axis_origin=AXIS_ORIGIN,
                    axis_length=AXIS_LENGTH,
                    scale_bar_center=SCALE_BAR_CENTER,
                    marker_radius=SCALE_MARKER_RADIUS,
                )
            for name, vertices in playback.sample(source_time).items():
                points[name].assign(vertices)
                vis.log_mesh(name, points[name], topology[name], color=MESH_COLORS[name])
            vis.end_frame()
            if not paused:
                if writer is not None:
                    writer.append_data(vis.get_frame().numpy())
                rendered += 1
                index += 1
            if REALTIME_PLAYBACK and not (args.headless or args.usd or args.save_movie):
                time.sleep(max(0, 1 / args.fps - (time.perf_counter() - frame_start)))
        if SAVE_SNAPSHOT and not args.usd and rendered and vis.is_running():
            imageio.imwrite(args.output_dir / SNAPSHOT_FILENAME, vis.get_frame().numpy())
    finally:
        if writer is not None:
            writer.close()
        vis.close()
    print(f"Rendered {rendered} frames. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
