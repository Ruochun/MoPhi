"""Validate and render converted GRAB hand-motion files without SMPL-X."""

import argparse
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from mophi.utils.grab_motion import load_motion

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAB_ROOT = Path.home() / "GRAB" / "extracted"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "output" / "inspect_grab_hand_motion"
HAND_COLORS = {"left": "#3288bd", "right": "#e88b31"}


def select_frames(motions: list[dict], requested: list[int] | None) -> list[int]:
    """Select source frame numbers shared by every supplied motion file."""
    common = set(motions[0]["source_frame_indices"].tolist())
    for motion in motions[1:]:
        common.intersection_update(motion["source_frame_indices"].tolist())
    if not common:
        raise ValueError("The hand files have no common source frames")
    ordered = sorted(common)
    selected = requested if requested else [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    if not selected or any(frame not in common for frame in selected):
        raise ValueError(
            f"Requested frames must be present in every hand file; shared range is {ordered[0]}..{ordered[-1]}"
        )
    return list(dict.fromkeys(selected))


def load_source_object(grab_root: Path, source_sequence: str):
    """Load the recorded object mesh and pose from the trusted local GRAB source."""
    path = grab_root / "grab" / source_sequence
    if not path.is_file():
        raise FileNotFoundError(f"GRAB source sequence needed for --object: {path}")
    with np.load(path, allow_pickle=True) as archive:
        obj = archive["object"].item()
        params = obj["params"]
        mesh_path = grab_root / str(obj["object_mesh"])
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh = trimesh.load(mesh_path, process=False)
    return mesh, params


def add_frame_geometry(ax, motions: list[dict], frame: int, source_object=None, object_face_limit=24000):
    """Draw one recorded frame and return its world-space bounds."""
    bounds = []
    for motion in motions:
        row = int(np.searchsorted(motion["source_frame_indices"], frame))
        world = motion["hand_vertices_local"][row] + motion["hand_origin_world"][row]
        bounds.append(world)
        ax.add_collection3d(
            Poly3DCollection(
                world[motion["hand_faces"]],
                facecolor=HAND_COLORS[str(motion["hand_side"])],
                edgecolor="none",
                alpha=0.88,
            )
        )
    if source_object is not None:
        mesh, params = source_object
        object_world = Rotation.from_rotvec(params["global_orient"][frame]).apply(mesh.vertices)
        object_world += params["transl"][frame]
        bounds.append(object_world)
        stride = max(1, len(mesh.faces) // object_face_limit)
        ax.add_collection3d(
            Poly3DCollection(
                object_world[mesh.faces[::stride]],
                facecolor="#98a1aa",
                edgecolor="none",
                alpha=0.75,
            )
        )
    return np.concatenate(bounds)


def set_camera(ax, center: np.ndarray, radius: float, frame: int, fps: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    ax.view_init(elev=22, azim=-65)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"source frame {frame} · {frame / fps:.3f} s")


def render(motions: list[dict], frames: list[int], output: Path, source_object=None) -> None:
    """Render hand and optional recorded-object snapshots in metric world space."""
    figure = plt.figure(figsize=(5 * len(frames), 5.2), dpi=150)
    for column, frame in enumerate(frames, 1):
        ax = figure.add_subplot(1, len(frames), column, projection="3d")
        all_points = add_frame_geometry(ax, motions, frame, source_object)
        low = all_points.min(axis=0)
        high = all_points.max(axis=0)
        center = (low + high) / 2
        radius = max(float(np.max(high - low)) * 0.65, 0.06)
        set_camera(ax, center, radius, frame, float(motions[0]["source_fps"]))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def select_movie_frames(motions: list[dict], start: int | None, stop: int | None, stride: int) -> list[int]:
    """Select a contiguous shared source interval, sampled every stride frames."""
    if stride < 1:
        raise ValueError("--movie-stride must be positive")
    common = set(motions[0]["source_frame_indices"].tolist())
    for motion in motions[1:]:
        common.intersection_update(motion["source_frame_indices"].tolist())
    ordered = sorted(frame for frame in common if (start is None or frame >= start) and (stop is None or frame < stop))
    if not ordered:
        raise ValueError("No shared source frames in the requested movie range")
    if np.any(np.diff(ordered) != 1):
        raise ValueError("Movie playback requires consecutive source frames")
    return ordered[::stride]


def movie_bounds(motions: list[dict], frames: list[int], source_object=None) -> tuple[np.ndarray, float, float]:
    """Find whole-clip bounds and a constant scale for a following camera."""
    clip_low = np.full(3, np.inf)
    clip_high = np.full(3, -np.inf)
    follow_radius = 0.06
    corners = None
    if source_object is not None:
        mesh, params = source_object
        corners = np.array(list(product(*zip(mesh.bounds[0], mesh.bounds[1]))))
    for frame in frames:
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for motion in motions:
            row = int(np.searchsorted(motion["source_frame_indices"], frame))
            world = motion["hand_vertices_local"][row] + motion["hand_origin_world"][row]
            low = np.minimum(low, world.min(axis=0))
            high = np.maximum(high, world.max(axis=0))
        if source_object is not None:
            points = Rotation.from_rotvec(params["global_orient"][frame]).apply(corners)
            points += params["transl"][frame]
            low = np.minimum(low, points.min(axis=0))
            high = np.maximum(high, points.max(axis=0))
        clip_low = np.minimum(clip_low, low)
        clip_high = np.maximum(clip_high, high)
        follow_radius = max(follow_radius, float(np.max(high - low)) * 0.65)
    return (clip_low + clip_high) / 2, max(float(np.max(clip_high - clip_low)) * 0.65, 0.06), follow_radius


def render_movie(
    motions: list[dict],
    frames: list[int],
    output: Path,
    fps: float,
    source_object=None,
    camera: str = "follow",
) -> None:
    """Write an MP4 with a constant scale and one frame per selected source pose."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Movie fps must be positive and finite")
    if camera not in ("follow", "fixed"):
        raise ValueError("Movie camera must be 'follow' or 'fixed'")
    import imageio.v2 as imageio

    fixed_center, fixed_radius, follow_radius = movie_bounds(motions, frames, source_object)
    figure = plt.figure(figsize=(6.4, 6.4), dpi=100)
    ax = figure.add_subplot(111, projection="3d")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            output, format="FFMPEG", mode="I", fps=fps, codec="libx264", macro_block_size=1
        ) as writer:
            for frame in frames:
                ax.cla()
                world = add_frame_geometry(ax, motions, frame, source_object, object_face_limit=6000)
                center = (world.min(axis=0) + world.max(axis=0)) / 2 if camera == "follow" else fixed_center
                radius = follow_radius if camera == "follow" else fixed_radius
                set_camera(ax, center, radius, frame, float(motions[0]["source_fps"]))
                figure.canvas.draw()
                rgba = np.asarray(figure.canvas.buffer_rgba())
                writer.append_data(rgba[:, :, :3])
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motions", nargs="+", type=Path, help="One or two converted hand NPZ files")
    parser.add_argument("--frames", nargs="+", type=int, help="Source frame numbers; default is first, middle, last")
    parser.add_argument("--object", action="store_true", help="Overlay the object from the original GRAB sequence")
    parser.add_argument("--grab-root", type=Path, default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--movie", action="store_true", help="Write an MP4 time series instead of a PNG contact sheet")
    parser.add_argument("--movie-start", type=int, help="First source frame, inclusive")
    parser.add_argument("--movie-stop", type=int, help="Last source frame, exclusive")
    parser.add_argument("--movie-stride", type=int, default=1, help="Use every Nth source frame")
    parser.add_argument("--movie-fps", type=float, help="Playback FPS; default preserves source timing after stride")
    parser.add_argument("--movie-camera", choices=("follow", "fixed"), default="follow")
    args = parser.parse_args()
    if len(args.motions) > 2:
        parser.error("Supply at most two hand files")
    motions = [load_motion(path) for path in args.motions]
    if len({str(motion["hand_side"]) for motion in motions}) != len(motions):
        raise ValueError("Supply at most one file per hand side")
    sources = {str(motion["source_sequence"]) for motion in motions}
    if len(sources) != 1:
        raise ValueError("Hand files must come from the same GRAB sequence")
    source_object = load_source_object(args.grab_root, sources.pop()) if args.object else None
    if args.movie:
        if args.frames is not None:
            parser.error("Use --movie-start/--movie-stop to select movie frames, not --frames")
        frames = select_movie_frames(motions, args.movie_start, args.movie_stop, args.movie_stride)
        fps = args.movie_fps if args.movie_fps is not None else float(motions[0]["source_fps"]) / args.movie_stride
        output = args.output or DEFAULT_OUTPUT_ROOT / f"{args.motions[0].stem}_inspection.mp4"
        render_movie(motions, frames, output, fps, source_object, camera=args.movie_camera)
    else:
        if (
            args.movie_start is not None
            or args.movie_stop is not None
            or args.movie_fps is not None
            or args.movie_stride != 1
            or args.movie_camera != "follow"
        ):
            parser.error("Movie range and FPS options require --movie")
        frames = select_frames(motions, args.frames)
        output = args.output or DEFAULT_OUTPUT_ROOT / f"{args.motions[0].stem}_inspection.png"
        render(motions, frames, output, source_object)
    for motion in motions:
        print(
            f"{motion['path']}: {len(motion['timestamps_s'])} frames, "
            f"{motion['hand_vertices_local'].shape[1]} vertices, {len(motion['hand_faces'])} faces; valid"
        )
    print(f"Rendered {len(frames)} source frames ({frames[0]}..{frames[-1]}) to {output}")


if __name__ == "__main__":
    main()
