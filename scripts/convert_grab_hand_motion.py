"""Prepare GRAB hand surfaces and recorded rigid-object geometry for playback."""

import argparse
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAB_ROOT = Path.home() / "GRAB" / "extracted"
DEFAULT_MODEL_ROOT = Path.home() / "GRAB" / "models"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "grab"
HAND_NAMES = {"left": "lhand", "right": "rhand"}


def convert_object(grab_root: Path, sequence: Path, output_root: Path, start=0, stop=None) -> Path:
    """Export the original object mesh and rigid poses into a pickle-free NPZ."""
    relative_sequence = sequence.resolve().relative_to((grab_root / "grab").resolve())
    with np.load(sequence, allow_pickle=True) as data:
        frame_count = int(data["n_frames"])
        fps = float(data["framerate"])
        obj = data["object"].item()
    end = frame_count if stop is None else stop
    if not 0 <= start < end <= frame_count or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid object frame range or recording rate")
    mesh_path = grab_root / str(obj["object_mesh"])
    mesh = trimesh.load(mesh_path, process=False)
    params = obj["params"]
    if params["transl"].shape != (frame_count, 3) or params["global_orient"].shape != (frame_count, 3):
        raise ValueError("Object parameter shapes disagree with n_frames")
    frames = np.arange(start, end, dtype=np.int32)
    suffix = "" if start == 0 and end == frame_count else f"_frames_{start:06d}_{end:06d}"
    path = output_root / relative_sequence.parent / f"{relative_sequence.stem}{suffix}_object.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.int32(1),
        source_sequence=str(relative_sequence),
        source_frame_indices=frames,
        source_fps=np.float64(fps),
        timestamps_s=frames.astype(np.float64) / fps,
        object_name=mesh_path.stem,
        object_vertices_local=np.asarray(mesh.vertices, dtype=np.float32),
        object_faces=np.asarray(mesh.faces, dtype=np.int32),
        object_translation_world=np.asarray(params["transl"][frames], dtype=np.float32),
        object_quaternion_xyzw=Rotation.from_rotvec(params["global_orient"][frames]).as_quat().astype(np.float32),
    )
    return path


def convert_sequence(
    grab_root: Path,
    model_root: Path,
    sequence: Path,
    output_root: Path,
    hands: tuple[str, ...],
    batch_size: int,
    start: int,
    stop: int | None,
) -> list[Path]:
    """Reconstruct fixed-topology hand surfaces from one trusted GRAB sequence."""
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    sequence = sequence.resolve()
    source_root = (grab_root / "grab").resolve()
    try:
        relative_sequence = sequence.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"Sequence must be beneath {source_root}: {sequence}") from exc
    if not sequence.is_file():
        raise FileNotFoundError(sequence)

    with np.load(sequence, allow_pickle=True) as data:
        frame_count = int(data["n_frames"])
        fps = float(data["framerate"])
        gender = str(data["gender"].item())
        n_comps = int(data["n_comps"])
        body = data["body"].item()
        parameters = body["params"]
        template_path = grab_root / str(body["vtemp"])
    end = frame_count if stop is None else stop
    if not (0 <= start < end <= frame_count):
        raise ValueError(f"Frame range [{start}, {end}) is invalid for {frame_count} frames")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid GRAB frame rate: {fps}")
    if any(len(value) != frame_count for key, value in parameters.items() if key != "fullpose"):
        raise ValueError("Body parameter frame counts disagree with n_frames")
    model_file = model_root / "smplx" / f"SMPLX_{gender.upper()}.npz"
    if not model_file.is_file():
        raise FileNotFoundError(f"Extract {model_file.name} from models_smplx_v1_1.zip into {model_file.parent}")
    if not template_path.is_file():
        raise FileNotFoundError(template_path)

    template = np.asarray(trimesh.load(template_path, process=False).vertices, dtype=np.float32)
    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender=gender,
        ext="npz",
        num_pca_comps=n_comps,
        batch_size=batch_size,
        v_template=template,
    ).eval()
    frame_indices = np.arange(start, end, dtype=np.int32)
    geometry = {}
    for hand in hands:
        stem = HAND_NAMES[hand]
        correspondence = grab_root / "tools" / "smplx_correspondence"
        ids = np.load(correspondence / f"{stem}_smplx_ids.npy").astype(np.int64)
        faces = np.load(correspondence / f"{stem}_faces.npy").astype(np.int32)
        if ids.ndim != 1 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Invalid {hand} hand correspondence topology")
        if np.any(faces < 0) or np.any(faces >= len(ids)):
            raise ValueError(f"Invalid {hand} hand face indices")
        geometry[hand] = (ids, faces, np.empty((len(frame_indices), len(ids), 3), dtype=np.float32))

    with torch.no_grad():
        for offset in range(0, len(frame_indices), batch_size):
            take = frame_indices[offset : offset + batch_size]
            arguments = {}
            for key, value in parameters.items():
                if key == "fullpose":
                    continue
                batch = np.asarray(value[take], dtype=np.float32)
                if len(take) < batch_size:
                    batch = np.concatenate((batch, np.repeat(batch[-1:], batch_size - len(take), axis=0)))
                arguments[key] = torch.from_numpy(batch)
            vertices = model(**arguments).vertices[: len(take)].numpy()
            for ids, _, positions in geometry.values():
                positions[offset : offset + len(take)] = vertices[:, ids]

    output_paths = []
    for hand, (_, faces, world_vertices) in geometry.items():
        # This mesh reference is geometric, not a physical center of mass.
        origin = world_vertices.mean(axis=1, dtype=np.float64).astype(np.float32)
        local_vertices = world_vertices - origin[:, None, :]
        frame_suffix = "" if start == 0 and end == frame_count else f"_frames_{start:06d}_{end:06d}"
        path = output_root / relative_sequence.parent / f"{relative_sequence.stem}{frame_suffix}_{hand}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.int32(1),
            source_sequence=str(relative_sequence),
            hand_side=hand,
            origin_kind="mean_hand_vertex_position",
            source_frame_indices=frame_indices,
            source_fps=np.float64(fps),
            timestamps_s=frame_indices.astype(np.float64) / fps,
            hand_origin_world=origin,
            hand_vertices_local=local_vertices,
            hand_faces=faces,
        )
        output_paths.append(path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="Sequence .npz under <grab-root>/grab/<subject>/")
    parser.add_argument("--grab-root", type=Path, default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--object-only", action="store_true", help="Prepare only the object; reuse existing hand exports"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0, help="First source frame, inclusive")
    parser.add_argument("--stop", type=int, help="Last source frame, exclusive")
    args = parser.parse_args()
    hands = ("left", "right") if args.hand == "both" else (args.hand,)
    paths = (
        []
        if args.object_only
        else convert_sequence(
            args.grab_root,
            args.model_root,
            args.sequence,
            args.output_root,
            hands,
            args.batch_size,
            args.start,
            args.stop,
        )
    )
    paths.append(convert_object(args.grab_root, args.sequence, args.output_root, args.start, args.stop))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
