"""Validated GRAB assets and time sampling for recorded geometry playback."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def validate_metadata(data: dict, path: Path, names) -> None:
    """Require scalar metadata before interpreting version, provenance, or units."""
    for name in names:
        if data[name].shape != ():
            raise ValueError(f"{path}: {name} must be scalar")
    if data["source_sequence"].dtype.kind not in "US" or not str(data["source_sequence"]):
        raise ValueError(f"{path}: source_sequence must be a nonempty string")


def load_motion(path: Path) -> dict:
    """Load and validate one converter output before any plotting."""
    required = {
        "format_version",
        "source_sequence",
        "hand_side",
        "origin_kind",
        "source_frame_indices",
        "source_fps",
        "timestamps_s",
        "hand_origin_world",
        "hand_vertices_local",
        "hand_faces",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
        data = {key: archive[key] for key in required}
    validate_metadata(data, path, ("format_version", "source_sequence", "source_fps", "hand_side", "origin_kind"))
    if int(data["format_version"]) != 1:
        raise ValueError(f"{path}: unsupported format_version")
    side = str(data["hand_side"])
    if side not in ("left", "right"):
        raise ValueError(f"{path}: unknown hand_side {side!r}")
    if str(data["origin_kind"]) != "mean_hand_vertex_position":
        raise ValueError(f"{path}: unknown origin_kind")
    frames = data["source_frame_indices"]
    times = data["timestamps_s"]
    if times.ndim != 1:
        raise ValueError(f"{path}: timestamps_s must be one-dimensional")
    origins = data["hand_origin_world"]
    vertices = data["hand_vertices_local"]
    faces = data["hand_faces"]
    count = len(times)
    expected_dtypes = {
        "source_frame_indices": np.int32,
        "timestamps_s": np.float64,
        "hand_origin_world": np.float32,
        "hand_vertices_local": np.float32,
        "hand_faces": np.int32,
    }
    for name, dtype in expected_dtypes.items():
        if data[name].dtype != dtype:
            raise ValueError(f"{path}: {name} must have dtype {np.dtype(dtype)}")
    if (
        frames.shape != (count,)
        or origins.shape != (count, 3)
        or vertices.ndim != 3
        or vertices.shape[0] != count
        or vertices.shape[2] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or count == 0
        or vertices.shape[1] == 0
        or len(faces) == 0
    ):
        raise ValueError(f"{path}: inconsistent frame or mesh shapes")
    fps = float(data["source_fps"])
    if not np.isfinite(fps) or fps <= 0 or not np.all(np.diff(frames) > 0):
        raise ValueError(f"{path}: invalid frame indices or source_fps")
    if np.any(frames < 0):
        raise ValueError(f"{path}: source frame indices must be nonnegative")
    if not np.allclose(times, frames / fps, rtol=0, atol=1e-8):
        raise ValueError(f"{path}: timestamps do not match source frames and fps")
    if not (np.isfinite(origins).all() and np.isfinite(vertices).all()):
        raise ValueError(f"{path}: nonfinite geometry")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[1]):
        raise ValueError(f"{path}: invalid hand face indices")
    if np.max(np.abs(vertices.mean(axis=1, dtype=np.float64))) > 1e-5:
        raise ValueError(f"{path}: local vertices are not centered on the stored origin")
    data["path"] = path
    return data


def load_object_motion(path: Path) -> dict:
    """Load a prepared rigid object without pickle, GRAB sources, or SMPL-X."""
    required = {
        "format_version",
        "source_sequence",
        "source_frame_indices",
        "source_fps",
        "timestamps_s",
        "object_name",
        "object_vertices_local",
        "object_faces",
        "object_translation_world",
        "object_quaternion_xyzw",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
        data = {key: archive[key] for key in required}
    validate_metadata(data, path, ("format_version", "source_sequence", "source_fps", "object_name"))
    if data["format_version"].shape != () or int(data["format_version"]) != 1:
        raise ValueError(f"{path}: unsupported format_version")
    for name, dtype in {
        "source_frame_indices": np.int32,
        "timestamps_s": np.float64,
        "object_vertices_local": np.float32,
        "object_faces": np.int32,
        "object_translation_world": np.float32,
        "object_quaternion_xyzw": np.float32,
    }.items():
        if data[name].dtype != dtype:
            raise ValueError(f"{path}: {name} must have dtype {np.dtype(dtype)}")
    times, frames = data["timestamps_s"], data["source_frame_indices"]
    fps = float(data["source_fps"])
    if (
        times.ndim != 1
        or len(times) == 0
        or frames.shape != times.shape
        or not np.isfinite(fps)
        or fps <= 0
        or np.any(frames < 0)
        or not np.all(np.diff(frames) > 0)
        or not np.allclose(times, frames / fps, rtol=0, atol=1e-8)
    ):
        raise ValueError(f"{path}: invalid timestamps, source frames, or source_fps")
    vertices, faces = data["object_vertices_local"], data["object_faces"]
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) == 0
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError(f"{path}: invalid object mesh or face indices")
    translations, quaternions = data["object_translation_world"], data["object_quaternion_xyzw"]
    if translations.shape != (len(times), 3) or quaternions.shape != (len(times), 4):
        raise ValueError(f"{path}: inconsistent object pose shapes")
    if not all(np.isfinite(a).all() for a in (vertices, translations, quaternions)):
        raise ValueError(f"{path}: nonfinite geometry or poses")
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError(f"{path}: object quaternions must be normalized xyzw rotations")
    data["path"] = path
    return data


def sample_interval(timestamps: np.ndarray, time_s: float) -> tuple[int, int, float]:
    """Return bracketing rows and linear weight; reject extrapolation."""
    if not np.isfinite(time_s) or time_s < timestamps[0] or time_s > timestamps[-1]:
        raise ValueError(f"Time {time_s} outside [{timestamps[0]}, {timestamps[-1]}]")
    left = max(0, int(np.searchsorted(timestamps, time_s, side="right")) - 1)
    right = min(left + 1, len(timestamps) - 1)
    weight = 0.0 if left == right else (time_s - timestamps[left]) / (timestamps[right] - timestamps[left])
    return left, right, float(weight)


class GrabPlayback:
    """Sample synchronized hand surfaces and rigid object poses in metres.

    Hand positions and object translations interpolate linearly. Object rotation
    follows shortest-path quaternion SLERP. This CPU geometry utility performs
    no dynamics, contact, or velocity transfer to a solver.
    """

    def __init__(self, hand_paths, object_path):
        self.hands = [load_motion(Path(path)) for path in hand_paths]
        self.object = load_object_motion(Path(object_path))
        if not 1 <= len(self.hands) <= 2:
            raise ValueError("Playback requires one or two hand files")
        if len({str(hand["hand_side"]) for hand in self.hands}) != len(self.hands):
            raise ValueError("Hand sides must be unique")
        assets = [*self.hands, self.object]
        if len({str(asset["source_sequence"]) for asset in assets}) != 1:
            raise ValueError("All assets must come from the same source_sequence")
        if len({float(asset["source_fps"]) for asset in assets}) != 1:
            raise ValueError("All assets must have the same source_fps")
        self.start_time = max(float(asset["timestamps_s"][0]) for asset in assets)
        self.end_time = min(float(asset["timestamps_s"][-1]) for asset in assets)
        if self.start_time > self.end_time:
            raise ValueError("Assets have no overlapping time interval")
        self._rotations = Rotation.from_quat(self.object["object_quaternion_xyzw"])
        self._slerp = (
            Slerp(self.object["timestamps_s"], self._rotations) if len(self.object["timestamps_s"]) > 1 else None
        )

    def sample(self, time_s: float) -> dict[str, np.ndarray]:
        """Return world-space vertices keyed by left/right/object at source time."""
        if not np.isfinite(time_s) or not self.start_time <= time_s <= self.end_time:
            raise ValueError("Requested time is outside the shared playback interval")
        result = {}
        for hand in self.hands:
            a, b, w = sample_interval(hand["timestamps_s"], time_s)
            origin = (1 - w) * hand["hand_origin_world"][a] + w * hand["hand_origin_world"][b]
            local = (1 - w) * hand["hand_vertices_local"][a] + w * hand["hand_vertices_local"][b]
            result[str(hand["hand_side"])] = np.asarray(local + origin, dtype=np.float32)
        obj = self.object
        a, b, w = sample_interval(obj["timestamps_s"], time_s)
        translation = (1 - w) * obj["object_translation_world"][a] + w * obj["object_translation_world"][b]
        rotation = self._rotations[0] if self._slerp is None else self._slerp(time_s)
        result["object"] = np.asarray(rotation.apply(obj["object_vertices_local"]) + translation, dtype=np.float32)
        return result

    @property
    def faces(self) -> dict[str, np.ndarray]:
        """Fixed indexed triangle topology for every surface."""
        return {**{str(h["hand_side"]): h["hand_faces"] for h in self.hands}, "object": self.object["object_faces"]}


def playback_times(start: float, stop: float, fps: float, speed: float = 1.0) -> np.ndarray:
    """Source timestamps at a fixed playback cadence, including the final pose.

    The last interval may be shorter than speed/fps. Holding every sample for
    one movie frame can extend the movie by up to two frame periods.
    """
    if not np.isfinite([start, stop, fps, speed]).all() or stop < start or fps <= 0 or speed <= 0:
        raise ValueError("Require finite start <= stop, positive fps and playback speed")
    times = start + np.arange(int(np.floor((stop - start) * fps / speed)) + 1) * speed / fps
    times = times[times < stop]
    return np.append(times, stop)
