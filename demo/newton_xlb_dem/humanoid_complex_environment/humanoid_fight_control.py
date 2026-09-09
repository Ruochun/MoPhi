"""Scripted strike trajectories for the humanoid fighting scenario."""

import numpy as np


def _interpolate(progress, keyframes):
    progress = float(np.clip(progress, 0.0, 1.0))
    for keyframe_idx in range(len(keyframes) - 1):
        time_0, offsets_0 = keyframes[keyframe_idx]
        time_1, offsets_1 = keyframes[keyframe_idx + 1]
        if progress <= time_1:
            blend = (progress - time_0) / (time_1 - time_0)
            return {
                name: (1.0 - blend) * offsets_0.get(name, 0.0) + blend * offsets_1.get(name, 0.0)
                for name in offsets_0.keys() | offsets_1.keys()
            }
    return dict(keyframes[-1][1])


def scripted_attack_offsets(attack_kind, side, progress, fist_curl, punch_motion_scale):
    """Return the scenario-defined whole-body residual for one punch or kick."""
    if side not in ("left", "right"):
        raise ValueError(f"Attack side must be 'left' or 'right', got {side!r}.")
    side_sign = -1.0 if side == "left" else 1.0
    prefix = f"{side}_"
    if attack_kind == "punch":
        fist = {
            f"{prefix}hand_index_0_joint": fist_curl,
            f"{prefix}hand_index_1_joint": fist_curl,
            f"{prefix}hand_middle_0_joint": fist_curl,
            f"{prefix}hand_middle_1_joint": fist_curl,
            f"{prefix}hand_thumb_0_joint": 0.6 * fist_curl,
            f"{prefix}hand_thumb_1_joint": fist_curl,
            f"{prefix}hand_thumb_2_joint": fist_curl,
        }
        windup = {
            **fist,
            f"{prefix}shoulder_pitch_joint": 0.45,
            f"{prefix}shoulder_roll_joint": 0.25 * side_sign,
            f"{prefix}shoulder_yaw_joint": -0.35 * side_sign,
            f"{prefix}elbow_joint": 1.25,
            "waist_yaw_joint": -0.25 * side_sign,
        }
        strike = {
            **fist,
            f"{prefix}shoulder_pitch_joint": -1.25,
            f"{prefix}shoulder_roll_joint": -0.12 * side_sign,
            f"{prefix}shoulder_yaw_joint": 0.18 * side_sign,
            f"{prefix}elbow_joint": 0.05,
            "waist_yaw_joint": 0.35 * side_sign,
        }
        offsets = _interpolate(progress, ((0.0, fist), (0.30, windup), (0.58, strike), (0.78, strike), (1.0, fist)))
        return {name: value if "hand_" in name else punch_motion_scale * value for name, value in offsets.items()}
    if attack_kind == "kick":
        other_side = "right" if side == "left" else "left"
        chamber = {
            f"{prefix}hip_pitch_joint": -0.75,
            f"{prefix}knee_joint": 1.15,
            f"{prefix}ankle_pitch_joint": -0.45,
            f"{other_side}_knee_joint": 0.18,
            "waist_pitch_joint": 0.18,
            "waist_roll_joint": -0.12 * side_sign,
        }
        strike = {
            f"{prefix}hip_pitch_joint": -1.05,
            f"{prefix}knee_joint": 0.05,
            f"{prefix}ankle_pitch_joint": 0.15,
            f"{other_side}_knee_joint": 0.22,
            "waist_pitch_joint": 0.28,
            "waist_roll_joint": -0.16 * side_sign,
        }
        return _interpolate(progress, ((0.0, {}), (0.38, chamber), (0.62, strike), (0.78, strike), (1.0, {})))
    raise ValueError(f"Attack kind must be 'punch' or 'kick', got {attack_kind!r}.")
