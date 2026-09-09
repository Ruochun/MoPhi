"""Shared UR10 construction and trajectory helpers for MoPhi demos."""

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def build_ur10_submodel(
    asset_file: str,
    pedestal_height: float,
    pedestal_radius: float,
    joint_stiffness: float,
    joint_damping: float,
    initial_joint_q=None,
):
    """Build the common fixed-pedestal, position-controlled UR10 submodel."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0.0, 0.0, pedestal_height)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    builder.add_shape_cylinder(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, pedestal_height / 2.0)),
        half_height=pedestal_height / 2.0,
        radius=pedestal_radius,
        label="robot_pedestal",
    )
    for joint_index in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[joint_index] = joint_stiffness
        builder.joint_target_kd[joint_index] = joint_damping
        builder.joint_target_mode[joint_index] = int(JointTargetMode.POSITION)
    if initial_joint_q is not None:
        initial_joint_q = np.asarray(initial_joint_q)
        if len(builder.joint_q) < len(initial_joint_q):
            raise ValueError(
                f"UR10 exposes only {len(builder.joint_q)} joint coordinates; expected {len(initial_joint_q)}."
            )
        builder.joint_q[: len(initial_joint_q)] = initial_joint_q.tolist()
    return builder, builder.body_label.index("/ur10/ee_link")
