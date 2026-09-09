"""Newton robot construction for the ANYmal multiphysics experiment."""

from dataclasses import dataclass

import warp as wp

import newton


@dataclass
class AnymalNewtonScene:
    builder: object
    model: object
    solver: object
    foot_spheres: object
    body_name_to_index: dict
    visual_descriptors: list
    foot_descriptors: list
    foot_radii: list
    foot_body_indices: list[int]


def build_anymal_newton_scene(
    urdf_path,
    geometry_utils,
    initial_joint_positions,
    foot_shank_names,
    base_height,
    base_yaw,
    joint_armature,
    joint_limit_stiffness,
    joint_limit_damping,
    shape_stiffness,
    shape_damping,
    shape_friction_stiffness,
    shape_friction,
    target_stiffness,
    target_damping,
    solver_iterations,
    maximum_joints,
    maximum_contacts,
):
    """Build the configured robot, descriptors, model, and Newton solver."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=joint_armature,
        limit_ke=joint_limit_stiffness,
        limit_kd=joint_limit_damping,
    )
    builder.default_shape_cfg.ke = shape_stiffness
    builder.default_shape_cfg.kd = shape_damping
    builder.default_shape_cfg.kf = shape_friction_stiffness
    builder.default_shape_cfg.mu = shape_friction
    builder.add_urdf(
        urdf_path,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, base_height),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), base_yaw),
        ),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    foot_spheres, body_name_to_index = geometry_utils.scan_and_enlarge_foot_spheres(builder)
    visual_descriptors = geometry_utils.collect_visual_body_part_descriptors(builder)
    builder.add_ground_plane()
    joint_name_to_index = {label.split("/")[-1]: index for index, label in enumerate(builder.joint_label)}
    for name, value in initial_joint_positions.items():
        index = joint_name_to_index.get(name)
        if index is None:
            raise ValueError(f"Joint '{name}' not found in builder.joint_label")
        builder.joint_q[index + 6] = value
    for index in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[index] = target_stiffness
        builder.joint_target_kd[index] = target_damping
    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        ls_parallel=False,
        ls_iterations=solver_iterations,
        njmax=maximum_joints,
        nconmax=maximum_contacts,
    )
    foot_descriptors, foot_radii = geometry_utils.build_foot_tip_descriptors(
        body_name_to_index, foot_spheres, foot_shank_names
    )
    return AnymalNewtonScene(
        builder,
        model,
        solver,
        foot_spheres,
        body_name_to_index,
        visual_descriptors,
        foot_descriptors,
        foot_radii,
        [int(descriptor["body_idx"]) for descriptor in foot_descriptors],
    )
