"""Scenario helpers shared by the Newton and DEME flexible-hand demos."""

import numpy as np
import warp as wp


@wp.kernel
def update_finger_targets(
    target_indices: wp.array(dtype=wp.int32),
    centers: wp.array(dtype=wp.float32),
    amplitudes: wp.array(dtype=wp.float32),
    phase_cycles: wp.array(dtype=wp.float32),
    lower_limits: wp.array(dtype=wp.float32),
    upper_limits: wp.array(dtype=wp.float32),
    sim_time: float,
    frequency: float,
    targets: wp.array(dtype=wp.float32),
):
    target_idx = wp.tid()
    target_dof_idx = target_indices[target_idx]
    phase = 2.0 * wp.pi * (frequency * sim_time + phase_cycles[target_idx])
    target = centers[target_idx] + amplitudes[target_idx] * wp.sin(phase)
    targets[target_dof_idx] = wp.clamp(target, lower_limits[target_dof_idx], upper_limits[target_dof_idx])


@wp.kernel
def update_hand_root_descend_lift(
    root_q_starts: wp.array(dtype=wp.int32),
    root_qd_starts: wp.array(dtype=wp.int32),
    initial_root_heights: wp.array(dtype=wp.float32),
    sim_time: float,
    descend_start: float,
    descend_duration: float,
    descend_depth: float,
    lift_start: float,
    lift_duration: float,
    lift_height: float,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    hand_idx = wp.tid()
    q_start = root_q_starts[hand_idx]
    qd_start = root_qd_starts[hand_idx]
    descend_fraction = wp.clamp((sim_time - descend_start) / descend_duration, 0.0, 1.0)
    descend_smooth = descend_fraction * descend_fraction * (3.0 - 2.0 * descend_fraction)
    descend_velocity = 6.0 * descend_fraction * (1.0 - descend_fraction) * descend_depth / descend_duration
    if sim_time <= descend_start or sim_time >= descend_start + descend_duration:
        descend_velocity = 0.0
    lift_fraction = wp.clamp((sim_time - lift_start) / lift_duration, 0.0, 1.0)
    lift_smooth = lift_fraction * lift_fraction * (3.0 - 2.0 * lift_fraction)
    lift_velocity = 6.0 * lift_fraction * (1.0 - lift_fraction) * lift_height / lift_duration
    if sim_time <= lift_start or sim_time >= lift_start + lift_duration:
        lift_velocity = 0.0
    joint_q[q_start + 2] = initial_root_heights[hand_idx] - descend_depth * descend_smooth + lift_height * lift_smooth
    joint_qd[qd_start + 0] = 0.0
    joint_qd[qd_start + 1] = 0.0
    joint_qd[qd_start + 2] = -descend_velocity + lift_velocity
    joint_qd[qd_start + 3] = 0.0
    joint_qd[qd_start + 4] = 0.0
    joint_qd[qd_start + 5] = 0.0


def configure_replicated_controls(
    hand,
    local_finger_dof_indices,
    local_centers,
    local_amplitudes,
    local_phases,
    local_root_q_start,
    local_root_qd_start,
    hand_count,
    grasp_profiles,
    phase_spread_cycles,
):
    """Expand one hand's controls and configured grasp profiles across replicas."""
    target_indices, centers, amplitudes, phases = [], [], [], []
    root_q_starts, root_qd_starts, profile_names = [], [], []
    for hand_idx in range(hand_count):
        profile_name, closure_scale, motion_scale, profile_phase = grasp_profiles[hand_idx % len(grasp_profiles)]
        profile_names.append(profile_name)
        world_dof_offset = hand_idx * hand.joint_dof_count
        world_coord_offset = hand_idx * hand.joint_coord_count
        hand_phase = hand_idx * (phase_spread_cycles / hand_count) + profile_phase
        root_q_starts.append(world_coord_offset + local_root_q_start)
        root_qd_starts.append(world_dof_offset + local_root_qd_start)
        for local_idx, center, amplitude, phase in zip(
            local_finger_dof_indices,
            local_centers * closure_scale,
            local_amplitudes * motion_scale,
            local_phases + hand_phase,
        ):
            target_indices.append(world_dof_offset + int(local_idx))
            centers.append(center)
            amplitudes.append(amplitude)
            phases.append(phase)
    return (
        np.asarray(target_indices, dtype=np.int32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(amplitudes, dtype=np.float32),
        np.asarray(phases, dtype=np.float32),
        np.asarray(root_q_starts, dtype=np.int32),
        np.asarray(root_qd_starts, dtype=np.int32),
        profile_names,
    )
