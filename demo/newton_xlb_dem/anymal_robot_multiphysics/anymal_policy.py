"""GPU policy observation mechanics for the ANYmal demo variants."""

import torch


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of an XYZW quaternion."""
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_observation(actions, joint_q, joint_qd, initial_joint_positions, indices, gravity, command):
    """Compute the 48-component observation expected by the walking policy."""
    root_quaternion = joint_q[3:7].to(dtype=torch.float32).unsqueeze(0)
    linear_velocity = joint_qd[:3].to(dtype=torch.float32).unsqueeze(0)
    angular_velocity = joint_qd[3:6].to(dtype=torch.float32).unsqueeze(0)
    joint_position = joint_q[7:].to(dtype=torch.float32).unsqueeze(0)
    joint_velocity = joint_qd[6:].to(dtype=torch.float32).unsqueeze(0)
    body_linear_velocity = quat_rotate_inverse(root_quaternion, linear_velocity)
    body_angular_velocity = quat_rotate_inverse(root_quaternion, angular_velocity)
    projected_gravity = quat_rotate_inverse(root_quaternion, gravity)
    position_error = torch.index_select(joint_position - initial_joint_positions, 1, indices)
    ordered_velocity = torch.index_select(joint_velocity, 1, indices)
    return torch.cat(
        [
            body_linear_velocity,
            body_angular_velocity,
            projected_gravity,
            command,
            position_error,
            ordered_velocity,
            actions,
        ],
        dim=1,
    )


def evaluate_joint_targets(policy, observation, initial_joint_positions, reorder_indices, free_joint_zeros, scale):
    """Evaluate the policy and return its action plus Newton-ordered targets."""
    with torch.no_grad():
        actions = policy(observation)
        ordered_actions = torch.gather(actions, 1, reorder_indices.unsqueeze(0))
        targets = initial_joint_positions + scale * ordered_actions
        return actions, torch.cat([free_joint_zeros, targets.squeeze(0)])
