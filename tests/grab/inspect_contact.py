"""Plot and audit cup-only mesh-plane force and rotation diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# ── Configuration ────────────────────────────────────────────────────────────
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output/inspect_grab_contact"
LATE_INTERVAL_FRACTION = 0.5
FIGURE_SIZE = (14, 13)
FIGURE_DPI = 150


def reconstruct_contact_wrenches(positions, contacts):
    """Sum force and r-cross-force about each sampled cup CoM.

    contacts.npz stores force on mesh owner A (the isolated cup); the reported
    model torque excludes r-cross-force. Empty contact spans remain zero.
    """
    offsets = contacts["sample_offsets"]
    if len(offsets) != len(positions) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("Invalid contact sample offsets")
    if offsets[-1] != len(contacts["force_on_cup_world"]):
        raise ValueError("Contact offsets do not match contact array length")
    rows = np.repeat(np.arange(len(positions)), np.diff(offsets))
    forces = contacts["force_on_cup_world"]
    normal = contacts["normal_world"]
    normal_forces = np.sum(forces * normal, axis=1)[:, None] * normal
    arms = contacts["point_world"] - positions[rows]
    result = {
        key: np.zeros_like(positions, dtype=float)
        for key in ("force", "normal_torque", "tangent_torque", "model_torque")
    }
    np.add.at(result["force"], rows, forces)
    np.add.at(result["normal_torque"], rows, np.cross(arms, normal_forces))
    np.add.at(result["tangent_torque"], rows, np.cross(arms, forces - normal_forces))
    np.add.at(result["model_torque"], rows, contacts["model_torque_on_cup_world"])
    result["torque"] = result["normal_torque"] + result["tangent_torque"] + result["model_torque"]
    return result


def analyze_run(path):
    summary = json.loads((path / "summary.json").read_text())
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        d = {key: archive[key] for key in archive.files}
    with np.load(path / "contacts.npz", allow_pickle=False) as archive:
        c = {key: archive[key] for key in archive.files}
    t = d["time_s"]
    rotation = Rotation.from_quat(d["cup_quaternion_xyzw"])
    relative = (rotation * rotation[0].inv()).as_matrix()
    yaw = np.degrees(np.unwrap(np.arctan2(relative[:, 1, 0], relative[:, 0, 0])))
    omega = d["cup_angular_velocity_world"]
    # L_world = R I_principal R^T omega_world, not I_principal * omega_world.
    local_omega = rotation.inv().apply(omega)
    momentum = rotation.apply(local_omega * d["principal_moi_kg_m2"])
    kinetic = 0.5 * np.sum(local_omega**2 * d["principal_moi_kg_m2"], axis=1)
    torque = d["cup_contact_torque_world"]
    impulse = np.cumsum(torque[1:] * np.diff(t)[:, None], axis=0)
    impulse = np.vstack([np.zeros(3), impulse])
    reconstruction = reconstruct_contact_wrenches(d["cup_com_world"], c)
    late = t >= t[-1] * LATE_INTERVAL_FRACTION
    counts = np.diff(c["sample_offsets"])
    metrics = {
        "source": str(path.resolve()),
        "deme_version": summary["deme_version"],
        "collision_triangles": summary["cup_collision_triangles"],
        "patch_count": summary["cup_patch_count"],
        "floor_friction": summary["push_floor_friction"],
        "dt_s": summary["dt_s"],
        "duration_s": float(t[-1]),
        "final_yaw_deg": float(yaw[-1]),
        "final_omega_z_rad_s": float(omega[-1, 2]),
        "final_angular_momentum_z": float(momentum[-1, 2]),
        "integrated_contact_torque_z": float(impulse[-1, 2]),
        "angular_impulse_z_residual": float(momentum[-1, 2] - momentum[0, 2] - impulse[-1, 2]),
        "final_rotational_kinetic_energy_J": float(kinetic[-1]),
        "late_mean_force_N": d["cup_contact_force_world"][late].mean(axis=0).tolist(),
        "late_mean_torque_Nm": torque[late].mean(axis=0).tolist(),
        "maximum_active_contacts": int(counts.max()),
        "force_reconstruction_max_error_N": float(
            np.max(np.abs(reconstruction["force"] - d["cup_contact_force_world"]))
        ),
        "torque_reconstruction_max_error_Nm": float(np.max(np.abs(reconstruction["torque"] - torque))),
        "normal_yaw_torque_max_Nm": float(np.abs(reconstruction["normal_torque"][:, 2]).max()),
        "tangential_yaw_torque_max_Nm": float(np.abs(reconstruction["tangent_torque"][:, 2]).max()),
        "model_torque_max_Nm": float(np.abs(reconstruction["model_torque"]).max()),
        "minimum_individual_normal_force_N": (
            float(np.sum(c["force_on_cup_world"] * c["normal_world"], axis=1).min()) if len(c["time_s"]) else 0.0
        ),
        "first_contact_time_s": float(t[np.flatnonzero(counts)[0]]) if counts.any() else None,
    }
    return d, c, reconstruction, yaw, momentum, impulse, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="cup_rest result directories containing trajectory, contacts, and summary",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, layout="constrained")
    reports = {}
    for index, path in enumerate(args.run_dirs):
        d, c, w, yaw, momentum, impulse, metrics = analyze_run(path)
        label = f"{path.parent.name}/{path.name}"
        reports[label] = metrics
        t = d["time_s"]
        axes[0, 0].plot(t, yaw, label=label)
        axes[0, 1].plot(t, d["cup_angular_velocity_world"][:, 2], label=label)
        axes[1, 0].plot(t, momentum[:, 2], label=label)
        axes[1, 1].plot(t, np.diff(c["sample_offsets"]), label=label)
        fig, a = plt.subplots(4, 2, figsize=FIGURE_SIZE, sharex=True, layout="constrained")
        a[0, 0].plot(t, yaw)
        a[0, 0].set_ylabel("Relative yaw (degrees)")
        a[0, 1].plot(t, d["cup_angular_velocity_world"])
        a[0, 1].legend(["X", "Y", "Z"])
        a[0, 1].set_ylabel("Angular velocity (rad/s)")
        a[1, 0].plot(t, d["cup_contact_force_world"])
        a[1, 0].legend(["X", "Y", "Z"])
        a[1, 0].set_ylabel("Contact force (N)")
        a[1, 1].plot(t, d["cup_contact_torque_world"])
        a[1, 1].legend(["X", "Y", "Z"])
        a[1, 1].set_ylabel("Contact torque about CoM (Nm)")
        a[2, 0].plot(t, momentum[:, 2] - momentum[0, 2], label="Change in Lz")
        a[2, 0].plot(t, impulse[:, 2], "--", label="Integrated contact torque")
        a[2, 0].legend()
        a[2, 0].set_ylabel("Angular momentum / impulse (kg m²/s)")
        for name in ("normal_torque", "tangent_torque", "model_torque"):
            a[2, 1].plot(t, w[name][:, 2], label=name)
        a[2, 1].legend()
        a[2, 1].set_ylabel("Vertical-axis torque (Nm)")
        a[3, 0].plot(t, d["cup_min_z"] * 1000)
        a[3, 0].set_ylabel("Cup bottom height (mm)")
        a[3, 1].plot(t, np.diff(c["sample_offsets"]))
        a[3, 1].set_ylabel("Active contact patches")
        for axis in a.flat:
            axis.grid(alpha=0.25)
            axis.set_xlabel("Time (s)")
        fig.suptitle(label + ": isolated cup on analytical plane")
        fig.savefig(args.output_dir / f"run_{index}_{path.parent.name}.png", dpi=FIGURE_DPI)
        plt.close(fig)
    for axis, ylabel in zip(
        axes.flat,
        (
            "Relative yaw (degrees)",
            "Vertical angular velocity (rad/s)",
            "Vertical angular momentum (kg m²/s)",
            "Active contact patches",
        ),
    ):
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.25)
        axis.legend()
    comparison.savefig(args.output_dir / "comparison.png", dpi=FIGURE_DPI)
    plt.close(comparison)
    (args.output_dir / "force_analysis.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
