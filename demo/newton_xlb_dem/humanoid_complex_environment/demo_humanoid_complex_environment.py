"""First-stage humanoid complex-environment demo.

This demo adapts Newton 1.0.0's public ``example_robot_policy.py`` Unitree G1
walking baseline for MoPhi. Newton downloads the robot model, policy, and YAML
configuration from the public ``newton-assets`` repository on first run.

The demo adds a finite run, movie generation, run metadata, and a final-state
snapshot. Complex terrain and objects added interactively during simulation
are planned follow-on stages.

Keyboard controls
-----------------
  I / K  -- walk forward / backward
  J / L  -- strafe left / right
  U / O  -- turn left / right
  P      -- reset

Running
-------
From the MoPhi build tree:

    cmake --build <build-dir> --target demo_humanoid_complex_environment

Or from the repository root after installing the mophi package:

    python demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_complex_environment.py
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp
import yaml

import mophi
import newton
import newton.utils
from newton.examples.robot import example_robot_policy as newton_robot_policy

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
RENDER_FPS = 50
NEWTON_DT = 1.0 / 200.0
NUM_FRAMES = 500

# -- Robot --------------------------------------------------------------------
ROBOT_NAME = "g1_29dof"

# -- Visualization ------------------------------------------------------------
USE_OMNIVERSE_VISUALIZATION = False
CAMERA_POSITION = (3.0, -4.0, 2.0)
CAMERA_PITCH = -12.0
CAMERA_YAW = 135.0

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_humanoid_complex_environment.mp4"
MOVIE_FPS = RENDER_FPS
SAVE_FINAL_STATE = True
FINAL_STATE_OUTPUT_PATH = OUTPUT_DIRECTORY / "final_state.npz"
RUN_METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
SIM_SUBSTEPS = int(round(FRAME_DT / NEWTON_DT))
if not np.isclose(SIM_SUBSTEPS * NEWTON_DT, FRAME_DT, rtol=0.0, atol=1.0e-12):
    raise ValueError("FRAME_DT must be an integer multiple of NEWTON_DT.")


def _write_run_metadata(frame_count: int, asset_directory: Path) -> None:
    metadata = {
        "baseline": "newton.examples.robot.example_robot_policy",
        "robot": ROBOT_NAME,
        "newton_version": newton.__version__,
        "asset_directory": str(asset_directory),
        "newton_dt": NEWTON_DT,
        "render_fps": RENDER_FPS,
        "requested_frames": NUM_FRAMES,
        "completed_frames": frame_count,
        "sim_duration": frame_count * FRAME_DT,
    }
    RUN_METADATA_OUTPUT_PATH.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    if USE_OMNIVERSE_VISUALIZATION:
        mophi.fatal("Newton's interactive G1 policy baseline currently requires ViewerGL.")
    if ROBOT_NAME not in newton_robot_policy.ROBOT_CONFIGS:
        mophi.fatal(f"Unknown Newton policy robot '{ROBOT_NAME}'.")

    mophi.check_newton_warp_mujoco_versions(
        mophi.REQUIRED_NEWTON_VERSION,
        mophi.REQUIRED_WARP_VERSION,
        mophi.REQUIRED_MUJOCO_VERSION,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    wp.init()

    robot_config = newton_robot_policy.ROBOT_CONFIGS[ROBOT_NAME]
    print(f"[Assets] Downloading or locating public Newton assets for '{ROBOT_NAME}' ...")
    try:
        asset_directory = Path(newton.utils.download_asset(robot_config.asset_dir))
    except RuntimeError as exc:
        mophi.fatal(
            "Newton could not download the Unitree G1 assets. Check GitHub connectivity, "
            "then retry with a fresh cache directory:\n"
            "  NEWTON_CACHE_PATH=$HOME/.cache/newton-mophi "
            "python -c \"import newton.utils; print(newton.utils.download_asset('unitree_g1'))\"\n"
            f"Newton error: {exc}"
        )
    print(f"[Assets] Ready at '{asset_directory}'.")

    yaml_path = asset_directory / robot_config.yaml_path
    if not yaml_path.is_file():
        mophi.fatal(f"Downloaded robot configuration was not found: {yaml_path}")
    with yaml_path.open(encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    policy_path = asset_directory / robot_config.policy_path["mjw"]
    if not policy_path.is_file():
        mophi.fatal(f"Downloaded robot policy was not found: {policy_path}")
    model_path = asset_directory / robot_config.asset_path
    if not model_path.is_file():
        mophi.fatal(
            "The Newton Unitree G1 asset cache is incomplete. The robot model "
            f"was not found:\n  {model_path}\n"
            "Download into a fresh cache directory, then run this demo with the "
            "same NEWTON_CACHE_PATH value:\n"
            "  NEWTON_CACHE_PATH=$HOME/.cache/newton-mophi "
            "python -c \"import newton.utils; print(newton.utils.download_asset('unitree_g1'))\""
        )

    dof_indices = list(range(config["num_dofs"]))
    viewer = newton.viewer.ViewerGL()
    example = newton_robot_policy.Example(
        viewer,
        robot_config,
        config,
        str(asset_directory),
        dof_indices,
        dof_indices,
    )
    newton_robot_policy.load_policy_and_setup_tensors(
        example,
        str(policy_path),
        config["num_dofs"],
        slice(7, None),
    )
    if not np.isclose(example.frame_dt, NEWTON_DT, rtol=0.0, atol=1.0e-12):
        mophi.fatal(f"Newton G1 policy timestep changed: expected {NEWTON_DT}, got {example.frame_dt}.")

    if hasattr(viewer, "set_camera"):
        viewer.set_camera(wp.vec3(*CAMERA_POSITION), CAMERA_PITCH, CAMERA_YAW)

    movie_writer = None
    if SAVE_MOVIE:
        movie_writer = imageio.get_writer(MOVIE_OUTPUT_PATH, fps=MOVIE_FPS)
        print(f"[Movie] Recording simulation to '{MOVIE_OUTPUT_PATH}' at {MOVIE_FPS} fps.")

    print(
        f"Running up to {NUM_FRAMES} interactive Unitree G1 frame(s) "
        f"({SIM_SUBSTEPS} Newton policy steps x {NEWTON_DT * 1000.0:.1f} ms per rendered frame) ..."
    )

    completed_frames = 0
    try:
        for frame in range(NUM_FRAMES):
            if not viewer.is_running():
                print(f"\n[Viewer] Window closed by user after frame {frame}.")
                break
            for _ in range(SIM_SUBSTEPS):
                example.step()
            example.render()
            completed_frames = frame + 1
            if movie_writer is not None:
                movie_writer.append_data(viewer.get_frame().numpy())
            if completed_frames % RENDER_FPS == 0 or completed_frames == 1:
                print(f"  frame {completed_frames:>4}/{NUM_FRAMES}  sim_time={example.sim_time:.2f} s")
    except KeyboardInterrupt:
        print(f"\n[Input] Stopped by user after frame {completed_frames}.")
    finally:
        if movie_writer is not None:
            movie_writer.close()
        viewer.close()

    if SAVE_FINAL_STATE:
        np.savez_compressed(
            FINAL_STATE_OUTPUT_PATH,
            body_q=example.state_0.body_q.numpy(),
            joint_q=example.state_0.joint_q.numpy(),
        )
        print(f"[Output] Saved final state to '{FINAL_STATE_OUTPUT_PATH}'.")

    _write_run_metadata(completed_frames, asset_directory)
    print(f"[Output] Saved run metadata to '{RUN_METADATA_OUTPUT_PATH}'.")
    if SAVE_MOVIE:
        print(f"[Movie] Saved simulation recording to '{MOVIE_OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
