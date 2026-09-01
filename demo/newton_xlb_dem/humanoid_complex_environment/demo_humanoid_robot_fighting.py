"""Configurable Unitree G1 robot-fighting contact demo.

Two instances of the learned locomotion policy drive G1 robots toward each
other. Newton always manages articulated dynamics and ground contact. Select
either DEME proxy-mesh contact with wrench feedback or Newton's original coarse
robot-to-robot collision shapes.
"""

from pathlib import Path

import numpy as np

import demo_humanoid_complex_environment as humanoid_demo
from mophi.utils import load_package_provider

# =============================================================================
# Simulation configuration
# All user-tunable constants are defined here.
# =============================================================================

# -- Timing -------------------------------------------------------------------
RENDER_FPS = 50
NEWTON_DT = 1.0 / 200.0
POLICY_DECIMATION = 4
DEME_DT = 1.0 / 1000.0
SIM_DURATION_SECONDS = 5.0

# -- Robot --------------------------------------------------------------------
ROBOT_NAME = "g1_29dof"
ROBOT_INITIAL_POSES = [
    ((0.0, -1.5, 0.76), (0.0, 0.0, 0.7071, 0.7071)),
    ((0.0, 1.5, 0.76), (0.0, 0.0, -0.7071, 0.7071)),
]
FIGHTER_COMMANDS = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
ENABLE_SCRIPTED_FIGHT = True
FIGHT_APPROACH_END_TIME = 1.9
FIGHT_APPROACH_COMMANDS = FIGHTER_COMMANDS.copy()
FIGHT_HOLD_COMMANDS = np.zeros((2, 3), dtype=np.float32)
# Each entry is (start time, duration, attacking robot, attack kind, side).
FIGHT_ATTACK_SEQUENCE = (
    (2.05, 0.70, 0, "punch", "right"),
    (2.75, 0.70, 1, "punch", "left"),
    (3.45, 0.70, 0, "punch", "left"),
    (4.15, 0.70, 1, "punch", "right"),
)
FIGHT_FIST_CURL = 0.8
FIGHT_PUNCH_MOTION_SCALE = 1.35

# -- Contact ------------------------------------------------------------------
ROBOT_CONTACT_MODEL = "deme"  # Supported values: "deme", "newton".
MAX_CONTACT_COUNT = 4096
MAX_CONSTRAINT_COUNT = 8192

# -- Contact-proxy visualization ---------------------------------------------
SHOW_CONTACT_PROXY_MESHES = False
CONTACT_PROXY_MESH_PATH = Path(__file__).resolve().parents[3] / "data" / "mesh" / "cube.obj"
CONTACT_PROXY_PADDING = 0.005
CONTACT_PROXY_LINE_WIDTH = 0.006
CONTACT_PROXY_COLORS = ((0.1, 0.9, 1.0), (1.0, 0.45, 0.1))
DEME_ROBOT_FAMILIES = (10, 11)
DEME_DOMAIN_X = (-2.0, 2.0)
DEME_DOMAIN_Y = (-2.5, 2.5)
DEME_DOMAIN_Z = (-0.5, 2.5)
DEME_CONTACT_MATERIAL = {"E": 5.0e6, "nu": 0.3, "CoR": 0.1, "mu": 0.6, "Crr": 0.0}
DEME_PROXY_DIRECTORY_NAME = "deme_contact_proxies"

# -- Visualization ------------------------------------------------------------
USE_OMNIVERSE_VISUALIZATION = False
CAMERA_POSITION = (3.4, -3.7, 2.2)
CAMERA_PITCH = -12.0
CAMERA_YAW = 132.0
SCENE_SKY_UPPER_COLOR = (0.72, 0.78, 0.86)
SCENE_SKY_LOWER_COLOR = (0.42, 0.48, 0.56)
SCENE_LIGHT_COLOR = (2.0, 2.05, 2.1)
REFERENCE_AXIS_ORIGIN = (1.2, -2.1, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.2, -2.1, 0.06)

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_FPS = RENDER_FPS
SAVE_FINAL_STATE = True

# -- Derived constants --------------------------------------------------------
FRAME_DT = 1.0 / RENDER_FPS
POLICY_DT = POLICY_DECIMATION * NEWTON_DT
NUM_FRAMES = int(round(SIM_DURATION_SECONDS / FRAME_DT))


def _configure_shared_demo() -> None:
    """Apply fighting-specific configuration to the shared G1 harness."""
    if ROBOT_CONTACT_MODEL not in ("deme", "newton"):
        raise ValueError(f"ROBOT_CONTACT_MODEL must be 'deme' or 'newton', got {ROBOT_CONTACT_MODEL!r}.")
    deme_module = load_package_provider("deme") if ROBOT_CONTACT_MODEL == "deme" else None

    humanoid_demo.RENDER_FPS = RENDER_FPS
    humanoid_demo.NEWTON_DT = NEWTON_DT
    humanoid_demo.POLICY_DECIMATION = POLICY_DECIMATION
    humanoid_demo.NUM_FRAMES = NUM_FRAMES
    humanoid_demo.FRAME_DT = 1.0 / RENDER_FPS
    humanoid_demo.POLICY_DT = POLICY_DECIMATION * NEWTON_DT
    humanoid_demo.SIM_SUBSTEPS = int(round(humanoid_demo.FRAME_DT / humanoid_demo.POLICY_DT))
    humanoid_demo.DEME_DT = DEME_DT
    humanoid_demo.DEME_SUBSTEPS = int(round(NEWTON_DT / DEME_DT))
    humanoid_demo.ROBOT_NAME = ROBOT_NAME
    humanoid_demo.ROBOT_INITIAL_POSES = ROBOT_INITIAL_POSES
    humanoid_demo.DEFAULT_WALK_COMMANDS = FIGHTER_COMMANDS
    humanoid_demo.ENABLE_SCRIPTED_FIGHT = ENABLE_SCRIPTED_FIGHT
    humanoid_demo.FIGHT_APPROACH_END_TIME = FIGHT_APPROACH_END_TIME
    humanoid_demo.FIGHT_APPROACH_COMMANDS = FIGHT_APPROACH_COMMANDS
    humanoid_demo.FIGHT_HOLD_COMMANDS = FIGHT_HOLD_COMMANDS
    humanoid_demo.FIGHT_ATTACK_SEQUENCE = FIGHT_ATTACK_SEQUENCE
    humanoid_demo.FIGHT_FIST_CURL = FIGHT_FIST_CURL
    humanoid_demo.FIGHT_PUNCH_MOTION_SCALE = FIGHT_PUNCH_MOTION_SCALE

    # Newton always keeps its policy colliders for ground contact. Its
    # cross-robot pairs are filtered only when DEME supplies those forces.
    humanoid_demo.USE_ROBOT_VISUAL_MESH_COLLIDERS = False
    humanoid_demo.ENABLE_DEME_ROBOT_CONTACT = ROBOT_CONTACT_MODEL == "deme"
    humanoid_demo.DEME_MODULE = deme_module
    humanoid_demo.DEME_ROBOT_FAMILIES = DEME_ROBOT_FAMILIES
    humanoid_demo.DEME_DOMAIN_X = DEME_DOMAIN_X
    humanoid_demo.DEME_DOMAIN_Y = DEME_DOMAIN_Y
    humanoid_demo.DEME_DOMAIN_Z = DEME_DOMAIN_Z
    humanoid_demo.DEME_CONTACT_MATERIAL = DEME_CONTACT_MATERIAL
    humanoid_demo.DEME_PROXY_DIRECTORY_NAME = DEME_PROXY_DIRECTORY_NAME
    humanoid_demo.ENABLE_DYNAMIC_CONTACT_BOXES = False
    humanoid_demo.BOX_POSES = []
    humanoid_demo.ENABLE_INTERACTIVE_OBJECT_SPAWNING = False
    humanoid_demo.MAX_CONTACT_COUNT = MAX_CONTACT_COUNT
    humanoid_demo.MAX_CONSTRAINT_COUNT = MAX_CONSTRAINT_COUNT
    humanoid_demo.SHOW_CONTACT_PROXY_MESHES = SHOW_CONTACT_PROXY_MESHES
    humanoid_demo.CONTACT_PROXY_MESH_PATH = CONTACT_PROXY_MESH_PATH
    humanoid_demo.CONTACT_PROXY_PADDING = CONTACT_PROXY_PADDING
    humanoid_demo.CONTACT_PROXY_LINE_WIDTH = CONTACT_PROXY_LINE_WIDTH
    humanoid_demo.CONTACT_PROXY_COLORS = CONTACT_PROXY_COLORS

    humanoid_demo.USE_OMNIVERSE_VISUALIZATION = USE_OMNIVERSE_VISUALIZATION
    humanoid_demo.CAMERA_POSITION = CAMERA_POSITION
    humanoid_demo.CAMERA_PITCH = CAMERA_PITCH
    humanoid_demo.CAMERA_YAW = CAMERA_YAW
    humanoid_demo.SCENE_SKY_UPPER_COLOR = SCENE_SKY_UPPER_COLOR
    humanoid_demo.SCENE_SKY_LOWER_COLOR = SCENE_SKY_LOWER_COLOR
    humanoid_demo.SCENE_LIGHT_COLOR = SCENE_LIGHT_COLOR
    humanoid_demo.REFERENCE_AXIS_ORIGIN = REFERENCE_AXIS_ORIGIN
    humanoid_demo.REFERENCE_SCALE_BAR_CENTER = REFERENCE_SCALE_BAR_CENTER
    humanoid_demo.SHOW_WAREHOUSE_ENVIRONMENT = False

    humanoid_demo.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    humanoid_demo.SAVE_MOVIE = SAVE_MOVIE
    humanoid_demo.MOVIE_OUTPUT_PATH = OUTPUT_DIRECTORY / "demo_humanoid_robot_fighting.mp4"
    humanoid_demo.MOVIE_FPS = MOVIE_FPS
    humanoid_demo.SAVE_FINAL_STATE = SAVE_FINAL_STATE
    humanoid_demo.FINAL_STATE_OUTPUT_PATH = OUTPUT_DIRECTORY / "final_state.npz"
    humanoid_demo.RUN_METADATA_OUTPUT_PATH = OUTPUT_DIRECTORY / "run_metadata.json"
    humanoid_demo.VISUAL_MESH_MANIFEST_OUTPUT_PATH = OUTPUT_DIRECTORY / "visual_mesh_manifest.json"


def main() -> None:
    """Run two opposing G1 policies with the configured contact model."""
    _configure_shared_demo()
    humanoid_demo.main()


if __name__ == "__main__":
    main()
