"""Newton-only Unitree G1 robot-fighting groundwork demo.

Two instances of the learned locomotion policy drive G1 robots toward each
other. Newton's policy-trained coarse collision shapes handle ground and
robot-to-robot contact. High-resolution visual mesh objects are retained and
validated, but deliberately do not participate in contact yet.
"""

from pathlib import Path

import numpy as np

import demo_humanoid_complex_environment as humanoid_demo

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

# -- Contact ------------------------------------------------------------------
MAX_CONTACT_COUNT = 4096
MAX_CONSTRAINT_COUNT = 8192

# -- Future DEME contact-proxy visualization ----------------------------------
SHOW_CONTACT_PROXY_MESHES = True
CONTACT_PROXY_MESH_PATH = Path(__file__).resolve().parents[3] / "data" / "mesh" / "cube.obj"
CONTACT_PROXY_PADDING = 0.005
CONTACT_PROXY_LINE_WIDTH = 0.006
CONTACT_PROXY_COLORS = ((0.1, 0.9, 1.0), (1.0, 0.45, 0.1))

# -- Visualization ------------------------------------------------------------
USE_OMNIVERSE_VISUALIZATION = False
CAMERA_POSITION = (4.8, -5.2, 2.7)
CAMERA_PITCH = -14.0
CAMERA_YAW = 132.0
REFERENCE_AXIS_ORIGIN = (1.2, -2.1, 0.02)
REFERENCE_SCALE_BAR_CENTER = (0.2, -2.1, 0.06)

# -- Output -------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / Path(__file__).stem
SAVE_MOVIE = True
MOVIE_FPS = RENDER_FPS
SAVE_FINAL_STATE = True


def _configure_shared_demo() -> None:
    """Apply fighting-specific configuration to the shared G1 harness."""
    humanoid_demo.RENDER_FPS = RENDER_FPS
    humanoid_demo.NEWTON_DT = NEWTON_DT
    humanoid_demo.NUM_FRAMES = NUM_FRAMES
    humanoid_demo.FRAME_DT = 1.0 / RENDER_FPS
    humanoid_demo.SIM_SUBSTEPS = int(round(humanoid_demo.FRAME_DT / NEWTON_DT))
    humanoid_demo.ROBOT_NAME = ROBOT_NAME
    humanoid_demo.ROBOT_INITIAL_POSES = ROBOT_INITIAL_POSES
    humanoid_demo.DEFAULT_WALK_COMMANDS = FIGHTER_COMMANDS

    # Keep visual meshes available as data, while Newton uses its original
    # coarse policy colliders for all contact in this stage.
    humanoid_demo.USE_ROBOT_VISUAL_MESH_COLLIDERS = False
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
    """Run two opposing G1 policies with coarse Newton contact."""
    _configure_shared_demo()
    humanoid_demo.main()


if __name__ == "__main__":
    main()
