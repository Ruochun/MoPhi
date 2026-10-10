"""Shared configuration for the GRAB cylinder-grasp simulation and viewer."""

from pathlib import Path

# CUP_* names identify the manipulated object in the shared output/configuration schema.
# ── Configuration ────────────────────────────────────────────────────────────
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OBJECT_PATH = REPOSITORY_ROOT / "data/grab/s2/cylindermedium_lift_frames_000360_000510_object.npz"
HAND_PATH = REPOSITORY_ROOT / "data/grab/s2/cylindermedium_lift_frames_000360_000510_right.npz"
OBJECT_LABEL = "cylinder"
CASE = "grab"
CUP_PATCH_MODE = "spread"
CUP_PATCH_COUNT = 10  # Used by the spread mode.
HAND_PATCH_MODE = "spread"
HAND_PATCH_COUNT = 10  # Assigned on the initial hand shape; retained during deformation.
# Radius and normal limits apply only to connected mode.
CUP_PATCH_RADIUS = 0.013
HAND_PATCH_RADIUS = 0.0085
PATCH_MAX_NORMAL_ANGLE = 45.0
CUP_PATCH_SECTORS = 8  # Used only by the optional sectors mode.
REST_MAX_ANGULAR_DRIFT_DEG = 1.0
REST_MAX_ANGULAR_SPEED = 0.02
CONTACT_OUTPUT_FORCE_THRESHOLD = 1e-20
GRAB_START_FRAME = 360
GRAB_STOP_FRAME = 490
PLAYBACK_SPEED = 0.5
GRAB_DIAGNOSTIC_INTERVAL = 0.05
MIN_GRAB_LIFT = 0.01
MAX_GRAB_REFERENCE_ERROR = 0.05
GRAB_DETECTION_SPEED_FACTOR = 1.25
GRAB_TRANSFER_TOLERANCE = 1e-05
GRAB_SUPPORT_FORCE_FRACTION = 0.5
GRAB_LIFT_HOLD_INTERVAL = 0.1
CUDA_DEVICE = 0
DEME_DT = 0.0001
DURATION = 0.8  # Cup-rest only; grasp duration follows the recording interval.
RENDER_FPS = 60.0
PROGRESS_INTERVAL = 0.1
SETTLE_TIME = 0.25
CUP_MASS = 0.25
# A compact proxy keeps universal mesh contact affordable; error gates check fidelity.
COLLISION_FACE_COUNT = 600
GEOMETRY_CHECK_SAMPLES = 2000
GEOMETRY_CHECK_SEED = 0
MAX_PROXY_SURFACE_ERROR = 0.0015
MAX_PROXY_VOLUME_ERROR = 0.03
GRAVITY = 9.81
REST_GAP = 0.0005
CONTACT_YOUNG_MODULUS = 1000000.0
CONTACT_POISSON_RATIO = 0.3
CONTACT_RESTITUTION = 0.1
CONTACT_FRICTION = 0.5
FLOOR_FRICTION = 0.1
FIXED_OWNER_MASS = 1000000.0
FIXED_OWNER_MOI = (1.0, 1.0, 1.0)
CUP_FAMILY = 0
MOVING_FAMILY = 1
FLOOR_FAMILY = 2
DOMAIN_X = (-0.7, 0.7)
DOMAIN_Y = (-0.6, 0.6)
DOMAIN_Z = (-0.2, 0.8)
BIN_SIZE = 0.02
COMPACT_REST_DOMAIN = False
REST_DOMAIN_X = (-0.15, 0.15)
REST_DOMAIN_Y = (-0.15, 0.15)
REST_DOMAIN_Z = (-0.05, 0.25)
MAX_VELOCITY = 5.0
DETECTION_SAFETY_SPEED = 0.2
FLOOR_VISUAL_EXTENTS = (0.7, 0.5, 0.01)
HEADLESS = False
RENDER = True
SAVE_MOVIE = False
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_POSITION = (0.48, -0.65, 0.38)
CAMERA_TARGET = (0.0, 0.0, 0.07)
AXIS_ORIGIN = (-0.25, 0.18, 0.01)
AXIS_LENGTH = 0.08
SCALE_BAR_CENTER = (0.0, 0.3, 0.01)
SCALE_MARKER_RADIUS = 0.007
MESH_COLORS = {"cup": (0.65, 0.7, 0.75), "surface": (0.91, 0.55, 0.19), "floor": (0.35, 0.4, 0.45)}
VIEWER_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output/render_cylinder_comparison"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output/demo_grab_cylinder"
# Acceptance tolerance for the intentionally soft contact material.
MAX_ALLOWED_PENETRATION = 0.020
MAX_WEIGHT_BALANCE_RELATIVE_ERROR = 0.15
