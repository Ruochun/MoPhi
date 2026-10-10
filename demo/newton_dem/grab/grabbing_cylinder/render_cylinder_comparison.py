"""Run the grabbing cylinder comparison viewer."""

from pathlib import Path
import sys

# Resolve shared demo code in both source and installed demo trees.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grabbing_cylinder import scene_config
from grasp_viewer import main

if __name__ == "__main__":
    main(scene_config)
