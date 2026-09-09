#!/usr/bin/env python3
"""Run MoPhi's documented, actively developed demos serially."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

IMPORTANT_DEMOS = (
    "demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_multiphysics.py",
    "demo/newton_xlb_dem/anymal_robot_multiphysics/demo_anymal_robot_newton_baseline.py",
    "demo/newton_xlb_dem/excavation_comparison/demo_claw_newton.py",
    "demo/newton_xlb_dem/excavation_comparison/demo_claw_newton_cubes.py",
    "demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_complex_environment.py",
    "demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py",
    "demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_swimming.py",
    "demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_newton.py",
    "demo/newton_xlb_dem/flexible_hand_manipulation/demo_flexible_hand_deme.py",
    "demo/newton_dem/robotic_3d_printing/demo_robotic_3d_printing.py",
)

FERIS_DEMO = "demo/feris_newton/demo_feris_newton.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the important demos listed in docs/how-to.md, waiting for each demo to finish before starting the next."
        )
    )
    parser.add_argument(
        "--include-feris",
        action="store_true",
        help="also run the less-developed FERIS/Newton demo",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failed demo and report all failures at the end",
    )
    parser.add_argument("--list", action="store_true", help="print the selected demos without running them")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    demos = list(IMPORTANT_DEMOS)
    if args.include_feris:
        demos.append(FERIS_DEMO)
    if args.list:
        print("\n".join(demos))
        return 0

    environment = os.environ.copy()
    source_python = str(REPOSITORY_ROOT / "python")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_python, existing_pythonpath)) if existing_pythonpath else source_python
    )

    failures = []
    for index, relative_path in enumerate(demos, start=1):
        print(f"\n=== [{index}/{len(demos)}] {relative_path} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / relative_path)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )
        if result.returncode == 0:
            continue
        failures.append((relative_path, result.returncode))
        if not args.keep_going:
            break

    if failures:
        print("\nDemo failures:", file=sys.stderr)
        for relative_path, return_code in failures:
            print(f"  {relative_path}: exit status {return_code}", file=sys.stderr)
        return 1
    print(f"\nAll {len(demos)} demos completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
