#!/usr/bin/env python3
"""Validate required Newton/Warp/MuJoCo Python package versions."""

import argparse
import importlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--newton-version", required=True)
    parser.add_argument("--warp-version", required=True)
    parser.add_argument("--mujoco-version", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    required = {
        "newton": args.newton_version,
        "warp": args.warp_version,
        "mujoco": args.mujoco_version,
    }
    errors = []
    for module_name, version in required.items():
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            errors.append(f"{module_name}: missing (required {version})")
            continue

        found = getattr(mod, "__version__", None)
        found_text = found if found is not None else "unknown"
        if found != version:
            errors.append(f"{module_name}: found {found_text} (required {version})")

    if errors:
        if not args.quiet:
            details = "\n".join(errors)
            sys.stderr.write(
                "MoPhi: Newton-based couplers require specific Python dependency versions\n"
                f"Required:\n"
                f"  - newton=={args.newton_version}\n"
                f"  - warp-lang=={args.warp_version}\n"
                f"  - mujoco=={args.mujoco_version}\n"
                f"Problems:\n{details}\n\n"
                "Install/fix with:\n"
                f"  python -m pip install --upgrade newton=={args.newton_version} "
                f"warp-lang=={args.warp_version} mujoco=={args.mujoco_version}\n"
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
