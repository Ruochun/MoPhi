"""Install the checkout's declared runtime requirements without building MoPhi."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib


def main():
    metadata = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    project = metadata["project"]
    extras = project.get("optional-dependencies", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra", action="append", choices=sorted(extras), default=[], help="Also install a demo extra"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the pip command without installing packages")
    args = parser.parse_args()
    requirements = list(project["dependencies"])
    for extra in args.extra:
        requirements.extend(extras[extra])
    command = [sys.executable, "-m", "pip", "install", *dict.fromkeys(requirements)]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
