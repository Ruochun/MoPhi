"""Version-check helpers for optional Python dependencies used by MoPhi demos."""

from __future__ import annotations

import importlib
import warnings


def check_python_package_versions(
    required_versions: dict[str, str],
    *,
    package_names: dict[str, str] | None = None,
    strict: bool = False,
) -> bool:
    """Check installed package versions against required versions.

    By default, mismatches emit a warning and return ``False``.  Set ``strict=True``
    to raise ``RuntimeError`` instead.
    """
    package_names = package_names or {}
    mismatches: list[str] = []

    for module_name, required_version in required_versions.items():
        install_name = package_names.get(module_name, module_name)

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            mismatches.append(f"{install_name}: missing (required {required_version})")
            continue

        found_version = getattr(module, "__version__", None)
        found_text = found_version if found_version is not None else "unknown"
        if found_version != required_version:
            mismatches.append(
                f"{install_name}: found {found_text} (required {required_version})"
            )

    if not mismatches:
        return True

    message = "MoPhi dependency version mismatch:\n  - " + "\n  - ".join(mismatches)
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return False


def check_newton_warp_versions(
    required_newton_version: str,
    required_warp_version: str,
    *,
    strict: bool = False,
) -> bool:
    """Check Newton and Warp packages with warning-on-mismatch by default.

    Returns:
        True if both package versions match requirements, else False.
    """
    return check_python_package_versions(
        {"newton": required_newton_version, "warp": required_warp_version},
        package_names={"newton": "newton", "warp": "warp-lang"},
        strict=strict,
    )
