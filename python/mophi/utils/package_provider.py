"""Resolve replaceable providers for MoPhi's Python package dependencies.

Distribution names used by pip are independent from module names used by
Python.  This registry records both names for every dependency declared by
MoPhi and provides one uniform override mechanism for API-compatible
alternatives.
"""

from dataclasses import dataclass
import importlib
import os
from types import ModuleType


@dataclass(frozen=True)
class PackageProvider:
    """An installable distribution and the module through which it is used."""

    distribution: str
    import_module: str


# Keep these defaults aligned with pyproject.toml. Keys are stable logical
# dependency names; distribution and import names may change independently.
DEFAULT_PACKAGE_PROVIDERS = {
    "newton": PackageProvider("newton==1.0.0", "newton"),
    "warp": PackageProvider("warp-lang==1.12.1", "warp"),
    "mujoco": PackageProvider("mujoco==3.6.0", "mujoco"),
    "deme": PackageProvider("deme3>=3.0.9,<4", "deme"),
    "xlb": PackageProvider("xlb[cuda]", "xlb"),
    "torch": PackageProvider("torch", "torch"),
    "gitpython": PackageProvider("GitPython", "git"),
    "pyyaml": PackageProvider("PyYAML", "yaml"),
    "pycollada": PackageProvider("pycollada", "collada"),
    "mujoco_warp": PackageProvider("mujoco_warp==3.6.0", "mujoco_warp"),
    "pyglet": PackageProvider("pyglet", "pyglet"),
    "imageio": PackageProvider("imageio", "imageio"),
    "imageio_ffmpeg": PackageProvider("imageio-ffmpeg", "imageio_ffmpeg"),
}


def _environment_prefix(name: str) -> str:
    return f"MOPHI_PACKAGE_{name.upper()}"


def get_package_provider(name: str) -> PackageProvider:
    """Return provider configuration after applying environment overrides.

    For a logical name such as ``deme``, the supported variables are
    ``MOPHI_PACKAGE_DEME_DISTRIBUTION`` and
    ``MOPHI_PACKAGE_DEME_IMPORT_MODULE``.
    """

    try:
        default = DEFAULT_PACKAGE_PROVIDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DEFAULT_PACKAGE_PROVIDERS))
        raise ValueError(f"Unknown package provider '{name}'. Available providers: {available}") from exc

    prefix = _environment_prefix(name)
    return PackageProvider(
        distribution=os.environ.get(f"{prefix}_DISTRIBUTION", default.distribution),
        import_module=os.environ.get(f"{prefix}_IMPORT_MODULE", default.import_module),
    )


def load_package_provider(name: str) -> ModuleType:
    """Import and return the module configured for a logical dependency."""

    return importlib.import_module(get_package_provider(name).import_module)


__all__ = [
    "DEFAULT_PACKAGE_PROVIDERS",
    "PackageProvider",
    "get_package_provider",
    "load_package_provider",
]
