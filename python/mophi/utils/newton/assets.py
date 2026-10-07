"""Validated, self-repairing downloads for Newton example assets."""

from collections.abc import Iterable
from pathlib import Path, PurePosixPath


def _normalize_required_paths(required_paths: Iterable[str | Path]) -> tuple[Path, ...]:
    normalized = []
    for required_path in required_paths:
        posix_path = PurePosixPath(str(required_path))
        if posix_path.is_absolute() or ".." in posix_path.parts or not posix_path.parts:
            raise ValueError(f"Newton asset requirements must be relative paths: {required_path!s}")
        normalized.append(Path(*posix_path.parts))
    if not normalized:
        raise ValueError("At least one required Newton asset path must be provided.")
    return tuple(normalized)


def _missing_files(asset_directory: Path, required_paths: tuple[Path, ...]) -> list[Path]:
    return [relative_path for relative_path in required_paths if not (asset_directory / relative_path).is_file()]


def download_newton_asset(asset_folder: str, required_paths: Iterable[str | Path]) -> Path:
    """Download a Newton asset folder and repair an incomplete sparse-checkout cache.

    Newton 1.0 may accept a cached Git checkout based on its directory and
    repository metadata even when an interrupted sparse checkout omitted files.
    This helper validates files required by the caller and uses Newton's public
    ``force_refresh`` path once when the cache is incomplete.
    """
    from newton.utils import download_asset

    requirements = _normalize_required_paths(required_paths)
    asset_directory = Path(download_asset(asset_folder))
    missing = _missing_files(asset_directory, requirements)
    if not missing:
        return asset_directory

    missing_text = ", ".join(path.as_posix() for path in missing)
    print(f"[Newton assets] Incomplete cache for '{asset_folder}' (missing: {missing_text}); refreshing it ...")
    try:
        asset_directory = Path(download_asset(asset_folder, force_refresh=True))
    except Exception as exc:
        raise RuntimeError(
            f"Could not refresh incomplete Newton asset '{asset_folder}'. "
            f"Missing before refresh: {missing_text}. Check network access and NEWTON_CACHE_PATH. "
            "As a last resort, set NEWTON_CACHE_PATH to a new writable directory and run the demo again."
        ) from exc

    missing = _missing_files(asset_directory, requirements)
    if missing:
        missing_text = ", ".join(path.as_posix() for path in missing)
        raise RuntimeError(
            f"Newton refreshed asset '{asset_folder}', but required files are still missing: {missing_text}. "
            "The installed Newton version and newton-assets revision may be incompatible. As a last resort, set "
            "NEWTON_CACHE_PATH to a new writable directory and run the demo again."
        )
    return asset_directory
