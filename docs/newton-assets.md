# Newton asset cache handling

This document describes `python/mophi/utils/newton_assets.py`, the shared
Newton example-asset download helper used by MoPhi demos.

## Purpose and boundary

`mophi.download_newton_asset(asset_folder, required_paths)` delegates downloads
to Newton's public `newton.utils.download_asset` API. It adds validation for the
specific relative files a demo needs. It does not maintain a second downloader,
choose Newton asset revisions, or modify assets.

Newton 1.0 can consider a cached sparse Git checkout valid when its directory
and repository metadata exist, even if an interrupted checkout left required
files absent. When validation finds missing files, MoPhi invokes Newton's
`force_refresh=True` path once and validates the replacement checkout again.

## Usage

```python
asset_directory = mophi.download_newton_asset(
    "unitree_g1",
    ["usd/g1_isaac.usd", "rl_policies/g1_29dof.yaml"],
)
```

Requirements must be relative paths contained by the downloaded asset folder.
Absolute paths and parent traversal are rejected. `NEWTON_CACHE_PATH` continues
to control the cache root because Newton itself performs both download calls.

## Supported behavior and limitations

- Complete caches are reused without a refresh.
- Incomplete caches are refreshed once through Newton's supported API.
- Download failures and files still absent after refresh raise `RuntimeError`.
- Repair requires network access. An offline incomplete cache cannot be fixed.
- A post-refresh missing file can indicate that the installed Newton version
  expects a different `newton-assets` layout; the helper reports this rather
  than retrying indefinitely.

## Last-resort fresh cache

Automatic force-refresh keeps Newton's configured cache root. If that root is
unwritable or corrupted, or automatic refresh still fails, use a new cache
directory as a **last resort**:

```bash
export NEWTON_CACHE_PATH="$HOME/.cache/newton-mophi-fresh"
PYTHONPATH=python python3 demo/newton_xlb_dem/humanoid_complex_environment/demo_humanoid_robot_fighting.py
```

Keep `NEWTON_CACHE_PATH` set for later runs that should reuse this fresh cache.
This fallback still requires network access and does not resolve an incompatible
Newton/newton-assets version pair.

## Tests

```bash
PYTHONPATH=python pytest -q tests/test_newton_assets.py
```
