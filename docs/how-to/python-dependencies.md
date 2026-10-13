# Python dependencies

From a checkout, with Python 3.11 or newer, install the common runtime set in one command:

```bash
python scripts/setup/install_dependencies.py
```

The [installer](../../scripts/setup/install_dependencies.py) reads
[`pyproject.toml`](../../pyproject.toml) and invokes the active interpreter's pip.
It installs dependencies only; it does not configure CMake, build MoPhi, fetch
licensed assets, or install a compiler/CUDA toolkit. Use `--dry-run` to inspect
the exact command without changing your environment. Pip failures terminate the script.

The common set includes Newton/Warp/MuJoCo, DEME, XLB, PyTorch, asset loaders,
and visualization/movie packages. It is a convenient superset for the demos,
not a minimal dependency selection for each coupler. Version requirements live
in package metadata; avoid copying a separate pinned list into demo instructions.
The installer uses the distribution's default providers; custom solver providers
must be installed separately as described in [provider configuration](../couplers/newton-xlb-deme.md#provider-configuration).

## Optional demo dependencies

GRAB preparation, geometry diagnostics, and rendering need an additional set:

```bash
python scripts/setup/install_dependencies.py --extra grab
```

This includes the common set and the `grab` extra declared in package metadata.
It does not download GRAB or SMPL-X assets; follow the [GRAB workflow](grab.md).
FERIS additionally requires its native solver and build prerequisites; see
[FERIS + Newton](../couplers/feris-newton.md). Other demo-specific assets and hardware
requirements are listed in their guides and the [demo matrix](../how-to.md#demo-build-requirements).

Installing a MoPhi wheel resolves the common dependencies automatically. For an
installed distribution with GRAB tools, use `python -m pip install 'mophi[grab]'`.
Building a wheel and installing its runtime dependencies are separate operations;
see [Python build workflows](python-builds.md).
