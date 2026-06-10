# Humanoid complex-environment demo

This folder starts from Newton 1.0.0's public
[`example_robot_policy.py`][newton-policy]
Unitree G1 walking baseline. It provides an interactive keyboard-controlled
humanoid walking policy.

`demo_humanoid_complex_environment.py` keeps Newton's policy and simulation
implementation as the baseline, then adds MoPhi-style configuration,
a finite run loop, MP4 generation, run metadata, and a final-state snapshot.
Generated files are written to
`<repository-root>/output/demo_humanoid_complex_environment/`.

## Assets

No manual asset placement is required. On first run, Newton calls
`newton.utils.download_asset("unitree_g1")` and downloads the Unitree G1 model,
walking policy, and policy configuration from the public
[`newton-assets`](https://github.com/newton-physics/newton-assets) repository.
Newton prints the resulting cache directory as `[Assets] Ready at ...`.

Install the downloader and policy-configuration dependencies, then optionally
prefetch the assets:

```bash
pip install GitPython PyYAML
python -c "import newton.utils; print(newton.utils.download_asset('unitree_g1'))"
```

Newton uses `NEWTON_CACHE_PATH` when that environment variable is set.
Otherwise, it stores assets in the platform user cache, normally
`~/.cache/newton/` on Linux. Do not place assets inside the Python
`site-packages` directory.

If a download was interrupted and the cache is missing files such as
`usd/g1_isaac.usd`, download into a fresh cache directory:

```bash
export NEWTON_CACHE_PATH="$HOME/.cache/newton-mophi"
python -c "import newton.utils; print(newton.utils.download_asset('unitree_g1'))"
```

Keep `NEWTON_CACHE_PATH` set when running the demo.

The next stages will add complex terrain and objects spawned interactively
during simulation.

[newton-policy]: https://github.com/newton-physics/newton/blob/v1.0.0/newton/examples/robot/example_robot_policy.py
