# DEME contact getter comparison

`demo_deme_contact_getter_comparison.py` is a small Newton--DEME A/B test for
DEME's contact-acceleration and reduced-wrench getters. Two identical pairs of
free Newton boxes collide in isolated lanes. DEME owns both lanes' mesh contact;
Newton owns all rigid-body integration.

The lower, cyan lane obtains DEME contact and angular accelerations and forms a
wrench using the configured mass and body-frame principal MOI. The upper,
orange lane obtains DEME's directly reduced per-owner wrench. Poses, velocities,
getter results, and feedback wrenches move through caller-owned CUDA buffers.
Only periodic Newton motion snapshots are copied to the CPU for comparison and
output.

Each DEME proxy family uses no-expression prescribed pose and velocity. This
keeps every Newton-fed state unchanged during DEME substeps while allowing the
proxies to participate in contact and report reaction loads.

The default `ACCELERATION_FEEDBACK_MODE = "mass_moi"` performs the physically
consistent acceleration-to-wrench conversion. Set it to
`"assumed_unit_mass_moi"` to deliberately assume that contact acceleration is
force and angular acceleration is torque even when the configured body mass
and MOI are not unity.

Run from the repository root with a MoPhi build that enables
`MOPHI_BUILD_NEWTON_XLB_DEM` and with deme3 3.0.9 or newer:

```bash
PYTHONPATH=python python demo/newton_xlb_dem/deme_getter_comparison/demo_deme_contact_getter_comparison.py
```

The script writes mode-labelled `newton_motion_comparison_*.npz` and
`comparison_summary_*.json` files beneath
`output/demo_deme_contact_getter_comparison/`. The summary reports maximum and
final differences between corresponding Newton bodies in the two lanes.
