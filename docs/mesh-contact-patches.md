# Mesh contact patches

[`python/mophi/utils/geometry/mesh_contact_patches.py`](../python/mophi/utils/geometry/mesh_contact_patches.py)
provides two initialization-time, solver-independent NumPy utilities:
`partition_mesh_contact_patches` uses radius/normal limits, and
`partition_mesh_seeded_patches` takes a target count. Both assign one integer ID
per triangle without changing coordinates, topology, or triangle ordering.
Neither implementation depends on DEME or Newton.

```python
from mophi.utils import partition_mesh_contact_patches

ids = partition_mesh_contact_patches(vertices, faces, radius=0.013,
                                     max_normal_angle_deg=45.0)
# Optional DEME use:
deme_mesh.SetPatchIDs(ids.tolist())
```

## Algorithm and contract

1. Compute triangle centroids and unit normals; build edge adjacency.
2. Visit faces in index order. Each unassigned face seeds a patch.
3. Grow a region by breadth-first search across edges shared by exactly two faces.
4. Accept an unassigned neighbor if all its vertices lie within `radius` of the
   seed centroid and its normal lies within `max_normal_angle_deg` of the seed normal.
5. Continue with the next unassigned seed; return contiguous zero-based int32 IDs.

Inputs are finite `(N, 3)` vertices and nonempty integer `(M, 3)` face indices.
Radius must be positive and finite, in the coordinate units; angle is in
`[0, 90)` degrees. Invalid indices and degenerate triangles raise `ValueError`.
Input arrays are not mutated. Boundaries and nonmanifold edges are not crossed,
and coincident vertices are not welded. An oversized seed remains a singleton.

The greedy result is deterministic for a fixed input ordering, but it does not
optimize count, patch uniformity, or area. Increasing radius generally reduces
count but is not a guarantee of monotonic counts for every mesh. The routine
neither remeshes nor merges small leftover patches. Triangle reordering may
change the partition.

For deformation, partition a reference shape once and retain IDs during vertex
updates. Geometric bounds then apply only to the reference shape. The caller
owns reference-pose choice, thresholds, physical coefficients, contact law, and
validation. Patches do not split a mesh into independent bodies. Changing their
number can alter a solver's area-dependent stiffness/damping and friction
history, so this operation alone cannot guarantee correct contact physics.

The original import from `mophi.couplers.newton_deme` remains available and
refers to the same implementation. The [GRAB workflow](how-to/grab.md)
shows the DEME integration, saved partitions, diagnostics and visualization.

## Specifying the patch count

```python
from mophi.utils import partition_mesh_seeded_patches

ids = partition_mesh_seeded_patches(vertices, faces, patch_count=10)
deme_mesh.SetPatchIDs(ids.tolist())
```

This alternative shares geometry validation and manifold-edge adjacency with
`partition_mesh_contact_patches`. It selects the first seed farthest from the
area-weighted centroid, then spreads further seeds by shortest-path distance
between adjacent face centroids. Multi-source Dijkstra growth assigns each face
to its nearest seed, returning exactly the requested number of nonempty,
connected regions with contiguous int32 IDs. Ties are deterministic for fixed
input ordering. Inputs are unchanged; no solver dependencies are introduced.

The count must be an integer between the number of edge-connected components
and the number of faces; invalid counts raise `ValueError`. This mode imposes
no radius or normal-angle bound and guarantees neither convexity, equal area,
nor an evenly divided support footprint. It runs once during preparation;
retain its IDs during deformation. Contact stability still requires validation.

## Tests

```bash
PYTHONPATH=python python -m unittest discover -s tests -p test_grab_contact_patches.py -v
```

Tests cover determinism, connected regions and radius bounds, separated/opposing
faces, oversized seeds, unchanged inputs, and rejection of invalid geometry.
Count-based tests additionally check exact counts (including ten), connectivity,
determinism, disconnected components, and invalid counts.
