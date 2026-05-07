/// demo/tlfea_dem/demo_tlfea_dem.cpp
///
/// Demonstrates the full lifecycle of a TLFEADEMCoupler simulation:
///   1. Construct the coupler.
///   2. Initialize both solvers (TLFEA + DEM-Engine).
///   3. Advance the co-simulation for a fixed number of steps.
///   4. Finalize and release all resources.
///
/// This demo is intentionally minimal — its purpose is to verify that the
/// MoPhi build system correctly fetches, builds, and links both external
/// solvers so that the coupler can be instantiated and exercised.  When the
/// solvers gain real simulation setups (mesh, materials, particles), this
/// demo should be extended with those details.

#include <iostream>
#include <stdexcept>

#include "TLFEADEMCoupler.h"

int main() {
    std::cout << "=== MoPhi TLFEA+DEM-Engine co-simulation demo ===\n\n";

    // ── 1. Construct ─────────────────────────────────────────────────────────
    mophi::TLFEADEMCoupler coupler;
    coupler.SetVerbosity(mophi::VERBOSITY_INFO);

    // ── 2. Initialize ────────────────────────────────────────────────────────
    // tlfea_config and dem_config are left empty; real simulations would
    // supply paths to mesh / scene description files here.
    // num_gpus controls hardware parallelism; use the value appropriate for the machine.
    coupler.Initialize(/*tlfea_config=*/"",
                       /*dem_config=*/"",
                       /*num_gpus=*/1);

    // ── 3. Time-stepping loop ────────────────────────────────────────────────
    constexpr int num_steps = 5;
    std::cout << "\nRunning " << num_steps << " co-simulation step(s)...\n";
    for (int i = 0; i < num_steps; ++i) {
        std::cout << "  step " << (i + 1) << " / " << num_steps << "\n";
        coupler.Step();
    }

    // ── 4. Finalize ──────────────────────────────────────────────────────────
    coupler.Finalize();

    std::cout << "\nDemo completed successfully.\n";
    return 0;
}
