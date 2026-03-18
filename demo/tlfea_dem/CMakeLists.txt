# ─────────────────────────────────────────────────────────────────────────────
# demo/tlfea_dem/CMakeLists.txt
#
# Builds the demo executable that exercises TLFEADEMCoupler.
# ─────────────────────────────────────────────────────────────────────────────

add_executable(demo_tlfea_dem demo_tlfea_dem.cpp)

target_compile_features(demo_tlfea_dem PRIVATE cxx_std_17)

# mophi_couplers is the aggregate INTERFACE target; when MOPHI_BUILD_TLFEA_DEM
# is ON it transitively pulls in mophi_coupler_tlfea_dem (and through it the
# external solver headers and libraries).
target_link_libraries(demo_tlfea_dem PRIVATE mophi_couplers)
