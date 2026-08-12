"""Scale every "atom"-spacing knob (and the timestep) by a single factor.

Writes the *source-of-truth* files only, never the generated ones:

  system-params-distances.json   phantom.mpm_grid_cube_size   (MPM grid, the main knob)
                                 phantom.particle_spacing     (always == grid/2, 8 PPC)
                                 rigid_static.particle_spacing(kept == phantom.particle_spacing)
  system-params.json             gmsh_mm.{vitactip,vein}.characteristic_length_factor
                                 contact.dt_override

Direction notes:
  * Gmsh's characteristic_length_factor *multiplies* the target element size, so
    a SMALLER factor => finer mesh => smaller FEM particle spacing. It therefore
    scales by the same factor as the distances.
  * The MPM particle spacing is derived, never independent: particle_spacing is
    forced to mpm_grid_cube_size / 2 (8 particles per cell in 3D).
  * dt is CFL-limited as dt ~ C * dx / c. The wave speed c depends only on
    material properties, which we are not touching, so dt scales LINEARLY with
    the spacing.

`apply_scaling.py` afterwards multiplies the -distances values by
meta.distance_scaling_factor to produce the sim-unit values in system-params.json,
so we only ever edit the SI sources here.

Usage:  python set_spacing.py 0.5
"""
import json
import sys
from pathlib import Path

# Baseline (unrefined) values, captured from the pristine repository state.
BASELINE = {
    "mpm_grid_cube_size": 0.002,   # m
    "characteristic_length_factor": 0.3,
    "dt_override": 1.0e-5,         # s
}

PARAMS_DIR = Path(__file__).resolve().parents[2] / "system_params"
DISTANCES = PARAMS_DIR / "system-params-distances.json"
MAIN = PARAMS_DIR / "system-params.json"


def main(factor: float) -> None:
    grid = BASELINE["mpm_grid_cube_size"] * factor
    # Standard MPM seeding: 2 particles per cell per axis => 8 PPC in 3D.
    pspacing = grid / 2.0
    clf = BASELINE["characteristic_length_factor"] * factor
    dt = BASELINE["dt_override"] * factor

    dist = json.loads(DISTANCES.read_text())
    dist["phantom"]["mpm_grid_cube_size"] = grid
    dist["phantom"]["particle_spacing"] = pspacing
    # rigid_static particles must stay in lockstep with the phantom's.
    dist["rigid_static"]["particle_spacing"] = pspacing
    DISTANCES.write_text(json.dumps(dist, indent=4) + "\n")

    main_params = json.loads(MAIN.read_text())
    # The two Gmsh factors are kept in sync at all times.
    main_params["gmsh_mm"]["vitactip"]["characteristic_length_factor"] = clf
    main_params["gmsh_mm"]["vein"]["characteristic_length_factor"] = clf
    # dt_override is the live timestep: main.py:2426 set_dt() reads it, and the
    # CFL auto-computation next to it is commented out.
    main_params["contact"]["dt_override"] = dt
    MAIN.write_text(json.dumps(main_params, indent=4) + "\n")

    print(f"factor                        = {factor}")
    print(f"phantom.mpm_grid_cube_size    = {grid:.6g} m")
    print(f"phantom.particle_spacing      = {pspacing:.6g} m  (= grid/2, 8 PPC)")
    print(f"rigid_static.particle_spacing = {pspacing:.6g} m")
    print(f"characteristic_length_factor  = {clf:.6g}  (vitactip and vein)")
    print(f"contact.dt_override           = {dt:.6g} s")


if __name__ == "__main__":
    main(float(sys.argv[1]))
