# Particle-spacing refinement study

Why: in the shipped configuration the ViTacTip FEM particles (green) **ghost into**
the MPM phantom particles (blue) — the two bodies interpenetrate instead of the
sensor deforming against the phantom surface. The fix is to refine the "atom"
spacing of both bodies so the contact stencil can resolve the interface.

## Result

**Multiplication factor 0.5 produced the desired deformation.** At 0.5 the two
bodies are cleanly separated with a distinct interface; the baseline shows green
particles buried inside the blue block. Compare, both at t=40 s:

| | |
|---|---|
| `iteration_baseline_1.0/shot_08_t040s.png` | ghosting — sensor sunk into the phantom |
| `iteration_0.5/shot_08_t040s.png` | clean contact interface, no interpenetration |

Factors 0.25 and 0.125 were **not run** — see "Why the finer factors were not run".

## What each knob was set to

Everything is driven from the single-source-of-truth files by `set_spacing.py`;
generated files (`system-params.json` distances, `system-params-computed.json`)
are produced from them by `script_apply_scaling` / `script_pre_main`.

| Knob | Baseline | factor 0.5 |
|---|---|---|
| `phantom.mpm_grid_cube_size` (SI) | 0.002 m | 0.001 m |
| `phantom.particle_spacing` (SI) | 0.001 m | 0.0005 m |
| `rigid_static.particle_spacing` (SI) | 0.001 m | 0.0005 m |
| `gmsh_mm.vitactip.characteristic_length_factor` | 0.30 | 0.15 |
| `gmsh_mm.vein.characteristic_length_factor` | 0.30 | 0.15 |
| `contact.dt_override` | 1e-5 s | 5e-6 s |

`particle_spacing` is always `mpm_grid_cube_size / 2` (8 particles per cell in
3D), and the two `characteristic_length_factor` values are always kept equal.
A *smaller* Gmsh factor means a *finer* mesh, so it scales the same way as the
distances.

## The timestep

`contact.dt_override` is the live knob. `Contact.set_dt()`
(`difftactile/main/main.py:2426`) reads it directly; the
`calculate_cfl_timestep(...)` call right above it is commented out, so nothing
auto-computes the timestep. The sibling key `contact.dt` is dead — no code reads it.

It is scaled **linearly** with the spacing. CFL gives `dt ≲ C·Δx/c`, and the wave
speed `c = sqrt((λ+2μ)/ρ)` depends only on material properties, which this study
does not touch. So halving the spacing halves the stable timestep.

## A geometry bug this uncovered

Refining the grid initially moved the phantom out from under the sensor, so the
press-and-slide never made contact. Cause: in `pre_main.py` the phantom's world
origin was derived purely from grid *node indices* times `d`. The zero-velocity
boundary band is a fixed **4 nodes** wide, so its physical extent `4·d` halved on
every refinement, translating the phantom (baseline origin `[0.095, 0.095, 0.045]`
collapsed to `[0.061, 0.061, 0.011]` at factor 0.25). The sensor start pose is
built from `phantom_closest_vertex` minus a *fixed* `sensor_xy_radius`, so the gap
between them grew and the trajectory fell short.

Fix (`pre_main.py`): keep the boundary band at 4 nodes — that requirement is
genuinely measured in cells — but anchor the phantom's world placement to a
reference grid spacing via `_origin_offset()`, absorbing the difference into the
padding in x/y and adding it explicitly in z. The origin now stays within a
quarter-cell of baseline at every factor tested.

## Why the finer factors were not run

Refining by 2× in 3D gives 8× the particles and 8× the grid cells. The MPM state
is stored as `(num_sub_frames, num_particles)` fields with `num_sub_frames = 125`,
so at factor 0.25 the phantom reaches 682,344 particles and those fields alone
need **~15.4 GB**, plus ~2.4 GB of grid fields — against a 10 GB RTX 3080. Both
0.25 and 0.125 died with `CUDA_ERROR_OUT_OF_MEMORY` during field allocation, before
any timestep ran. Factor 0.125 would need roughly 8× more again.

Running them against host RAM on the CPU arch was considered and deliberately
rejected. Factor 0.5 already removes the ghosting, so the finer levels were not
pursued.

## Files

- `set_spacing.py` — scales every spacing knob (and `dt_override`) by one factor.
- `run_one_iteration.sh` — runs one already-configured iteration and screenshots it.
- `run_spacing_sweep.sh` — the full sweep driver (configure → re-mesh → run → capture).
- `iteration_<f>/` — screenshots (`shot_NN_tNNNs.png`, every 5 s), `sim.log`, and
  the per-step regeneration logs for that factor. **Not tracked in git** (see
  `.gitignore`) — these are run output; re-create them with the commands below.

## Reproducing

taichi lives only in the `difftactile` container, and the GGUI window has to
actually render for screenshots to exist, so the container draws into a
**host-side Xvfb** virtual display which `ffmpeg -f x11grab` captures. A virtual
display is used rather than the real one so the run neither disturbs nor captures
the user's desktop.

```bash
./docker/docker-run.sh
Xvfb :95 -screen 0 1280x1024x24 -nolisten tcp &
python3 difftactile/manual_or_experimental_data/particle_spacing_study/set_spacing.py 0.5
docker exec -e DIFFTACTILE_HEADLESS=1 difftactile bash -lc 'cd /workspace/shallow-vessel-palpation-simulator-and-AI \
  && python -m difftactile.scripts.script_apply_scaling \
  && python -m difftactile.scripts.script_generate_vitactip_mesh_gmsh \
  && python -m difftactile.scripts.script_generate_vein_mesh_gmsh \
  && python -m difftactile.scripts.script_pre_main'
./difftactile/manual_or_experimental_data/particle_spacing_study/run_one_iteration.sh 0.5 60 9
```

The `characteristic_length_factor` change only takes effect once the Gmsh meshes
are rebuilt, hence the two `generate_*_mesh_gmsh` steps.
