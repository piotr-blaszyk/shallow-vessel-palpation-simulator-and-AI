import cProfile
import json
import math
import os
import pickle
import shutil
import time
from pathlib import Path
import time

import cv2
import matplotlib
from difftactile.main.display import finish_plot, is_headless
# Pick the non-interactive backend before pyplot is imported, so the loss plots
# below do not try to open a Tk window on a display-less machine.
if is_headless():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from difftactile.data_analysis.experiment.adjacency import *
from difftactile.data_analysis.experiment.bo_gp import *
from difftactile.main.apply_scaling import ScientificNotationEncoder
from difftactile.main.cfl_and_contact_params_estimation import *
from difftactile.main.constants import *
from difftactile.main.paths import repo_path
from difftactile.main.seeding import seed_everything
from difftactile.main.constants_bo_gp import *
from difftactile.main.synthetic_image_generator import *
from difftactile.object_model.phantom import Phantom
from difftactile.object_model.vein import Vein
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.sensor_model.vitactip import ViTacTip

RUN_ON_LAB_MACHINE = True

# -----------------------------------------------------------------------------
# Run-time overrides (environment variables).
#
# The published configuration lives in system-params.json and is the default for
# all of these; the env vars only exist so the same code can be run as a quick
# smoke test, or on a GPU with less memory, without editing tracked files.
#
#   DIFFTACTILE_NUM_LOOPS   number of outer training-data collection loops.
#                           Each loop runs 2 substeps x 4 trajectories = 8 trials.
#                           Default: contact.num_training_trajectories (100 -> 800
#                           trials, ~1 hour). Set to 1 for a ~1 minute smoke test.
#   DIFFTACTILE_HEADLESS    "1" to skip creating Taichi GGUI windows, so the
#                           simulator can run over SSH / in CI / in a container
#                           with no X server. Data collection is unaffected.
#   TI_DEVICE_MEMORY_GB     GPU memory budget handed to Taichi. Default 9, which
#                           suits a 10 GB card (e.g. RTX 3080).
# -----------------------------------------------------------------------------


def _env_int(name, default):
    """Read a positive integer env var, falling back to `default` if unset/invalid."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not an integer; using {default}")
        return default


HEADLESS = is_headless()

# Whether the first of the two substeps in each collection loop enables the
# sensor<->vein contact pair, so the loop sweeps the same trajectory once WITH a
# subsurface vein and once WITHOUT it. That pairing is the point of the training
# set (the GNN has to tell the two apart), but the vein half was hard-disabled
# (`if False and j < 1`), leaving every substep running without a vein. Defaults
# to 0 to preserve that committed behaviour; set to 1 to collect the pair.
COLLECT_VEIN_PAIR = os.environ.get("DIFFTACTILE_VEIN_PAIR", "0") == "1"

# Spacing between neighbouring markers on the undeformed sensor, in pixels at
# 1920x1080. This is the sensor's natural length scale: an error of one marker
# spacing means a marker has moved as far as the distance to its neighbour.
INTER_MARKER_SPACING_PX = 55.0

# Marker-distance conversion for reporting. The marker grid's ~55 px spacing is
# 2 mm on the real sensor, so a pixel error converts to millimetres by this
# factor. Used only for display - every stored MAE is in pixels.
PX_TO_MM = 2.0 / INTER_MARKER_SPACING_PX

# Score given to a parameter set whose FEM solve blows up (markers come back
# NaN). In the same pixel units as every other MAE, so no rescaling is involved -
# it simply has to sit far above any usable configuration, which in practice
# means anything over ~300 px.
DIVERGENCE_PENALTY_PX = 1000.0

# Per-trajectory timestep caps. These are safety bounds on the forward pass, not
# targets: the PID normally reaches its goal well before them, except on `slide`,
# where the cap is what stops the sensor sliding off the edge of the phantom.
#
# The two paths get DIFFERENT caps because they measure at different moments:
#
#   VESSEL-ABSENT scores at the trajectory's APEX, so the run must get there.
#     Back at 400 (it was briefly halved to 200): in the joint loop this run is
#     stopped at the VESSEL-PRESENT trigger timestep anyway, and that trigger has
#     been observed as late as ts=234 - a 200 cap would silently truncate the
#     comparison run before it reached the moment being compared.
#   VESSEL-PRESENT scores when the vessel passes under the sensor centre and
#     then SHORT-CIRCUITS, so its cap is only a backstop for the case where the
#     vessel never arrives. Left at 400 so that a late-arriving vessel is still
#     caught rather than silently scored 0.
DA_MAX_TIMESTEPS_NO_VEIN = 400
DA_MAX_TIMESTEPS_VEIN = 400

# How often to check that the sensor's FEM state is still physical. Every
# timestep would be wasteful (the check pulls the whole mesh back from the GPU);
# every 20 catches a divergence within a fraction of a trajectory, since once a
# solve blows up it never recovers.
HEALTH_CHECK_EVERY_TS = 20

# What a diverged run reports as its VESSEL-ABSENT mean marker error. Only used
# for the log and the record - a diverged iteration is scored -1 outright, so
# this value never enters the objective. 100 px sits well above the ~10-15 px of
# a usable configuration without distorting a plot.
DIVERGED_MEAN_MARKER_ERROR_PX = 100.0

# Ceiling for the fidelity term `van`: the vessel-absent mean marker error is
# clamped to [0, this] and divided by it, giving a value on [0, 1].
#
# NOT A GATE - nothing is rejected for exceeding it. It is the error at which
# the fidelity penalty saturates, so everything worse is treated as equally
# unfaithful.
#
# THREE inter-marker spacings, not one. At 55 px the term saturated too readily
# to be useful: every configuration with real vessel contact measured 96-160 px
# and so scored van = 1.0 identically, which flattened exactly the region the
# search needs to discriminate within. 165 px keeps those distinguishable while
# still treating a marker field displaced by three whole spacings as maximally
# unfaithful.
VAN_CLAMP_PX = 3.0 * INTER_MARKER_SPACING_PX

# Headroom above the vein's top in the penetration scale: a sensor sitting this
# far clear of the vein scores vpn = 0, and one pressed down to its rest-pose
# floor scores vpn = 1. 10 mm is the vein's own diameter, so the scale spans
# "one vessel-width clear" to "as deep as the sensor geometry allows".
# Weight on the penetration term of the joint objective. Both `vpn` and `van`
# therefore span [0, 1] and the objective ranges over [-1, +1], with a unit of
# vessel response worth exactly a unit of fidelity - no thumb on either scale.
VPN_WEIGHT = 1.0


def phantom_contact_enabled():
    """Is the sensor<->phantom contact pair (index 0) active?

    THE SEAM for pair 0. It is off by default: the phantom's particles are
    pinned, so it does not deform and resolving this pair changes nothing any
    objective measures - which is why its four coefficients were dropped from
    the BO search space.

    Turning it ON is a SANITY CHECK, not a modelling choice: if the sensor
    visibly deforms against the phantom when this is enabled, contact resolution
    works and a null vessel response has to be explained some other way. If it
    does not, the contact machinery itself is suspect.

    Reads `contact.enable_phantom_contact_pair` from system-params.json, with
    DIFFTACTILE_PHANTOM_CONTACT overriding it (1/0) so a script can flip it
    without editing the config.
    """
    override = os.environ.get("DIFFTACTILE_PHANTOM_CONTACT")
    if override is not None:
        return override == "1"
    return bool(getattr(
        SYSTEM_PARAMS.contact, "enable_phantom_contact_pair", False
    ))


def active_collision_pairs(with_vein):
    """The contact pairs to resolve: [0] if enabled, plus [2] when `with_vein`.

    Single source of truth for `collision_ixs`, so the pair-0 seam does not have
    to be re-checked at each of the several places a trajectory is set up.
    """
    pairs = [0] if phantom_contact_enabled() else []
    if with_vein:
        pairs.append(2)
    return pairs


class IterationLog:
    """Append-one-line-per-iteration CSV, flushed as the run proceeds.

    Written for watching a run LIVE - `tail -f` on the file shows each iteration
    as it completes, rather than after the whole search finishes. That matters
    because a 10-iteration run takes ~30 minutes and the JSON results are only
    written at the end, so a run that is stopped or crashes leaves nothing
    behind without this.

    Every write is flushed and fsync'd. Buffering would defeat the purpose: the
    lines would sit in the OS cache until the process exited, which is exactly
    when they are no longer needed.

    One file per run, named for the run's timestamp, all under the same folder
    (`difftactile/output/bo_logs/`) so runs can be compared without hunting
    through per-run directories.
    """

    def __init__(self, path, columns):
        self.path = path
        self.columns = list(columns)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write(",".join(self.columns) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append(self, row):
        """Write one iteration's values, in `columns` order. Missing -> empty."""
        cells = []
        for key in self.columns:
            value = row.get(key, "")
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        with open(self.path, "a") as f:
            f.write(",".join(cells) + "\n")
            f.flush()
            os.fsync(f.fileno())


def normalised_penetration(q_z, z_reference, z_vein_max):
    """Map a vessel-present depth onto [0, 1] against its vessel-free reference.

        vpn = VPN_WEIGHT * (clamp(q_z, lo, hi) - lo) / (hi - lo)
        lo  = z_reference     the vessel-FREE depth
        hi  = z_vein_max      the vessel's own apex height

    vpn = VPN_WEIGHT (its maximum) as soon as the sensor is held at the vessel's
    apex height OR ABOVE - the clamp saturates there, so being stopped early
    earns full credit and no more. vpn = 0 means the sensor sank all the way to
    where it would have gone with no vessel present, i.e. the vessel did nothing.

    The scale therefore runs from "the vessel had no effect" to "the vessel
    stopped the sensor at its surface", which is the full range of what a rigid
    inclusion can physically do. An earlier version extended `hi` 10 mm ABOVE the
    apex, which meant a sensor resting exactly on the vessel scored only ~0.4 of
    the available reward - the remaining 60% could only be earned by hovering
    above the vessel without touching it, which is not a better outcome.

    THE REFERENCE IS THE VESSEL-FREE DEPTH, NOT THE REST POSE. A soft sensor
    compresses under the phantom press and sits higher than its rest floor even
    with no vessel there; referencing the rest pose would read that as the vessel
    stopping it. Differencing against the vessel-free run at the SAME timestep
    removes that confound, leaving only what the vessel contributed.

    A pure function of three numbers, deliberately: both depths come from
    different simulation runs, and the mesh only ever holds one of them at a
    time, so taking them as arguments prevents measuring the wrong pose.

    Returns 0.0 for a non-finite input or a degenerate range - "no penetration
    demonstrated" rather than an exception or a spurious pass.
    """
    if not (np.isfinite(q_z) and np.isfinite(z_reference)):
        return 0.0
    lo = float(z_reference)
    hi = float(z_vein_max)
    if hi <= lo:
        return 0.0
    clamped = min(max(float(q_z), lo), hi)
    return float(VPN_WEIGHT * (clamped - lo) / (hi - lo))


def _trajectory_indices():
    """Which trajectory types training-data collection should execute.

    The four types are: 0 press (no vein), 1 slide (vein), 2 twist-y (no vein),
    3 twist-z (no vein) — see `Contact.trajectory_names`.

    DIFFTACTILE_TRAJECTORIES accepts a comma-separated list, e.g. "3" to
    reproduce the published dataset (which is entirely type 3) or "0,1,2,3".
    Defaults to all four, matching the current committed behaviour.
    """
    raw = os.environ.get("DIFFTACTILE_TRAJECTORIES")
    if not raw:
        return range(0, 4)
    try:
        ixs = [int(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        print(f"WARNING: DIFFTACTILE_TRAJECTORIES={raw!r} is not a list of ints; using 0-3")
        return range(0, 4)
    bad = [i for i in ixs if not 0 <= i < 4]
    if bad:
        print(f"WARNING: trajectory indices {bad} out of range 0-3; using 0-3")
        return range(0, 4)
    print(f"DIFFTACTILE_TRAJECTORIES={raw}: executing trajectory types {ixs}")
    return ixs


@ti.data_oriented
class Contact:
    def __init__(self):
        self.vein = Vein()
        self.phantom = Phantom(vein=self.vein)
        self.vitactip = ViTacTip()
        self.compute_sensor_bounds()
        self.fisheye_model = FisheyeModelNoTaichi()
        self.set_up_system_params()
        self.load_system_identification_data()
        self.set_up_initial_positions_and_trajectory_first_init_only()
        self.set_up_trajectories_and_phantom_states()
        self.set_up_initial_positions_state_and_trajectory()
        self.set_up_collision_detection()
        self.set_up_pid()
        self.set_up_snapshot()
        self.visualisation_initialise()
        self.training_data_collection_initialise()
        self.foo()
        self.bo = BoGp()
    
    # def write_da_total_loss_to_file(self):
    #     target_data = {
    #         'target': sum(self.da_losses),
    #     }
    #     with open(self.target_path, "w") as f:
    #         json.dump(target_data, f, indent=4)
    #     self.da_losses = []
    
    def compute_da_loss(self, out_dir=None):
        """MAE between simulated and real markers at the current pose.

        Appends the value (in pixels) to `self.da_losses` and writes the
        red/green alignment overlay. `out_dir` redirects that overlay, so a
        timestamped run keeps its own copy instead of overwriting the shared one.
        """
        h = int(SYSTEM_PARAMS.fisheye_model.target_image_height)
        # Projects the deformed markers into image space and fills
        # `sim_markers_deformed`, which the MAE below reads. Despite the name it
        # is NOT optional here: it is the projection step, and the tactile-readout
        # rendering merely happens to reuse it. Calling it explicitly is what
        # keeps the measurement independent of whether a window is being drawn -
        # headless runs used to score against an all-zero array, giving a
        # constant ~1169 px MAE that no parameter change could move.
        self.visualisation_prepare_tactile_readout_data_fp()
        self.move_og_resolution()
        sim_points = self.sim_markers_deformed.to_numpy()
        self.move_ti_resolution()
        sim_points[:, 1] = h-sim_points[:, 1]
        _, sim_points_reordered, _ = Adjacency.get_graph_connectivity(sim_points)
        trajectory_ix = self.trajectory_ix[None]
        trajectory_name = self.trajectory_names[trajectory_ix]
        file_path = self.da_npz_paths[trajectory_name]
        data = np.load(file_path)
        exp_points = data['points']
        a = sim_points_reordered
        b = exp_points
        distances = np.linalg.norm(a-b, axis=1)
        mae = distances.mean()
        if out_dir is None:
            img_out = self.da_overlay.format(trajectory_name)
        else:
            img_out = os.path.join(out_dir, f"da_overlay_{trajectory_name}.png")
        self.generate_validation_img(
            a, b,
            img_in=self.default_photo,
            img_out=img_out,
        )
        # CACHE THE RAW POSITIONS, not just the rendered overlay. The overlay
        # bakes the markers into a photograph at one particular style; the
        # coordinates let a figure be re-drawn later - different colours, a white
        # background, connector lines - WITHOUT re-running the simulation, which
        # is minutes per trajectory. `alignment_figures.sh` reads exactly this.
        if out_dir is not None:
            np.savez_compressed(
                os.path.join(out_dir, f"markers_{trajectory_name}.npz"),
                sim=a, real=b, distances=distances, mae_px=float(mae),
            )
        self.da_losses.append(mae)
        # The PER-MARKER distances behind that mean, kept so a caller can also
        # use their maximum (the worst-aligned marker) without recomputing the
        # projection. `da_losses` stays a list of means, so nothing that reads it
        # changes.
        self.da_last_distances = distances
        # The reordered simulated marker POSITIONS, so two runs can be compared
        # against each other rather than each against the photograph. Reordered
        # by `get_graph_connectivity`, so index i is the same physical marker in
        # every snapshot - which is what makes a sim-vs-sim difference
        # meaningful.
        self.da_last_sim_points = sim_points_reordered
        return distances
    
    def sensor_rest_min_z(self):
        """The sensor's lowest dome-surface z in its REST pose, cached.

        The floor of the penetration scale: the sensor cannot be higher than
        this when undeformed and undisplaced, so it anchors the normalisation at
        "no penetration at all". Measured once, on first use, before any
        trajectory has moved the mesh - hence the cache, since later calls would
        see a deformed pose.
        """
        if getattr(self, "_sensor_rest_min_z", None) is None:
            v = self.vitactip.vertices_deformed_A.to_numpy()[0]
            surface = v[self.vitactip.dome_surface_node_tags_npy]
            self._sensor_rest_min_z = float(np.min(surface[:, 2]))
        return self._sensor_rest_min_z

    def node_nearest_vein_apex(self, node_index=None):
        """z of the dome-surface node closest to the vessel's apex point p.

        Returns (q_z, num_nodes, node_index).

        With `node_index=None` it searches. Candidates are the surface nodes
        whose xy projection falls inside the vein's xy rectangle, and the winner
        is the one at minimum 3D distance to p - the top of the curved wall at
        the cylinder's mid-length (see `Vein.apex_point`).

        NEAREST-TO-APEX RATHER THAN LOWEST-Z. Taking the lowest node picks
        whichever part of the dome happens to hang furthest down, which need not
        be the part facing the vessel at all - it drifts around the footprint as
        the sensor slides and tilts, so consecutive measurements are not of the
        same region. Distance to a FIXED analytical point selects the node
        actually over the vessel's high point, which is where any contact would
        occur.

        p is computed from the pose, radius and length alone; no vein particles
        are consulted.

        With `node_index` given it does NOT search - it reports that specific
        node's z. This is what makes a vessel-present/vessel-free comparison
        meaningful: an independent search picks whichever node wins in EACH run,
        so comparing two independent winners compares two DIFFERENT physical
        points on the sensor. The measured difference was then mostly "which node
        won", not "what the vessel did" - and it came out with the wrong sign,
        the sensor appearing to sink 1-10 mm FURTHER when a rigid inclusion was
        present. Pinning the index turns it into a per-node displacement of one
        identified point, and doing so reversed the sign to a physical one.

        `num_nodes` is 0 and `q_z` NaN when nothing lies over the vein (searching
        mode only; a pinned index is always reported).
        """
        v = self.vitactip.vertices_deformed_A.to_numpy()[0]
        surface = v[self.vitactip.dome_surface_node_tags_npy]
        if node_index is not None:
            return float(surface[node_index, 2]), 1, int(node_index)
        x_min, x_max, y_min, y_max = self.vein.xy_footprint()
        inside = (
            (surface[:, 0] >= x_min) & (surface[:, 0] <= x_max)
            & (surface[:, 1] >= y_min) & (surface[:, 1] <= y_max)
        )
        num_nodes = int(inside.sum())
        if num_nodes == 0:
            return float("nan"), 0, None
        # Index into the full surface array, not into the filtered subset.
        candidates = np.flatnonzero(inside)
        apex = self.vein.apex_point()
        distances = np.linalg.norm(surface[candidates] - apex, axis=1)
        winner = int(candidates[np.argmin(distances)])
        return float(surface[winner, 2]), num_nodes, winner

    def vein_penetration_normalised(self, z_reference=None):
        """How far the sensor has pressed toward/into the vein, on [0, 1].

        Returns (vpn, q_z, num_nodes). `q_z` is the z of point q - the LOWEST
        dome-surface node lying inside the vein's xy rectangle - and `vpn` is
        that mapped onto [0, 1] by clamping to

            [z_reference, z_v]

        where z_v is the vein's analytical top.

        `z_reference` IS THE VESSEL-FREE q_z, NOT THE REST POSE. It is the depth
        the sensor reaches at this same moment with no vessel present - i.e. how
        far down it would go if the vessel did nothing - so the scale measures
        the VESSEL'S contribution alone. Referencing the rest pose instead
        confounds two effects: a very soft sensor compresses under the phantom
        press and so sits higher than its rest floor even with no vessel there,
        which would read as "the vessel stopped it".

        Falls back to the rest-pose floor when no reference is supplied, which
        only happens outside the joint loop.

        THE SCALE REWARDS THE SENSOR STOPPING AT THE VESSEL, NOT PASSING THROUGH
        IT. z increases upwards, so `vpn` is a direct min-max map of q_z:

            vpn = (clamp(q_z) - z_min) / (hi - z_min)

        giving vpn = 1 when the sensor's lowest point over the vein is still
        clear of it (a gap remains, at the top of the range) and vpn = 0 when it
        has sunk to the rest-pose floor - i.e. ghosted straight through the
        inclusion as though it were not there.

        That direction is the physically meaningful one HERE because the vein is
        RIGID: a real rigid inclusion stops the sensor, so a simulation in which
        the sensor passes through it is one where the sensor<->vein contact is
        not doing its job. Rewarding a large q_z therefore rewards contact
        actually being resolved, which is the precondition for any marker signal.

        A sensor not over the vein at all has no q. It returns vpn = 0 rather
        than 1: "not over the vessel" is not evidence of contact being resolved,
        and scoring it top would let the search win by avoiding the vessel
        entirely.
        """
        z_min = (
            self.sensor_rest_min_z() if z_reference is None
            else float(z_reference)
        )
        q_z, num_nodes, _ = self.node_nearest_vein_apex()
        if num_nodes == 0:
            return 0.0, q_z, num_nodes
        vpn = normalised_penetration(q_z, z_min, self.vein.max_z())
        return vpn, q_z, num_nodes

    def sensor_reaches_vein_depth(self):
        """Has the part of the sensor ABOVE the vein pressed down to the vein?

        Returns (reaches, z_s, z_v, num_nodes), where

            z_s = the MINIMUM z over the sensor's dome-surface nodes that lie
                  INSIDE the vein's xy footprint
            z_v = the MAXIMUM z over the vein's particles

        and `reaches` is `z_s <= z_v`.

        WHY THE XY RESTRICTION IS THE WHOLE POINT. z increases upwards: the vein
        spans z [0.050, 0.070] and the sensor's dome spans [0.060, 0.085] in the
        rest pose, so a GLOBAL minimum over the sensor surface is already below
        the vein's top before the trajectory even starts - the test would return
        True for every configuration and the objective would be constant. The
        sensor and the vein are ~0.15 m apart in xy at rest; only once the slide
        brings the sensor over the vein does any of its surface sit within the
        footprint, and only then does "how far down does it reach" mean anything.

        The footprint is the vein's analytical xy rectangle (see
        `Vein.xy_footprint`): the vein is a cylinder lying flat along +x, so seen
        from above it is x in [x0, x0+h] by y in [y0-r, y0+r].

        `num_nodes` is how many surface nodes fell inside. When it is zero the
        sensor is not over the vein at all, and the function reports
        `reaches=False` with z_s = NaN - "not reaching" rather than an accidental
        pass from an empty minimum.
        """
        v = self.vitactip.vertices_deformed_A.to_numpy()[0]
        surface = v[self.vitactip.dome_surface_node_tags_npy]
        vein = self.vein.particles_A.to_numpy()
        if surface.size == 0 or vein.size == 0:
            return False, float("nan"), float("nan"), 0
        x_min, x_max, y_min, y_max = self.vein.xy_footprint()
        inside = (
            (surface[:, 0] >= x_min) & (surface[:, 0] <= x_max)
            & (surface[:, 1] >= y_min) & (surface[:, 1] <= y_max)
        )
        z_v = float(np.max(vein[:, 2]))
        num_nodes = int(inside.sum())
        if num_nodes == 0:
            return False, float("nan"), z_v, 0
        z_s = float(np.min(surface[inside, 2]))
        return z_s <= z_v, z_s, z_v, num_nodes

    def sensor_state_is_healthy(self):
        """Is the sensor's FEM state still physical? Returns (healthy, reason).

        A blown-up solve is visually obvious - the green sensor particles vanish
        from the GGUI view, because their coordinates are NaN or so large that
        they fall outside the camera frustum. It is much less obvious in the
        numbers, and that is the dangerous part for a MAXIMISING objective: a
        diverged sensor produces enormous marker errors, so `3A - B` rewards it
        far above any real configuration. Iterations 1 and 5 of the first joint
        run scored 432 and 397 with mean marker errors of 123 and 115 px, an
        order of magnitude above a sane ~10 px - the search was being led by
        divergence, not by vessel sensitivity.

        Three checks, cheapest first:

          1. NON-FINITE vertices - NaN or inf anywhere in the deformed mesh.
             This is the definitive symptom; everything else is a leading
             indicator.
          2. RUNAWAY coordinates - the sensor is ~0.09 m across and lives inside
             a domain of order 0.2 m, so any vertex beyond 10 m has escaped,
             whatever the solver thinks.
          3. INVERTED elements - a tetrahedron with a non-positive Jacobian has
             turned inside out. Measured on `twist_x`, a healthy run keeps
             min J ~= 0.96 and a badly-conditioned but stable one ~= 0.45, so a
             J <= 0 element means the deformation gradient is degenerate and the
             stress is meaningless.

        Deliberately does NOT flag a merely large deformation: soft sensors
        legitimately compress a long way (min J 0.449 at E=1e4), and treating
        that as a failure would silently exclude a whole region of the search
        space rather than the broken part of it.
        """
        v = self.vitactip.vertices_deformed_A.to_numpy()[0]
        if not np.isfinite(v).all():
            return False, "non-finite vertex coordinates (NaN/inf)"
        max_abs = float(np.max(np.abs(v)))
        if max_abs > 10.0:
            return False, f"runaway vertex coordinate (max |x| = {max_abs:.3g} m)"

        # Element inversion, computed exactly as update_internal_forces() forms
        # the deformation gradient: F = Dm @ Dm_inv, J = det(F).
        tets = self.vitactip.tetrahedra_npy
        dm_inv = self.vitactip.initial_deformation_gradient_inverse.to_numpy()
        dm = np.stack(
            [v[tets[:, 0]] - v[tets[:, 3]],
             v[tets[:, 1]] - v[tets[:, 3]],
             v[tets[:, 2]] - v[tets[:, 3]]],
            axis=2,
        )
        jac = np.linalg.det(dm @ dm_inv)
        if not np.isfinite(jac).all():
            return False, "non-finite element Jacobian"
        num_inverted = int((jac <= 0).sum())
        if num_inverted:
            return False, (
                f"{num_inverted} inverted element(s), min J = {jac.min():.4f}"
            )
        return True, ""

    def vein_over_sensor_centre(self, radius_px=None):
        """Is any projected vein point within `radius_px` of the sensor centre?

        The trigger for the vessel-present objective: it marks the moment the
        sliding sensor is directly ABOVE the subsurface vessel, which is where
        the vessel's effect on the marker field is largest.

        "Centre" is the centroid of the DEFORMED simulated markers, not the
        sensor's rest pose - the sensor translates during a slide, so a fixed
        reference would drift out from under it and the trigger would fire at
        the wrong time (or never).

        MIND THE UNITS. These two fields are NOT in the same space:
          * `vein_2d_projection`      RAW PIXELS
          * `sim_markers_deformed`    NORMALISED to [0, 1], because
            visualisation_prepare_tactile_readout_data_fp() divides by
            `tactile_image_resolution`
        (`vein_2d_projection_flat` is the normalised twin of the first.)
        Comparing them directly measures a pixel-scale coordinate against a
        ~0.5 centroid, giving a large distance that does not vary with the
        simulation - it produced a constant 805.7 px across every parameter set,
        so the trigger never fired and every iteration scored 0. The markers are
        therefore scaled back to pixels here before the comparison.

        Vein points that were never filled are sentinel -1 and are dropped.

        Returns (triggered, distance_px). `distance_px` is the closest vein
        point's distance to the centre, or inf when the vein has no valid
        projection - which is the case whenever the vein pair is disabled.
        """
        radius_px = radius_px if radius_px is not None else _env_int(
            "DIFFTACTILE_VEIN_TRIGGER_RADIUS_PX", INTER_MARKER_SPACING_PX
        )
        vein = self.vein_2d_projection.to_numpy().reshape(-1, 2)
        # -1 is the fill value for "not projected"; a real projection is >= 0.
        valid = vein[(vein[:, 0] >= 0) & (vein[:, 1] >= 0)]
        if valid.size == 0:
            return False, float("inf")
        resolution = self.tactile_image_resolution[None]
        scale = np.array([resolution[0], resolution[1]], dtype=float)
        centre = self.sim_markers_deformed.to_numpy().mean(axis=0) * scale
        d = float(np.min(np.linalg.norm(valid - centre, axis=1)))
        return d <= radius_px, d

    def compute_vein_da_loss(self, out_dir=None, tag="vein"):
        """MAE between the CURRENT deformed markers and the real-photo markers.

        Same measurement as `compute_da_loss`, but taken at the moment the vein
        passes under the sensor centre rather than at the trajectory's apex, and
        written under a different name so it cannot overwrite the vessel-absent
        overlays.

        The vessel-present objective MAXIMISES this: a large disagreement with a
        photograph of a VESSEL-FREE phantom means the subsurface vessel is
        visibly deforming the marker field, which is the signal the GNN has to
        detect. Note this is the same real-world photograph the vessel-absent
        model is scored against - deliberately, since the whole quantity of
        interest is "how far does the vessel push the sensor away from its
        vessel-free appearance".
        """
        h = int(SYSTEM_PARAMS.fisheye_model.target_image_height)
        self.visualisation_prepare_tactile_readout_data_fp()
        self.move_og_resolution()
        sim_points = self.sim_markers_deformed.to_numpy()
        self.move_ti_resolution()
        sim_points[:, 1] = h - sim_points[:, 1]
        _, sim_points_reordered, _ = Adjacency.get_graph_connectivity(sim_points)
        trajectory_name = self.trajectory_names[self.trajectory_ix[None]]
        data = np.load(self.da_npz_paths[trajectory_name])
        exp_points = data['points']
        distances = np.linalg.norm(sim_points_reordered - exp_points, axis=1)
        if out_dir is not None:
            self.generate_validation_img(
                sim_points_reordered, exp_points,
                img_in=self.default_photo,
                img_out=os.path.join(out_dir, f"{tag}_overlay_{trajectory_name}.png"),
            )
        self.da_last_distances = distances
        self.da_last_sim_points = sim_points_reordered
        return float(distances.mean())

    def generate_validation_img(self, points1, points2, img_in, img_out):
        img = cv2.imread(img_in)
        points1 = points1.astype(np.int32)
        points2 = points2.astype(np.int32)
        for point in points1:
            cv2.circle(img, tuple(point), radius=3, color=(0, 0, 255), thickness=-1)  # BGR format: red
        for point in points2:
            cv2.circle(img, tuple(point), radius=3, color=(0, 255, 0), thickness=-1)  # BGR format: green
        cv2.imwrite(img_out, img)
    
    @ti.kernel
    def update_vitactip_tip_point(self):
        self.vitactip_tip_point[0] = self.vitactip.vertices_undeformed_A[
            self.num_sub_frames-1, 
            self.vitactip.tip_ix[None],
        ]
    
    @ti.kernel
    def update_clock_arm_points_3d(self):
        for i in range(self.vitactip.clock_arms_node_idxs.shape[0]):
            node_idx = self.vitactip.clock_arms_node_idxs[i]
            vertex = self.vitactip.vertices_undeformed_A[
                self.num_sub_frames-1,
                node_idx,
            ]
            self.clock_arm_points_3d[i] = vertex
    
    def foo(self):
        self.da_npz_paths = {
            'press': f'{SYSTEM_PARAMS.files.da_press_npz}',
            'twist_z': f'{SYSTEM_PARAMS.files.da_twist_z_npz}',
            'twist_x': f'{SYSTEM_PARAMS.files.da_twist_x_npz}',
            'slide': f'{SYSTEM_PARAMS.files.da_slide_npz}',
        }
        self.num_sub_frames = SYSTEM_PARAMS.contact.num_sub_frames
        self.max_ts = SYSTEM_PARAMS.meta.max_timesteps_per_trajectory
        self.vitactip_tip_point = ti.Vector.field(
            3,
            dtype=float,
            shape=(1,),
            needs_grad=False,
        )
        self.clock_arm_points_3d = ti.Vector.field(
            3,
            dtype=float,
            shape=(2,),
            needs_grad=False,
        )
        self.vitactip_vertices_temp = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.vitactip.vertices_deformed_A.shape[1],),
            needs_grad=False,
        )
        self.all_points = []
        # CONTACT PAIR 0 (sensor<->phantom) IS DISABLED PROJECT-WIDE.
        #
        # The phantom's particles are pinned in place, so it does not deform and
        # the sensor effectively ghosts through it: resolving pair-0 contact
        # changes nothing that any objective measures, and its four coefficients
        # are unidentifiable. They are set to -1 in system-params.json (a
        # physically impossible negative stiffness) so the file itself shows they
        # are inert, and set_contact_params() re-asserts that in code.
        #
        # Only pair 2 (sensor<->vein) is live. Re-enable pair 0 here, restore
        # real coefficients in system-params.json, and put its parameters back
        # into the BO search space if the phantom is ever made deformable.
        self.collision_ixs = active_collision_pairs(True)
        self.collision_resolvers = [
            self.collision0,
            self.collision1,
            self.collision2,
        ]
        self.collision_detectors = [
            self.check_collision0,
            self.check_collision1,
            self.check_collision2,
        ]
    
    def detect_collisions(self, f):
        for i in range(len(self.collision_ixs)):
            ix = self.collision_ixs[i]
            self.collision_detectors[ix](f)
    
    def resolve_collisions(self, f):
        for i in NP_RNG.permutation(len(self.collision_ixs)):
            ix = self.collision_ixs[i]
            self.collision_resolvers[ix](f)

    def vein_sparse_to_dense_init(self):
        self.num_veins = SYSTEM_PARAMS.meta.max_num_veins
        self.vein_counts = ti.field(int, (self.num_veins,), needs_grad=False)
        self.vein_indices = ti.field(
            int, (self.num_veins, self.phantom.num_particles), needs_grad=False
        )
        self.max_vein_count = ti.field(int, (), needs_grad=False)
    
    def vein_sparse_to_dense(self):
        vein_titles = self.phantom.vein_titles.to_numpy()
        unique_titles, counts = np.unique(vein_titles, return_counts=True)
        vein_counts = np.zeros(shape=(self.num_veins,), dtype=int)
        for vein_ix, count in zip(unique_titles, counts):
            if vein_ix != -1:
                vein_counts[vein_ix] = count
        self.vein_counts.from_numpy(vein_counts)
        self.max_vein_count[None] = np.max(vein_counts)
        vein_counts_temp = np.zeros(shape=(self.num_veins,), dtype=int)
        vein_indices = -np.ones(shape=(self.num_veins, self.phantom.num_particles), dtype=int)
        for particle_ix in range(len(vein_titles)):
            vein_ix = vein_titles[particle_ix]
            if vein_ix != -1:
                vein_particle_ix = vein_counts_temp[vein_ix]
                vein_indices[vein_ix, vein_particle_ix] = particle_ix
                vein_counts_temp[vein_ix] += 1
        self.vein_indices.from_numpy(vein_indices)
    
    def compute_sensor_bounds(self):
        _min = np.array(SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex[:2])
        _mid = np.array(SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose[:2])
        phantom_r = np.abs(_mid - _min)
        _max = _min + phantom_r * 2
        sensor_r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        sensor_min = _min + sensor_r
        sensor_max = _max - sensor_r

        self.sensor_x_range_world = np.array([
            sensor_min[0],
            sensor_max[0]
        ])
        self.sensor_y_range_world = np.array([
            sensor_min[1],
            sensor_max[1]
        ])

        self.sensor_x_range_phantom = self.sensor_x_range_world.copy()
        self.sensor_x_range_phantom -= _mid[0]
        self.sensor_y_range_phantom = self.sensor_y_range_world.copy()
        self.sensor_y_range_phantom -= _mid[1]

    def training_data_collection_initialise(self):
        self.marker_data = []
        self.vein_polyline_data = []
        self.vein_polyline_mask_data = []
        self.target_id_data = []
        self.pose_data = []
        # Optional override of where write_training_data_to_file() writes; None
        # means the timestamped `files.dataset_root` as always. Set by the
        # vessel-map trajectory entrypoint so its one trajectory does not land
        # among training-data collections.
        self.training_data_dir_override = None
        self.vein_cx_A = None
        self.target_3_ts = 12
        self.target_4_ts = 226
        # self.vein_sparse_to_dense_init()
        self.generate_tumour = False

    @ti.kernel
    def fp(self):
        self.fp_bp[None] = 0

    @ti.kernel
    def bp(self):
        self.fp_bp[None] = 1

    def load_system_identification_data(self):
        self.load_system_identification_data_1()
        self.load_system_identification_data_2()
        self.load_system_identification_data_3()

    def load_system_identification_data_1(self):
        self.exp_marker_shapes_np = np.zeros(shape=(4, 3), dtype=int)
        for i in range(4):
            with open(SYSTEM_PARAMS.files.traj_markers.format(i), "rb") as f:
                markers_array = pickle.load(f)
            self.exp_marker_shapes_np[i] = markers_array.shape
        
        self.exp_marker_shapes = ti.Vector.field(3, dtype=int, shape=(self.exp_marker_shapes_np.shape[0]))
        self.exp_marker_shapes.from_numpy(self.exp_marker_shapes_np)
        max_0, max_1, max_2 = self.exp_marker_shapes_np.max(axis=0)
        self.exp_markers_max_shapes_np = np.array([max_0, max_1, max_2])
        self.exp_markers_max_shapes = ti.field(dtype=int, shape=(3,), needs_grad=False)
        self.exp_markers_max_shapes.from_numpy(self.exp_markers_max_shapes_np)
        self.exp_markers_np = -np.ones(shape=(4, max_0, max_1, 2), dtype=float)

        for i in range(4):
            with open(SYSTEM_PARAMS.files.traj_markers.format(i), "rb") as f:
                markers_array = pickle.load(f)
            x, y, z = markers_array.shape
            self.exp_markers_np[i, :x, :y, :z] = markers_array

        self.exp_markers = ti.Vector.field(2, dtype=float, shape=(4, max_0, max_1), needs_grad=False)
        self.marker_position_exp = ti.Vector.field(2, dtype=float, shape=(max_1,), needs_grad=False)
        self.sim_to_exp_markers = ti.field(dtype=int, shape=(127,), needs_grad=False)
        self.exp_to_sim_markers = ti.field(dtype=int, shape=(127,), needs_grad=False)

        self.traj_1_exp_marker_pairs_np = np.array([
            [104, 105],
            [105, 67],
            [67, 46],
            [46, 68],
            [99, 79],
            [79, 43],
            [43, 44],
            [44, 66],
            [66, 16],
            [41, 98],
            [98, 63],
            [63, 78],
            [78, 42]
        ])
        self.traj_1_exp_marker_pairs = ti.Vector.field(
            2, 
            dtype=int, 
            shape=(self.traj_1_exp_marker_pairs_np.shape[0],),
            needs_grad=False
        )
        self.traj_1_exp_marker_pairs.from_numpy(self.traj_1_exp_marker_pairs_np)
        self.traj_1_critical_frames_exp_np = np.array([175, 231], dtype=int)
        self.traj_1_critical_frames_exp = ti.field(
            dtype=int,
            shape=(self.traj_1_critical_frames_exp_np.shape[0],),
            needs_grad=False
        )
        self.traj_1_critical_frames_exp.from_numpy(self.traj_1_critical_frames_exp_np)
        self.cur_exp_frame = ti.field(
            dtype=float,
            shape=(),
            needs_grad=False
        )
    
    @ti.kernel
    def load_system_identification_data_2(self):
        self.exp_markers.fill(-1)
        self.marker_position_exp.fill(-1)
        self.sim_to_exp_markers.fill(-1)
        self.exp_to_sim_markers.fill(-1)
    
    def load_system_identification_data_3(self):
        self.exp_markers.from_numpy(self.exp_markers_np)

    def compute_mapping_between_experimental_and_sim_markers(self):
        x, y, z = self.exp_marker_shapes_np[self.trajectory_ix[None]]
        exp_markers = self.exp_markers.to_numpy()[self.trajectory_ix[None], 0, :y, :z]
        sim_markers = self.vitactip.undeformed_markers.to_numpy()
        assert sim_markers.shape[0] == 127
        cost_matrix = cdist(sim_markers, exp_markers, metric="sqeuclidean")
        ixs_1, ixs_2 = linear_sum_assignment(cost_matrix)
        self.sim_to_exp_markers_np = np.full(127, -1, dtype=np.int32)
        for ix_1, ix_2 in zip(ixs_1, ixs_2):
            self.sim_to_exp_markers_np[ix_1] = ix_2
        self.sim_to_exp_markers.from_numpy(self.sim_to_exp_markers_np)
        self.exp_to_sim_markers_np = -np.ones_like(self.sim_to_exp_markers_np)
        for i in range(self.sim_to_exp_markers_np.shape[0]):
            exp = self.sim_to_exp_markers_np[i]
            if exp != -1:
                self.exp_to_sim_markers_np[exp] = i
        self.exp_to_sim_markers.from_numpy(self.exp_to_sim_markers_np)
    
    @ti.func
    def project_point_on_line(self, point, line_point1, line_point2):
        line_vector = line_point2 - line_point1
        point_vector = point - line_point1
        line_direction = line_vector / ti.math.length(line_vector)
        projection_length = ti.math.dot(point_vector, line_direction)
        projected_point = line_point1 + projection_length * line_direction
        return projected_point

    @ti.kernel
    def interpolate_experimental_frame(self, ts: ti.i32):
        start_ix = -1
        end_ix = -1
        # 5 target points
        # current_target_idx = 4
        # for i in range(4); i = 0,1,2,3
        # 0,1
        # 1,2
        # 2,3
        # 3,4
        for i in range(self.current_target_idx[None]):
            if ts == self.sim_keypoints[i]:
                start_ix = i
            elif ts > self.sim_keypoints[i] and ts < self.sim_keypoints[i + 1]:
                start_ix = i
                end_ix = start_ix + 1
            elif ts == self.sim_keypoints[i+1]:
                start_ix = i+1
        # if sim entry is invalid, it's -1
        # if exp entry is invalid, it's -1
        cur_exp_keypoints = self.exp_keypoints[self.trajectory_ix[None]]

        if (
            start_ix != -1 
            and cur_exp_keypoints[start_ix] != -1
        ):
            exp_keypoint = -1.0
            if (
                False
                and self.trajectory_ix[None] == 1
                and start_ix >= 3
            ):
                target = self.phantom.particles_A[
                    SYSTEM_PARAMS.contact.num_sub_frames - 1,
                    self.vein_endpoints_indices[0]
                ]
                x_E = self.exp_vein_3d_coords_E[0]
                y_E = self.exp_vein_3d_coords_E[1]
                x_A = self.vitactip.project_E_to_A(x_E)
                y_A = self.vitactip.project_E_to_A(y_E)
                target_projected = self.project_point_on_line(
                    target,
                    x_A,
                    y_A
                )

                min_dist = SYSTEM_PARAMS.geometry.high_dist
                min_ix = -1
                for i in range(self.exp_vein_3d_coords_E_all.shape[0]):
                    exp_vein_point_E = self.exp_vein_3d_coords_E_all[i]
                    if ti.math.length(
                        exp_vein_point_E
                        - ti.Vector([-1.0, -1.0, -1.0], dt=float)
                    ) > 1e-6:
                        exp_vein_point_A = self.vitactip.project_E_to_A(exp_vein_point_E)
                        dist = ti.math.length(
                            exp_vein_point_A
                            - target_projected
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_ix = i
                min_dist_0 = min_dist
                min_ix_0 = min_ix
                min_point_A_0 = self.vitactip.project_E_to_A(
                    self.exp_vein_3d_coords_E_all[min_ix_0]
                )

                min_dist = SYSTEM_PARAMS.geometry.high_dist
                min_ix = -1
                for i in range(self.exp_vein_3d_coords_E_all.shape[0]):
                    exp_vein_point_E = self.exp_vein_3d_coords_E_all[i]
                    if (
                        ti.math.length(
                        exp_vein_point_E
                        - ti.Vector([-1.0, -1.0, -1.0], dt=float)
                        ) > 1e-6
                        and i != min_ix_0
                    ):
                        exp_vein_point_A = self.vitactip.project_E_to_A(exp_vein_point_E)
                        dist = ti.math.length(
                            exp_vein_point_A
                            - target_projected
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_ix = i
                min_dist_1 = min_dist
                min_ix_1 = min_ix
                min_point_A_1 = self.vitactip.project_E_to_A(
                    self.exp_vein_3d_coords_E_all[min_ix_1]
                )
                
                # assert abs(min_ix_0 - min_ix_1) == 1, f"Interpolation video frames aren't consecutive ({min_ix_0}, {min_ix_1})"
                dist_sum = min_dist_0 + min_dist_1
                a = -1
                b = -1
                a_dist = -1.0
                b_dist = -1.0
                if min_ix_0 < min_ix_1:
                    a = min_ix_0
                    b = min_ix_1
                    a_dist = min_dist_0
                    b_dist = min_dist_1
                else:
                    a = min_ix_1
                    b = min_ix_0
                    a_dist = min_dist_1
                    b_dist = min_dist_0

                offset = a_dist / dist_sum
                exp_keypoint = a + offset

                dist_exp_veins = ti.math.length(
                    min_point_A_1
                    - min_point_A_0
                )

                if (
                    min_dist_0 < dist_exp_veins
                    and min_dist_1 < dist_exp_veins
                ):
                    self.vein_ix_base[None] = a
                    self.vein_ix_offset[None] = offset
                    self.interpolation_valid[None] = 1
                else:
                    self.vein_ix_base[None] = -1
                    self.vein_ix_offset[None] = -1.0
                    self.interpolation_valid[None] = 0

                if False:
                    print(f'target: {target}')
                    print(f'x_E: {x_E}')
                    print(f'y_E: {y_E}')
                    print(f'x_A: {x_A}')
                    print(f'y_A: {y_A}')
                    print(f'target_projected: {target_projected}')
                    print(f'min_dist_0: {min_dist_0}')
                    print(f'min_ix_0: {min_ix_0}')
                    print(f'min_dist_1: {min_dist_1}')
                    print(f'min_ix_1: {min_ix_1}')
                    print(f'dist_sum: {dist_sum}')
                    print(f'a: {a}')
                    print(f'b: {b}')
                    print(f'a_dist: {a_dist}')
                    print(f'b_dist: {b_dist}')
                    print(f'offset: {offset}')
                    print(f'exp_keypoint: {exp_keypoint}')
            else:
                if end_ix != -1:
                    exp_keypoint = (
                        cur_exp_keypoints[start_ix] 
                        + (
                            cur_exp_keypoints[end_ix]
                            - cur_exp_keypoints[start_ix]
                        ) * (
                            ts - self.sim_keypoints[start_ix]
                        ) / (
                            self.sim_keypoints[end_ix] - self.sim_keypoints[start_ix]
                        )
                    )
                else:
                    exp_keypoint = cur_exp_keypoints[start_ix]
            if self.interpolation_valid[None] == 1:
                self.cur_exp_frame[None] = exp_keypoint
            for i in range(self.exp_marker_shapes[self.trajectory_ix[None]][1]):
                for j in range(2):
                    if self.interpolation_valid[None] == 1:
                        if end_ix != -1:
                            self.marker_position_exp[i][j] = (
                                self.exp_markers[self.trajectory_ix[None], ti.floor(exp_keypoint, dtype=ti.i32), i][j] 
                                + (exp_keypoint - ti.floor(exp_keypoint)) 
                                * (
                                    self.exp_markers[self.trajectory_ix[None], ti.ceil(exp_keypoint, dtype=ti.i32), i][j]
                                    - self.exp_markers[self.trajectory_ix[None], ti.floor(exp_keypoint, dtype=ti.i32), i][j]
                                )
                            )
                        else:
                            self.marker_position_exp[i][j] = (
                                self.exp_markers[self.trajectory_ix[None], ti.floor(exp_keypoint, dtype=ti.i32), i][j]
                            )
                    else:
                        self.marker_position_exp[i][j] = -1.0

    @ti.kernel
    def compute_vein_exp_vis(self):
        start_ix = self.vein_ix_base[None]
        if start_ix != -1:
            end_ix = start_ix + 1
            offset = self.vein_ix_offset[None]
            start_E = self.exp_vein_3d_coords_E_all[start_ix]
            end_E = self.exp_vein_3d_coords_E_all[end_ix]
            start_A = self.vitactip.project_E_to_A(start_E)
            end_A = self.vitactip.project_E_to_A(end_E)
            point = start_A + offset * (end_A - start_A)
            self.vein_exp_vis[None] = point
    
    @ti.kernel
    def compute_vein_exp_vis_all(self):
        for i in range(self.exp_vein_3d_coords_E_all.shape[0]):
            point_E = self.exp_vein_3d_coords_E_all[i]
            if ti.math.length(
                point_E
                - ti.Vector([-1.0, -1.0, -1.0], dt=float)
            ) > 1e-6:
                point_A = self.vitactip.project_E_to_A(point_E)
                self.vein_exp_vis_all[i] = point_A
   
    @ti.kernel
    def compute_validation_point(self):
        point_E = self.validation_point_3d_E[None]
        point_A = self.vitactip.project_E_to_A(point_E)
        self.validation_point_3d_A[None] = point_A

    def set_up_collision_detection(self):
        self.triangle_ix_contact_0 = ti.field(
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.triangle_ix_contact_1 = ti.field(
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
                SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
            ),
            needs_grad=False,
        )
        self.triangle_ix_contact_2 = ti.field(
            dtype=int,
            shape=(
                SYSTEM_PARAMS.contact.num_sub_frames,
                self.vein.particles_A.shape[0],
            ),
            needs_grad=False,
        )

    def set_up_system_params(self):
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.collision2_contact_flat = ti.field(dtype=int, shape=(), needs_grad=False)
        self.target_path = SYSTEM_PARAMS.files.bo_gp_target_json
        self.da_overlay = SYSTEM_PARAMS.files.da_overlay
        self.photo_timesteps = {
            # 'press': 35,
            # 'twist_z': 180,
            # 'twist_x': 51,
            'slide': 327,
        }
        self.da_losses = []
        self.dist_sf = SYSTEM_PARAMS.meta.distance_scaling_factor
        self.sensor_r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        default_photo = SYSTEM_PARAMS.files.flat_sensor_default_state
        dir = SYSTEM_PARAMS.files.da_dir
        self.default_photo = f'{dir}{default_photo}'
        self.num_contact_pairs = SYSTEM_PARAMS.meta.num_contact_pairs
        self.trajectory_ix = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dt = ti.field(dtype=float, shape=(), needs_grad=False)
        self.dt[None] = SYSTEM_PARAMS.contact.dt_override
        self.normal_stiffness = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=False)
        self.normal_damping = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=False)
        self.tangential_stiffness = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=False)
        self.coulomb_friction_coeff = ti.field(dtype=float, shape=(self.num_contact_pairs,), needs_grad=False)
        self.normal_stiffness.from_numpy(np.array(SYSTEM_PARAMS.contact.normal_stiffness))
        self.normal_damping.from_numpy(np.array(SYSTEM_PARAMS.contact.normal_damping))
        self.tangential_stiffness.from_numpy(np.array(SYSTEM_PARAMS.contact.tangential_stiffness))
        self.coulomb_friction_coeff.from_numpy(np.array(SYSTEM_PARAMS.contact.coulomb_friction_coeff))
        self.gradients_printed = False
        self.courant_number = SYSTEM_PARAMS.meta.target_courant_number
        self.retry = False
        phantom_closest_vertex = SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex
        self.phantom_closest_vertex = np.array(phantom_closest_vertex, dtype=float)
        phantom_dimensions = SYSTEM_PARAMS_COMPUTED.phantom_dimensions
        self.phantom_dimensions = np.array(phantom_dimensions, dtype=float)
        self.gap = SYSTEM_PARAMS.geometry.gap

    def set_up_snapshot(self):
        self.predict_markers_snapshots = ti.Vector.field(
            2,
            dtype=ti.f32,
            shape=(SYSTEM_PARAMS.contact.num_training_trajectories, self.vitactip.num_markers),
            needs_grad=False,
        )
        self.virtual_markers_snapshots = ti.Vector.field(
            2,
            dtype=ti.f32,
            shape=(SYSTEM_PARAMS.contact.num_training_trajectories, self.vitactip.num_markers),
            needs_grad=False,
        )
        self.ground_truth_labels = ti.field(
            dtype=int, shape=(SYSTEM_PARAMS.contact.num_training_trajectories,), needs_grad=False
        )

    def set_up_pid(self):
        self.pos_error_sum = ti.Vector.field(3, dtype=float, shape=(), needs_grad=False)
        self.prev_pos_error = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.current_target_idx = ti.field(dtype=int, shape=(), needs_grad=False)
        self.current_target_idx[None] = 0
        self.ori_error_magnitude_degrees = ti.field(
            dtype=float, shape=(), needs_grad=False
        )
        self.dwell_frames = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_frames[None] = SYSTEM_PARAMS.contact.dwell_frames
        self.dwell_counter = ti.field(dtype=int, shape=(), needs_grad=False)
        self.dwell_counter[None] = 0
        self.is_dwelling = ti.field(dtype=int, shape=(), needs_grad=False)
        self.is_dwelling[None] = 0
        self.last_target_reached = ti.field(dtype=int, shape=(), needs_grad=False)
        self.last_target_reached[None] = 0
        self.frames_since_last_target_reached = ti.field(
            dtype=int, shape=(), needs_grad=False
        )
        self.frames_since_last_target_reached[None] = 0
        self.mesh_needs_to_be_saved = ti.field(dtype=int, shape=(), needs_grad=False)
        self.mesh_needs_to_be_saved[None] = 0

    def set_up_initial_positions_and_trajectory_first_init_only(self):
        self.phantom_closest_vertex = SYSTEM_PARAMS_COMPUTED.phantom_closest_vertex
        self.phantom_centroid_pose = SYSTEM_PARAMS_COMPUTED.phantom_centroid_pose
        self.vitactip_tip_pose = SYSTEM_PARAMS_COMPUTED.vitactip_tip_pose
        self.tactile_sensor_initial_position = ti.Vector.field(
            3, dtype=ti.f32, shape=1, needs_grad=False
        )
        self.phantom_initial_position = ti.Vector.field(
            3, dtype=ti.f32, shape=1, needs_grad=False
        )
        self.tumour_present_ground_truth_label = ti.field(dtype=int, shape=(), needs_grad=False)
        self.tumour_present_ground_truth_label[None] = 0
        self.sim_keypoints_np = -np.ones((5,))
        self.sim_keypoints = ti.field(
            dtype=int, shape=(self.sim_keypoints_np.shape[0],), needs_grad=False
        )
        self.sim_keypoints.from_numpy(self.sim_keypoints_np)
        self.exp_keypoints_np = np.array([
            [-1, -1, 0, 47, 93],
            [-1, -1, 0, 23, 230],
            [-1, -1, 0, 30, 95],
            [-1, -1, 0, 39, 133]
        ], dtype=int)
        self.exp_keypoints = ti.Vector.field(
            5, dtype=int, shape=(self.exp_keypoints_np.shape[0],), needs_grad=False
        )
        self.exp_keypoints.from_numpy(self.exp_keypoints_np)

        self.exp_vein_ixs = np.array([
            103,
            179
        ], dtype=int)
        self.exp_vein_2d_coords = np.array([
            [935, 881],
            [1197, 899]
        ], dtype=float)
        foo = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface
        bar = SYSTEM_PARAMS.trajectory.press_depth_slide
        self.exp_vein_3d_coords_E_np = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
            ps=self.exp_vein_2d_coords,
            dist_lens_to_plane=foo-bar,
        )
        self.exp_vein_3d_coords_E = ti.Vector.field(
            3, dtype=float, shape=(self.exp_vein_3d_coords_E_np.shape[0],), needs_grad=False
        )
        self.exp_vein_3d_coords_E.from_numpy(self.exp_vein_3d_coords_E_np)
        self.vein_speed_E_np = (
            (self.exp_vein_3d_coords_E_np[1] - self.exp_vein_3d_coords_E_np[0])
            /
            (self.exp_vein_ixs[1] - self.exp_vein_ixs[0])
        )
        self.vein_speed_E_np.reshape((1, 3))
        self.vein_speed_E = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.vein_speed_E.from_numpy(self.vein_speed_E_np.reshape(3))
        self.exp_vein_3d_coords_E_all_np = -np.ones(shape=(self.exp_keypoints_np[1][4]+1, 3), dtype=float)
        for i in range(self.exp_keypoints_np[1][3], self.exp_keypoints_np[1][4]+1):
            self.exp_vein_3d_coords_E_all_np[i] = (
                self.exp_vein_3d_coords_E_np[0]
                + (i - self.exp_vein_ixs[0]) * self.vein_speed_E_np
            )
        self.exp_vein_3d_coords_E_all = ti.Vector.field(
            3, dtype=float, shape=(self.exp_vein_3d_coords_E_all_np.shape[0],), needs_grad=False
        )
        self.exp_vein_3d_coords_E_all.from_numpy(self.exp_vein_3d_coords_E_all_np)
        validation_point_2d = np.array([
            [1028, 947]
        ])
        foo = SYSTEM_PARAMS.geometry.distance_from_camera_lens_to_outer_shell_surface
        bar = SYSTEM_PARAMS.trajectory.press_depth_slide
        self.validation_point_3d_E_np = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
            ps=validation_point_2d,
            dist_lens_to_plane=foo-bar,
        )
        with open(SYSTEM_PARAMS.files.validation_point_E, "wb") as f:
            pickle.dump(self.validation_point_3d_E_np, f)
        self.validation_point_3d_E = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.validation_point_3d_E.from_numpy(self.validation_point_3d_E_np.reshape(3))
        self.validation_point_3d_A = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.num_trajectories = SYSTEM_PARAMS.meta.num_trajectories
        self.max_num_trajectory_points = SYSTEM_PARAMS.meta.max_num_trajectory_points
        self.trajectories = ti.Vector.field(7, dtype=float, shape=(
            self.num_trajectories,
            self.max_num_trajectory_points
        ), needs_grad=False)
        self.trajectory_lengths = ti.field(dtype=int, shape=(self.num_trajectories,), needs_grad=False)
    
    def set_up_trajectories_and_phantom_states(self):
        x, y, z = self.vitactip_tip_pose[:3]
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        _dr = -SYSTEM_PARAMS.geometry.camera_rotation_angle
        dr = R.from_euler(seq="xyz", angles=[0, 0, _dr], degrees=True)
        og_r = og_r * dr
        twist_1_offset = R.from_euler(seq="xyz", angles=[30, 0, 0], degrees=True)
        twist_2_offset = R.from_euler(seq="xyz", angles=[0, 0, -45], degrees=True)
        # slide_offset = R.from_euler(seq="xyz", angles=[0, 0, 180], degrees=True)
        twist_1_base_offset = R.from_euler(seq="xyz", angles=[0, 0, 90], degrees=True)
        slide_r = og_r
        twist_1_r = og_r * twist_1_base_offset
        twist_1 = twist_1_r * twist_1_offset
        twist_2 = og_r * twist_2_offset
        press_depth_surface = SYSTEM_PARAMS.geometry.gap
        press_depth_0 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_0
        press_depth_1 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_1
        press_depth_2 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_2
        press_depth_3 = press_depth_surface + SYSTEM_PARAMS.trajectory.press_depth_3
        slide_dist = SYSTEM_PARAMS.trajectory.slide_distance
        self.trajectory_names = [
            'press (no vein)',
            'slide (vein)',
            'twist-y (no vein)',
            'twist-z (no vein)',
        ]
        trajectories_python_array = [
            [
                [x, y, z, *og_r.as_quat()],
                [x, y, z, *og_r.as_quat()],
                [x, y, z - press_depth_surface, *og_r.as_quat()],

                [x, y, z - press_depth_0, *og_r.as_quat()],

                [x, y, z - press_depth_surface, *og_r.as_quat()],
            ],
            [
                [x, y, z, *slide_r.as_quat()],
                [x, y, z, *slide_r.as_quat()],
                [x, y, z - press_depth_surface, *slide_r.as_quat()],

                [x, y, z - press_depth_1, *slide_r.as_quat()],
                [x + slide_dist, y, z - press_depth_1, *slide_r.as_quat()],
            ],
            [
                [x, y, z, *twist_1_r.as_quat()],
                [x, y, z, *twist_1_r.as_quat()],
                [x, y, z - press_depth_surface, *twist_1_r.as_quat()],

                [x, y, z - press_depth_2, *twist_1_r.as_quat()],
                [x, y, z - press_depth_2, *twist_1.as_quat()],
            ],
            [
                [x, y, z, *og_r.as_quat()],
                [x, y, z, *og_r.as_quat()],
                [x, y, z - press_depth_surface, *og_r.as_quat()],

                [x, y, z - press_depth_3, *og_r.as_quat()],
                [x, y, z - press_depth_3, *twist_2.as_quat()],
            ]
        ]
        self.set_trajectories(trajectories_python_array)
        cz_offset = SYSTEM_PARAMS.geometry.phantom_z_length / 2 - SYSTEM_PARAMS.geometry.vein.depth_beneath_surface
        self.state_dicts = [
            [],
            [
                {
                    'cx': 0,
                    'cy': 0,
                    'cz': cz_offset,
                    'theta': SYSTEM_PARAMS.geometry.vein.theta,
                    'h': SYSTEM_PARAMS.geometry.vein.h,
                    'r': SYSTEM_PARAMS.geometry.vein.r
                }
            ],
            [],
            [],
        ]
        assert(len(trajectories_python_array) == len(self.state_dicts))

    def generate_trajectories(self):
        self.trajectory_names = [
            'press',
            'twist_z',
            'twist_x',
            'slide',
        ]
        trajectories_python_array = [
            self.get_press_trajectory(),
            self.get_twist_z_trajectory(),
            self.get_twist_x_trajectory(),
            self.get_slide_trajectory(),
        ]
        self.set_trajectories(trajectories_python_array)

        # self.state_dicts = []
        # for i in range(len(trajectories_python_array)):
        #     self.state_dicts.append(
        #         self.generate_random_state_dicts()
        #     )    
    
    def get_vitactip_orientation(self):
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        _dr = -SYSTEM_PARAMS.geometry.camera_rotation_angle
        dr = R.from_euler(seq="xyz", angles=[0, 0, _dr], degrees=True)
        og_r = og_r * dr
        return og_r

    def get_random_slide_params(self):
        quat = self.vitactip_tip_pose[3:]
        og_r = R.from_quat(quat)
        _dr = -SYSTEM_PARAMS.geometry.camera_rotation_angle
        dr = R.from_euler(seq="xyz", angles=[0, 0, _dr], degrees=True)
        og_r = og_r * dr
        xr = NP_RNG.uniform(-10, 10)
        yr = NP_RNG.uniform(-10, 10)
        zr = NP_RNG.uniform(0, 60)
        rand_r = R.from_euler(seq="xyz", angles=[0, 0, zr], degrees=True)
        og_r = og_r * rand_r
        slide_r = og_r
        srq = slide_r.as_quat()

        press_depth_surface = SYSTEM_PARAMS.geometry.gap
        press_depth_1 = SYSTEM_PARAMS.trajectory.press_depth_slide
        if True:
            k_0 = SYSTEM_PARAMS.trajectory.press_depth_offset_0
            k_1 = SYSTEM_PARAMS.trajectory.press_depth_offset_1
            press_depth_rand = NP_RNG.uniform(-k_0, k_1)
            press_depth_1 = press_depth_1 + press_depth_rand
        
        return (
            srq,
            press_depth_surface,
            press_depth_1
        )

    def get_random_grid_search_trajectory(self):
        (
            srq,
            press_depth_surface,
            press_depth_1
        ) = self.get_random_slide_params()
        _, _, z = self.vitactip_tip_pose[:3]
        x = self.sensor_x_range_world[0]
        y = self.sensor_y_range_world[0]
        dx = self.sensor_x_range_world[1] - self.sensor_x_range_world[0]
        dy = self.sensor_y_range_world[1] - self.sensor_y_range_world[0]
        r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        d_single = NP_RNG.uniform(0.5 * r, 2 * r)
        trajectory = [
            [x, y, z, *srq],
            [x, y, z - press_depth_surface, *srq],
            [x, y, z - press_depth_1, *srq],
        ]
        return trajectory
        # 0,1,2,3,4!,5,6,7!,8,9,10!,11,12,13!,14,15,16!
        # x = 4 + 3*k, k >= 0
        # x >= 4 and (x - 4) % 3 == 0
        # ts >= 4 and (ts - 4) % 3 == 0
        xy_dirs = [
            [0, 1],
            [1, 0],
            [0, -1],
            [1, 0]
        ]
        xy_i = 0
        while True:
            a, b, c = trajectory[-1][:3]
            if (
                a > self.sensor_x_range_world[1]
                or len(trajectory) == self.trajectories.shape[1]
            ):
                break
            x_dir, y_dir = xy_dirs[xy_i]
            a2 = a + x_dir * d_single
            b2 = b + y_dir * dy
            if xy_i == 0 or xy_i == 2:
                trajectory.append(
                    [a2, (b+b2)/2, c, *srq]
                )
            trajectory.append(
                [a2, b2, c, *srq]
            )
            xy_i += 1
            xy_i %= 4
        return trajectory

    def get_fully_random_trajectory(self):
        (
            srq,
            press_depth_surface,
            press_depth_1
        ) = self.get_random_slide_params()
        _, _, z = self.vitactip_tip_pose[:3]
        x = self.sensor_x_range_world[0]
        y = self.sensor_y_range_world[0]
        
        # Initial press-down motion
        trajectory = [
            [x, y, z, *srq],
            [x, y, z - press_depth_surface, *srq],
            [x, y, z - press_depth_1, *srq],
        ]
        return trajectory
        
        # Calculate maximum possible magnitude based on sensor bounds
        x_min, x_max = self.sensor_x_range_world
        y_min, y_max = self.sensor_y_range_world
        max_dx = x_max - x_min
        max_dy = y_max - y_min
        max_magnitude = min(max_dx, max_dy) / 2  # Conservative estimate
        
        # Generate remaining trajectory points using polar coordinates
        current_x, current_y = x, y
        while len(trajectory) < self.trajectories.shape[1]:
            magnitude = NP_RNG.uniform(0, max_magnitude)
            
            # Keep trying angles until we find one that keeps point in bounds
            while True:
                angle = NP_RNG.uniform(0, 2 * math.pi)
                new_x = current_x + magnitude * math.cos(angle)
                new_y = current_y + magnitude * math.sin(angle)
                
                # Check if new point is within bounds
                if (x_min <= new_x <= x_max and 
                    y_min <= new_y <= y_max):
                    trajectory.append(
                        [new_x, new_y, z - press_depth_1, *srq]
                    )
                    current_x, current_y = new_x, new_y
                    break
                    
        return trajectory
    
    def get_straight_line_slide_trajectory(self):
        (
            srq,
            press_depth_surface,
            press_depth_1
        ) = self.get_random_slide_params()
        x, y, z = self.vitactip_tip_pose[:3]
        r = SYSTEM_PARAMS.geometry.sensor_xy_radius
        y_span = SYSTEM_PARAMS.geometry.phantom_y_length
        y_final = y+r+y_span+r
        trajectory = [
            [x, y, z, *srq],
            [x, y_final, z, *srq],
        ]
        return trajectory
    
    def get_press_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        x = cvx+dx/2
        y = cvx+dy/2
        z = cvz+dz+self.gap
        press_depth = 0.004*self.dist_sf
        ori = ori.as_quat()
        trajectory = [
            [x, y, z, *ori],
            [x, y, z-self.gap, *ori],
            [x, y, z-self.gap-press_depth, *ori],
        ]
        return trajectory
    
    def get_twist_z_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        x = cvx+dx/2
        y = cvx+dy/2
        z = cvz+dz+self.gap
        press_depth = 0.004*self.dist_sf
        angle = 30
        z_rot = R.from_euler(seq="xyz", angles=[0, 0, -angle], degrees=True)
        ori2 = ori * z_rot
        ori = ori.as_quat()
        ori2 = ori2.as_quat()
        trajectory = [
            [x, y, z, *ori],
            [x, y, z-self.gap, *ori],
            [x, y, z-self.gap-press_depth, *ori],
            [x, y, z-self.gap-press_depth, *ori2],
        ]
        return trajectory
    
    def get_twist_x_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        x = cvx+dx/2
        y = cvx+dy/3
        z = cvz+dz+self.gap
        press_depth = 0.002*self.dist_sf
        angle = 20
        z_rot = R.from_euler(seq="xyz", angles=[angle, 0, 0], degrees=True)
        ori2 = ori * z_rot
        ori = ori.as_quat()
        ori2 = ori2.as_quat()
        trajectory = [
            [x, y, z, *ori],
            [x, y, z-self.gap, *ori],
            [x, y, z-self.gap-press_depth, *ori],
            [x, y, z-self.gap-press_depth, *ori2],
        ]
        return trajectory
    
    def get_slide_trajectory(self):
        ori = self.get_vitactip_orientation()
        cvx, cvy, cvz = self.phantom_closest_vertex
        dx, dy, dz = self.phantom_dimensions
        press_depth = 0.003*self.dist_sf
        r = self.sensor_r
        y_span = SYSTEM_PARAMS.geometry.phantom_y_length
        x = cvx+dx/2
        y = cvy-r
        z = cvz+dz-press_depth
        y2 = y+r+y_span+r

        cx = cvx+dx/2
        cy = cvy+dy/2
        r21 = dy/2+r/2
        r22 = dy/2

        theta_degrees = -90+NP_RNG.uniform(-15, 15)
        theta = np.deg2rad(theta_degrees)
        
        # Get points on circle
        x1 = cx + r21 * np.cos(theta)
        y1 = cy + r21 * np.sin(theta)
        x2 = cx + r22 * np.cos(theta + np.pi)
        y2 = cy + r22 * np.sin(theta + np.pi)

        if False:
            xr = NP_RNG.uniform(-5, 5)
            yr = NP_RNG.uniform(-5, 5)
            zr = NP_RNG.uniform(0, 60)
            rand_r = R.from_euler(seq="xyz", angles=[0, 0, zr], degrees=True)
            ori = ori * rand_r

        ori = ori.as_quat()
        trajectory = [
            [x1, y1, z, *ori],
            [x2, y2, z, *ori],
        ]
        return trajectory

    def set_up_initial_positions_state_and_trajectory(self):
        sensor_dome_tip_initial_pose = self.trajectories[self.trajectory_ix[None], 0].to_numpy()
        self.vitactip.set_up_pose(sensor_dome_tip_initial_pose)
        self.tactile_sensor_initial_position[0] = ti.Vector(
            sensor_dome_tip_initial_pose[:3]
        )
        self.phantom_initial_position[0] = ti.Vector(self.phantom_centroid_pose[:3])
        self.phantom.initialise_point_cloud()
    
    def set_trajectories(self, trajectories_python_arr):
        # Create a zero-initialized array for padded trajectories
        trajectories_np = np.zeros((self.num_trajectories, self.max_num_trajectory_points, 7), dtype=np.float32)
        
        # Create an array to store the actual length of each trajectory
        trajectory_lengths = np.zeros(self.num_trajectories, dtype=int)
        
        # Fill in the actual trajectory data
        for i, trajectory in enumerate(trajectories_python_arr):
            traj_len = min(len(trajectory), self.max_num_trajectory_points)
            trajectory_lengths[i] = traj_len
            trajectories_np[i, :traj_len] = np.array(trajectory[:traj_len])
        
        self.trajectories.from_numpy(trajectories_np)
        self.trajectory_lengths.from_numpy(trajectory_lengths)
    
    def generate_random_state_dicts(self):
        return []
        if not self.generate_tumour:
            return []

        state_dicts = []
        num_veins = SYSTEM_PARAMS.meta.max_num_veins
        placed_cy_values = []
        min_separation = SYSTEM_PARAMS.geometry.min_vein_separation
        for i in range(num_veins):
            theta_rand = NP_RNG.uniform(-10, 10)
            cz_offset = SYSTEM_PARAMS.geometry.phantom_z_length / 2 - SYSTEM_PARAMS.geometry.vein.depth_beneath_surface
            cx = self.sensor_x_range_phantom[0]

            while True:
                cy = NP_RNG.uniform(*self.sensor_y_range_phantom)
                valid_position = True
                for prev_cy in placed_cy_values:
                    if abs(cy - prev_cy) < min_separation:
                        valid_position = False
                        break
                if valid_position:
                    placed_cy_values.append(cy)
                    break
                
            h = SYSTEM_PARAMS.geometry.vein.h
            state_dict = {
                'cx': cx,
                'cy': cy,
                'cz': cz_offset,
                'theta': SYSTEM_PARAMS.geometry.vein.theta + theta_rand,
                'h': h,
                'r': SYSTEM_PARAMS.geometry.vein.r
            }
            state_dicts.append(state_dict)
            
        print('placed_cy_values')
        print(placed_cy_values)
        return state_dicts

    @ti.kernel
    def reset_exp_sim_traj(self):
        self.marker_position_exp.fill(-1)
        self.sim_keypoints.fill(-1)
        self.sim_to_exp_markers.fill(-1)
        self.exp_to_sim_markers.fill(-1)
        self.exp_marker_points.fill(-1)
        self.sim_markers_deformed_filtered.fill(-1)
        self.sim_markers_deformed_filtered_z.fill(-1)
        self.sim_markers_deformed_z.fill(-1)
        self.cur_exp_frame.fill(-1)
        self.vein_ix_base.fill(-1)
        self.vein_ix_offset.fill(-1)
        self.vein_exp_vis.fill(0)
        self.vein_exp_vis_all.fill(0)
        self.interpolation_valid.fill(1)

    def reset_pid_controller(self):
        self.pos_error_sum.fill(0)
        self.prev_pos_error.fill(0)
        self.current_target_idx[None] = 0
        self.dwell_counter[None] = 0
        self.is_dwelling[None] = 0
        self.last_target_reached[None] = 0
        self.frames_since_last_target_reached[None] = 0

    def update(self, f):
        self.phantom.compute_trial_deformation_gradient(f)
        self.phantom.svd_of_trial_deformation_gradient(f)
        self.phantom.p2g(f)
        self.vitactip.update_internal_forces(f)
        self.phantom.check_grid_occupy(f)
        self.detect_collisions(f)
        self.resolve_collisions(f)
        self.phantom.grid_op(f)
        self.phantom.g2p(f)
        self.vitactip.update_external_forces(f)

    def reset_state(self):
        self.vitactip.reset_state()
        self.phantom.reset_state()
        self.triangle_ix_contact_0.fill(-1)
        self.triangle_ix_contact_1.fill(-1)
        self.triangle_ix_contact_2.fill(-1)
        self.collision2_contact_flat.fill(0)
        if False:
            self.coulomb_friction_coeff.fill(0)
            self.normal_stiffness.fill(0)
            self.tangential_stiffness.fill(0)
            self.normal_damping.fill(0)

    @ti.func
    def dist(self, a, b) -> ti.f32:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dx /= SYSTEM_PARAMS.fisheye_model.target_image_width
        dy /= SYSTEM_PARAMS.fisheye_model.target_image_height
        squared_error = dx * dx + dy * dy
        return ti.sqrt(squared_error)

    @ti.func
    def calculate_contact_force(
        self, 
        signed_distance, 
        surface_normal, 
        relative_velocity, 
        contact_pair_ix,
    ):
        i = contact_pair_ix
        tangential_force = ti.Vector([0.0, 0.0, 0.0])
        tangential_velocity = ti.Vector([0.0, 0.0, 0.0])
        contact_relative_velocity = relative_velocity
        normal_velocity_magnitude = ti.max(
            surface_normal.dot(contact_relative_velocity), 0
        )
        normal_force = (
            -(
                self.normal_stiffness[i]
                + self.normal_damping[i] * normal_velocity_magnitude
            )
            * signed_distance
            * surface_normal
        )
        tangential_velocity = (
            contact_relative_velocity
            - surface_normal.dot(contact_relative_velocity) * surface_normal
        )
        tangential_velocity_magnitude = tangential_velocity.norm(
            SYSTEM_PARAMS.contact.norm_eps
        )
        if (
            tangential_velocity_magnitude
            > SYSTEM_PARAMS.contact.tangential_velocity_detection_threshold
        ):
            tangential_force = (
                1.0
                * (tangential_velocity / tangential_velocity_magnitude)
                * ti.min(
                    self.tangential_stiffness[i] * tangential_velocity_magnitude,
                    self.coulomb_friction_coeff[i]
                    * normal_force.norm(SYSTEM_PARAMS.contact.norm_eps),
                )
            )
        total_contact_force = normal_force + tangential_force
        return total_contact_force, normal_force, tangential_force

    @ti.kernel
    def check_collision0(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                closest_triangle_ix = self.vitactip.find_closest(
                    grid_node_position, frame
                )
                self.triangle_ix_contact_0[frame, i, j, k] = closest_triangle_ix
    
    @ti.kernel
    def check_collision1(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                closest_triangle_ix = self.vein.find_closest(
                    grid_node_position
                )
                self.triangle_ix_contact_1[frame, i, j, k] = closest_triangle_ix
    
    @ti.kernel
    def check_collision2(self, frame: ti.i32):
        for i in range(self.vein.particles_A.shape[0]):
            point = self.vein.particles_A[i]
            closest_triangle_ix = self.vitactip.find_closest(point, frame)
            self.triangle_ix_contact_2[frame, i] = closest_triangle_ix

    @ti.kernel
    def collision0(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                grid_node_velocity = self.phantom.grid_node_momentum_in[
                    frame, i, j, k
                ] / (
                    self.phantom.grid_node_mass[frame, i, j, k]
                    + SYSTEM_PARAMS.phantom.mass_eps
                )
                closest_triangle_ix = self.triangle_ix_contact_0[frame, i, j, k]
                if closest_triangle_ix != -1:
                    (
                        penetration_depth,
                        surface_normal,
                        relative_velocity,
                        is_in_contact,
                    ) = self.vitactip.find_sdf(
                        grid_node_position,
                        grid_node_velocity,
                        closest_triangle_ix,
                        frame,
                    )
                    if is_in_contact:
                        total_contact_force, _, _ = self.calculate_contact_force(
                            penetration_depth,
                            -1*surface_normal,
                            -1*relative_velocity,
                            contact_pair_ix=0,
                        )
                        self.phantom.update_contact_impulse(
                            total_contact_force, frame, i, j, k
                        )
                        self.vitactip.update_contact_force(
                            closest_triangle_ix, -1*total_contact_force, frame
                        )
    
    @ti.kernel
    def collision1(self, frame: ti.i32):
        for i, j, k in ti.ndrange(
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_x,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_y,
            SYSTEM_PARAMS_COMPUTED.phantom.n_grid_z,
        ):
            if self.phantom.grid_occupy[frame, i, j, k] == 1:
                grid_node_position = ti.Vector(
                    [
                        (i + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (j + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                        (k + 0.5) * SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
                    ]
                )
                grid_node_velocity = self.phantom.grid_node_momentum_in[
                    frame, i, j, k
                ] / (
                    self.phantom.grid_node_mass[frame, i, j, k]
                    + SYSTEM_PARAMS.phantom.mass_eps
                )
                closest_triangle_ix = self.triangle_ix_contact_1[frame, i, j, k]
                if closest_triangle_ix != -1:
                    (
                        penetration_depth,
                        surface_normal,
                        relative_velocity,
                        is_in_contact,
                    ) = self.vein.find_sdf(
                        grid_node_position,
                        grid_node_velocity,
                        closest_triangle_ix,
                    )
                    if is_in_contact:
                        total_contact_force, _, _ = self.calculate_contact_force(
                            penetration_depth,
                            -1*surface_normal,
                            -1*relative_velocity,
                            contact_pair_ix=1,
                        )
                        self.phantom.update_contact_impulse(
                            total_contact_force, frame, i, j, k
                        )
    
    @ti.kernel
    def collision2(self, frame: ti.i32):
        for i in range(self.vein.particles_A.shape[0]):
            point = self.vein.particles_A[i]
            velocity = ti.Vector([0.0, 0.0, 0.0])
            closest_triangle_ix = self.triangle_ix_contact_2[frame, i]
            if closest_triangle_ix != -1:
                (
                    penetration_depth,
                    surface_normal,
                    relative_velocity,
                    is_in_contact,
                ) = self.vitactip.find_sdf(
                    point,
                    velocity,
                    closest_triangle_ix,
                    frame,
                )
                if is_in_contact:
                    self.collision2_contact_flat[None] = 1
                    total_contact_force, _, _ = self.calculate_contact_force(
                        penetration_depth, 
                        -1*surface_normal, 
                        -1*relative_velocity,
                        contact_pair_ix=2,
                    )
                    self.vitactip.update_contact_force(
                        closest_triangle_ix, 
                        -1*total_contact_force, 
                        frame,
                    )

    def copy_frame(self):
        self.vitactip.copy_frame(SYSTEM_PARAMS.contact.num_sub_frames - 1, 0)
        self.phantom.copy_frame(SYSTEM_PARAMS.contact.num_sub_frames - 1, 0)

    def memory_to_cache(self, t):
        self.vitactip.memory_to_cache(t)
        self.phantom.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.vitactip.memory_from_cache(t)
        self.phantom.memory_from_cache(t)

    def pid_controller_1(self):
        self.vitactip.compute_current_orientation()
        current_ori = R.from_quat(self.vitactip.R_BA_quat.to_numpy())
        target = self.trajectories[self.trajectory_ix[None], self.current_target_idx[None]].to_numpy()
        target_ori = R.from_quat(target[3:])
        ori_error = target_ori * current_ori.inv()
        current_angle_radians = ori_error.magnitude()
        current_axis = ori_error.as_rotvec() / (
            current_angle_radians
            if current_angle_radians > SYSTEM_PARAMS.contact.pid_angle_eps
            else 1.0
        )
        if np.isclose(current_angle_radians % (2 * np.pi), 0) or np.isclose(
            current_angle_radians % (2 * np.pi), 2 * np.pi
        ):
            current_axis = np.array([1.0, 0.0, 0.0])
        time_duration = self.dt[None] * (
            SYSTEM_PARAMS.contact.num_sub_frames - 1
        )
        rotation_per_second = (
            current_angle_radians
            * SYSTEM_PARAMS.contact.pid_orientation_kp
            / time_duration
        )
        if rotation_per_second > np.deg2rad(
            SYSTEM_PARAMS.contact.pid_max_rotation_per_second_degrees
        ):
            ori_control = R.from_rotvec(
                current_axis
                * np.deg2rad(SYSTEM_PARAMS.contact.pid_max_rotation_per_second_degrees)
                * time_duration
            )
        else:
            ori_control = R.from_rotvec(
                current_axis * rotation_per_second * time_duration
            )
        ori_control_quat = ori_control.as_quat()
        self.vitactip.R_A_quat.from_numpy(ori_control_quat.reshape(4))
        self.ori_error_magnitude_degrees[None] = np.rad2deg(current_angle_radians)

    @ti.kernel
    def pid_controller_2(self, ts: ti.i32):
        current_pos = self.vitactip.vertices_undeformed_A[0, self.vitactip.tip_ix[None]]
        target = self.trajectories[self.trajectory_ix[None], self.current_target_idx[None]]
        target_pos = ti.Vector([target[0], target[1], target[2]])
        pos_error = target_pos - current_pos
        pos_error_magnitude = pos_error.norm()
        if self.last_target_reached[None] == 1:
            self.frames_since_last_target_reached[None] += 1
        if (
            self.last_target_reached[None] == 0
            and self.is_dwelling[None] == 0
            and pos_error_magnitude < SYSTEM_PARAMS.contact.pid_position_tolerance
            and self.ori_error_magnitude_degrees[None]
            < SYSTEM_PARAMS.contact.pid_orientation_tolerance
        ):
            self.is_dwelling[None] = 1
            self.dwell_counter[None] = 0
            self.sim_keypoints[self.current_target_idx[None]] = ts
            self.mesh_needs_to_be_saved[None] = 1
            # print(
            #     f"target {self.current_target_idx[None]} ({target}) reached at time step {ts}!"
            # )
        target_reached_no_control = False
        if self.is_dwelling[None] == 1:
            self.dwell_counter[None] += 1
            if self.dwell_counter[None] >= self.dwell_frames[None]:
                self.is_dwelling[None] = 0
                if self.current_target_idx[None] < self.trajectory_lengths[self.trajectory_ix[None]] - 1:
                    self.current_target_idx[None] += 1
                    self.pos_error_sum[None] = ti.Vector([0.0, 0.0, 0.0])
                    self.prev_pos_error[None] = ti.Vector([0.0, 0.0, 0.0])
                    target_reached_no_control = True
                else:
                    self.last_target_reached[None] = 1
        if self.is_dwelling[None] == 1 or target_reached_no_control:
            self.vitactip.translation_A[None] = ti.Vector([0.0, 0.0, 0.0])
            self.vitactip.R_A_quat[None] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        else:
            self.pos_error_sum[None] += pos_error
            pos_derivative = pos_error - self.prev_pos_error[None]
            self.prev_pos_error[None] = pos_error
            pos_control = (
                SYSTEM_PARAMS.contact.pid_kp * pos_error
                + SYSTEM_PARAMS.contact.pid_ki * self.pos_error_sum[None]
                + SYSTEM_PARAMS.contact.pid_kd * pos_derivative
            )
            max_speed_pos = SYSTEM_PARAMS.contact.pid_max_speed_translation
            pos_control_norm = pos_control.norm()
            if pos_control_norm > max_speed_pos:
                pos_control = pos_control / pos_control_norm * max_speed_pos
            if SYSTEM_PARAMS.meta.enable_pid_controller == 1:
                self.vitactip.translation_A[None] = pos_control
            else:
                self.vitactip.translation_A[None] = ti.Vector([0.0, 0.0, 0.0])
                self.vitactip.R_A_quat[None] = ti.Vector([0.0, 0.0, 0.0, 1.0])

    def pid_controller_3(self):
        self.vitactip.R_A.from_numpy(
            R.from_quat(self.vitactip.R_A_quat.to_numpy()).as_matrix().reshape(3,3)
        )

    def clear_temp_images(self):
        folders = [
            SYSTEM_PARAMS.files.training_data_vein_full_folder,
            SYSTEM_PARAMS.files.training_data_contact_folder,
            SYSTEM_PARAMS.files.training_data_markers_folder,
            SYSTEM_PARAMS.files.training_data_segmentation_mask_folder,
        ]
        self.clear_training_data_folders_helper(folders)

    def clear_npz(self):
        folders = [
            SYSTEM_PARAMS.files.dataset_root
        ]
        self.clear_training_data_folders_helper(folders)

    def clear_training_data_folders_helper(self, folders):
        for folder in folders:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    @ti.kernel
    def copy_vitactip_vertices(self):
        for i in range(self.vitactip.vertices_deformed_A.shape[1]):
            point = self.vitactip.vertices_deformed_A[
                self.num_sub_frames-1,
                i
            ]
            self.vitactip_vertices_temp[i] = point

    def record_vitactip_mesh(self):
        self.copy_vitactip_vertices()
        self.all_points.append(
            self.vitactip_vertices_temp.to_numpy()
        )
    
    def write_vitactip_mesh_to_file(self):
        all_points = np.stack(self.all_points, axis=0)
        path = SYSTEM_PARAMS.files.vitactip_mesh_npz
        np.savez(
            path,
            all_points=all_points,
        )
        self.all_points = []

    def record_training_data_point(self):
        w = int(SYSTEM_PARAMS.fisheye_model.target_image_width)
        h = int(SYSTEM_PARAMS.fisheye_model.target_image_height)

        cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        r = SYSTEM_PARAMS.fisheye_model.circle_radius

        self.move_og_resolution()
        markers = self.sim_markers_deformed.to_numpy()
        self.move_ti_resolution()
        markers[:, 1] = h-markers[:, 1]
        self.marker_data.append(markers)
        markers_img = np.zeros((h, w), dtype=np.uint8)
        for point in markers:
            x, y = int(point[0]), int(point[1])
            cv2.circle(markers_img, (x, y), radius=1, color=255, thickness=-1)
        markers_mask = SyntheticImageGenerator.compute_mask(h, w, markers)

        vein = self.vein_2d_projection.to_numpy()
        vein[:, :, 1] = h-vein[:, :, 1]
        # vein_counts = self.vein_counts.to_numpy()
        # vein_python_arr = []
        # for i in range(vein.shape[0]):
        #     num_points = vein_counts[i]
        #     single_vein_points = vein[i, :num_points]
        #     vein_python_arr.append(
        #         single_vein_points
        #     )
        # vein = vein_python_arr
        # vein_polyline_python_arr = []
        # for i in range(len(vein)):
        #     single_vein = vein[i]
        #     single_vein = SyntheticImageGenerator.filter_using_mask(markers_mask, single_vein)
        #     polyline_points = SyntheticImageGenerator.fit_polynomial(single_vein)
        #     vein_polyline_python_arr.append(polyline_points)
        # vein_polyline_np, vein_polyline_mask = SyntheticImageGenerator.create_padded_array_with_mask(
        #     vein_polyline_python_arr, 
        #     k=SYSTEM_PARAMS.meta.polyline_num
        # )
        if self.collision2_contact_flat[None] == 1 and 2 in self.collision_ixs:
            vein_mask = np.ones(shape=(vein.shape[0], vein.shape[1]), dtype=bool)
        else:
            vein_mask = np.zeros(shape=(vein.shape[0], vein.shape[1]), dtype=bool)
        self.vein_polyline_data.append(vein)
        self.vein_polyline_mask_data.append(vein_mask)

        target_id_arr = np.array([
            self.current_target_idx[None]
        ])
        self.target_id_data.append(target_id_arr)

        # Sensor pose per recorded frame: the rigid transform from the sensor
        # body frame B (origin at the dome tip, z along the sensor axis) to the
        # world frame A. This is the simulated twin of the robot end-effector
        # pose the real datasets carry (`frames_poses.npz` / `poses`), and it is
        # what lets a simulated trajectory be reprojected onto the phantom plane
        # by the same 2D->3D->2D route as the real ones (see
        # data_analysis/experiment/vessel_map.py). Stored under its own key,
        # `T_BA`, rather than `poses`, so cnn/dataset.py - which slices a
        # `poses` array by clip and collates it - keeps ignoring it.
        self.pose_data.append(self.vitactip.T_BA.to_numpy())
    
    @staticmethod
    def get_endpoints(points):
        """
        Fit a straight line to points and return the endpoints.
        
        Args:
            points: numpy array of shape (num_points, 2)
            
        Returns:
            numpy array of shape (2, 2) containing the two endpoints
        """
        if len(points) < 2:
            foo = -np.ones(shape=(0, 2), dtype=float)
            return foo
        
        # Convert to numpy array if not already
        points = np.array(points, dtype=np.float64)
        
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Use PCA to find the best fitting line direction
        # The first principal component gives us the line direction
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # The eigenvector corresponding to the largest eigenvalue
        # gives us the direction of the line
        line_direction = eigenvectors[:, -1]
        
        # Project all points onto the line
        # For each point, compute the scalar projection onto the line direction
        projections_scalar = np.dot(centered_points, line_direction)
        
        # Find the indices of points with minimum and maximum projections
        min_idx = np.argmin(projections_scalar)
        max_idx = np.argmax(projections_scalar)
        
        # Compute the actual projected points on the line
        min_projection = centroid + projections_scalar[min_idx] * line_direction
        max_projection = centroid + projections_scalar[max_idx] * line_direction
        
        # Return the two endpoints
        endpoints = np.array([min_projection, max_projection])
        
        return endpoints

    def write_training_data_to_file(self, file_num):
        traj_ix = self.trajectory_ix[None]
        directory = self.training_data_dir_override or SYSTEM_PARAMS.files.dataset_root.format(self.timestamp)
        Path(directory).mkdir(parents=True, exist_ok=True)
        file = SYSTEM_PARAMS.files.dataset_data_point.format(
            file_num
        )
        path = f'{directory}/{file}'

        markers_array, markers_mask = SyntheticImageGenerator.create_padded_array_with_mask(self.marker_data)
        vein_polyline_array = np.array(self.vein_polyline_data)
        vein_polyline_mask = np.array(self.vein_polyline_mask_data)
        target_id_array = np.array(self.target_id_data)
        trajectory_type = np.array([traj_ix], dtype=int)
        pose_array = np.array(self.pose_data, dtype=np.float32)

        np.savez(
            path,
            markers=markers_array,
            markers_mask=markers_mask,
            vein_polyline=vein_polyline_array,
            vein_polyline_mask=vein_polyline_mask,
            target_id_array=target_id_array,
            trajectory_type=trajectory_type,
            # Per-frame sensor pose (frames, 4, 4), see record_training_data_point.
            T_BA=pose_array,
            # The vein is a static rigid body: its centreline in the world frame
            # (50 points, sim units) and its radius, so the map code can draw the
            # true vessel without re-deriving it from any projection. Only the
            # sim <-> world unit conversion is needed downstream
            # (meta.distance_scaling_factor: sim length = real length x 5).
            vein_centreline_A=self.vein.centerline_A.to_numpy(),
            vein_radius=np.array([self.vein.r], dtype=np.float32),
            distance_scaling_factor=np.array(
                [SYSTEM_PARAMS.meta.distance_scaling_factor], dtype=np.float32
            ),
        )
        
        self.marker_data = []
        self.vein_polyline_data = []
        self.vein_polyline_mask_data = []
        self.target_id_data = []
        self.pose_data = []

    def take_2d_markers_snapshot(self, k):
        self.take_snapshot_1(k)
        self.take_snapshot_2(k)

    @ti.kernel
    def take_snapshot_1(self, k: ti.i32):
        for i in range(self.vitactip.num_markers):
            self.predict_markers_snapshots[k, i] = self.vitactip.deformed_markers[i]
            self.virtual_markers_snapshots[k, i] = self.vitactip.undeformed_markers[
                i
            ]

    @ti.kernel
    def take_snapshot_2(self, k: ti.i32):
        self.ground_truth_labels[k] = self.tumour_present_ground_truth_label[None]

    def maybe_save_tactile_sensor_mesh_to_pickle(self, ts):
        if self.mesh_needs_to_be_saved[None] == 1:
            particles = self.vitactip.vertices_deformed_A.to_numpy()[0]
            path = SYSTEM_PARAMS.files.deformed_node_coordinates.format(ts)
            # mesh_snapshots/ is gitignored output, so it is absent in a fresh
            # clone or after a clean restore; create it rather than failing.
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(particles, f)
            self.mesh_needs_to_be_saved[None] = 0
    
    def save_sensor_mesh_to_npz(self):
        particles = self.vitactip.vertices_deformed_A.to_numpy()[0]
        path = SYSTEM_PARAMS.files.sensor_mesh
        np.savez(
            path,
            particles=particles,
        )

    def save_tactile_sensor_mesh_node_mapping_to_pickle(self):
        f2v = self.vitactip.tetrahedra.to_numpy()
        with open(SYSTEM_PARAMS.files.tactile_sensor_f2v, "wb") as f:
            pickle.dump(f2v, f)

    def visualisation_initialise(self):
        self.vein_exp_vis_all = ti.Vector.field(
            3, dtype=float, shape=(self.exp_vein_3d_coords_E_all.shape[0],), needs_grad=False
        )
        self.num_keypoints = 3
        self.key_points = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.num_keypoints,), needs_grad=False
        )
        self.sensor_points = ti.Vector.field(
            3, dtype=float, shape=(self.vitactip.num_vertices), needs_grad=False
        )
        self.healthy_tissue_points = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.phantom.num_particles,),
            needs_grad=False,
        )
        self.vein_2d_projection = ti.Vector.field(
            2,
            dtype=float,
            shape=(
                SYSTEM_PARAMS.meta.max_num_veins,
                self.vein.centerline_A.shape[0],
            ),
            needs_grad=False,
        )
        self.vein_2d_projection_flat = ti.Vector.field(
            2,
            dtype=float,
            shape=(
                self.vein.centerline_A.shape[0],
            ),
            needs_grad=False,
        )
        self.sim_markers_undeformed = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        self.sim_markers_deformed = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        self.sim_markers_deformed_filtered = ti.Vector.field(
            2, dtype=float, shape=(self.marker_position_exp.shape[0],), needs_grad=False
        )
        self.sim_markers_deformed_filtered_z = ti.field(
            dtype=float, shape=(self.marker_position_exp.shape[0],), needs_grad=False
        )
        self.sim_markers_deformed_z = ti.field(
            dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        self.sim_marker_offsets = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers,), needs_grad=False
        )
        self.exp_marker_points = ti.Vector.field(
            2, dtype=float, shape=(self.marker_position_exp.shape[0],), needs_grad=False
        )
        self.arrow_line_vertices = ti.Vector.field(
            2, dtype=float, shape=(self.vitactip.num_markers * 2,), needs_grad=False
        )
        self.clock_arm_points = ti.Vector.field(
            2, dtype=float, shape=(2,), needs_grad=False
        )
        self.clock_arm_points_per_vertex_color = ti.Vector.field(
            3, dtype=ti.f32, shape=(2,), needs_grad=False
        )
        self.tactile_image_resolution = ti.Vector.field(2, dtype=float, shape=(), needs_grad=False)
        self.tactile_image_resolution[None] = ti.Vector([
            SYSTEM_PARAMS.fisheye_model.target_image_width,
            SYSTEM_PARAMS.fisheye_model.target_image_height
        ])
        self.fp_bp = ti.field(dtype=int, shape=(), needs_grad=False)
        self.vein_ix_base = ti.field(dtype=int, shape=(), needs_grad=False)
        self.vein_ix_offset = ti.field(dtype=float, shape=(), needs_grad=False)
        self.vein_exp_vis = ti.Vector.field(
            3, dtype=float, shape=(), needs_grad=False
        )
        self.interpolation_valid = ti.field(
            dtype=int, shape=(), needs_grad=False
        )
        self.interpolation_valid[None] = 1

    @ti.kernel
    def visualisation_reset_scene(self):
        self.healthy_tissue_points.fill(0)
        self.vein_2d_projection.fill(-1)
        self.vein_2d_projection_flat.fill(-1)

    @ti.kernel
    def visualisation_draw_3d_scene(self, f: ti.i32):
        for p in range(self.phantom.num_particles):
            # if self.phantom.grid_node_vein_indices[p] == 0:
            self.healthy_tissue_points[p] = self.phantom.particles_A[f, p]
        for p in range(self.vitactip.num_vertices):
            self.sensor_points[p] = self.vitactip.vertices_deformed_A[f, p]

    @ti.kernel
    def visualisation_project_vein_2d(self):
        for i in range(self.vein.centerline_A.shape[0]):
            point = self.vein.centerline_A[i]
            projection_2d = self.vitactip.project_A_point_2d(point)
            projection_2d[1] = self.tactile_image_resolution[None][1] - projection_2d[1]
            self.vein_2d_projection[0, i] = projection_2d
            self.vein_2d_projection_flat[i] = projection_2d / self.tactile_image_resolution[None]

    @ti.kernel
    def visualisation_prepare_tactile_readout_data_fp(self):
        for i in range(self.vitactip.num_markers):
            undeformed = self.vitactip.undeformed_markers[i]
            deformed = self.vitactip.deformed_markers[i]
            deformed_z = self.vitactip.deformed_markers_z[i]
            undeformed[1] = self.tactile_image_resolution[None][1] - undeformed[1]
            deformed[1] = self.tactile_image_resolution[None][1] - deformed[1]
            undeformed = undeformed / self.tactile_image_resolution[None]
            deformed = deformed / self.tactile_image_resolution[None]
            offset = deformed - undeformed
            self.sim_markers_undeformed[i] = undeformed
            self.sim_markers_deformed[i] = deformed
            self.arrow_line_vertices[i * 2] = undeformed
            self.arrow_line_vertices[i * 2 + 1] = undeformed + offset
            self.sim_markers_deformed_z[i] = deformed_z

            exp_ix = self.sim_to_exp_markers[i]
            if exp_ix != -1:
                self.sim_markers_deformed_filtered[exp_ix] = deformed
                self.sim_markers_deformed_filtered_z[exp_ix] = deformed_z
    
    @ti.kernel
    def move_og_resolution(self):
        for i in range(self.sim_markers_deformed_filtered.shape[0]):
            if abs(self.sim_markers_deformed_filtered[i][0] - (-1.0)) > 1e-6:
                self.sim_markers_deformed_filtered[i] *= self.tactile_image_resolution[None]
        for i in range(self.sim_markers_deformed.shape[0]):
            self.sim_markers_deformed[i] *= self.tactile_image_resolution[None]

    @ti.kernel
    def move_ti_resolution(self):
        for i in range(self.sim_markers_deformed_filtered.shape[0]):
            if abs(self.sim_markers_deformed_filtered[i][0] - (-1.0)) > 1e-6:
                self.sim_markers_deformed_filtered[i] /= self.tactile_image_resolution[None]
        for i in range(self.sim_markers_deformed.shape[0]):
            self.sim_markers_deformed[i] /= self.tactile_image_resolution[None]

    @ti.kernel
    def visualisation_prepare_tactile_readout_data_bp(self):
        for i in range(self.marker_position_exp.shape[0]):
            point = self.marker_position_exp[i]
            if point[0] > 0 and point[1] > 0:
                point[1] = self.tactile_image_resolution[None][1] - point[1]
                self.exp_marker_points[i] = point / self.tactile_image_resolution[None]
            else:
                self.exp_marker_points[i] =  ti.Vector([-1.0, -1.0], dt=float)

    @ti.kernel
    def visualisation_prepare_clock_arm_points(self):
        for i in range(2):
            point = self.vitactip.projection_2d_clock_arms[i]
            point[1] = self.tactile_image_resolution[None][1] - point[1]
            self.clock_arm_points[i] = point / self.tactile_image_resolution[None]

    @staticmethod
    def line_equation(points):
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Calculate the coefficients of the line equation ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        return a, b, c

    @staticmethod
    def vector_point_to_line(a, b, c, p):
        p = np.array(p)
        if len(p.shape) == 1:  # Single point
            numerator = a * p[0] + b * p[1] + c
            denominator = np.sqrt(a * a + b * b)
            
            if abs(denominator) < 1e-10:
                return np.zeros(2)
                
            normal = np.array([a, b]) / denominator
            return -normal * numerator / denominator
        else:  # Multiple points
            numerator = a * p[..., 0] + b * p[..., 1] + c
            denominator = np.sqrt(a * a + b * b)
            
            if abs(denominator) < 1e-10:
                return np.zeros(p.shape)
                
            normal = np.array([a, b]) / denominator
            return -normal[None, :] * (numerator / denominator)[..., None]

    @staticmethod
    def line_point_to_line_pass_through_point(a, b, c, p):
        """
        Compute coefficients (a', b', c') of a line that:
        1. Passes through point p
        2. Is perpendicular to line ax + by + c = 0
        
        Args:
            a, b, c: Coefficients of original line ax + by + c = 0
            p: Point [x, y] that the perpendicular line should pass through
            
        Returns:
            a', b', c': Coefficients of perpendicular line a'x + b'y + c' = 0
        """
        # For line ax + by + c = 0, vector (-b, a) is perpendicular to the line
        # This will be our direction vector for the new line
        a_new = -b  # Use negative b as new a coefficient
        b_new = a   # Use a as new b coefficient
        
        # For the new line to pass through point p:
        # a_new * p[0] + b_new * p[1] + c_new = 0
        # Therefore:
        c_new = -(a_new * p[0] + b_new * p[1])
        
        return a_new, b_new, c_new

    def visualisation_draw_tactile_readout(self):
        self.visualisation_project_vein_2d()
        self.vitactip.extract_clock_arm_2d_projections(SYSTEM_PARAMS.contact.num_sub_frames - 1)
        self.visualisation_prepare_clock_arm_points()
        self.tactile_canvas.set_image(self.bg_image)
        self.visualisation_prepare_tactile_readout_data_fp()
        if (
            self.fp_bp[None] == 1
            and SYSTEM_PARAMS.visualisation.visualise_exp_markers_during_bp == 1
        ):
            self.visualisation_prepare_tactile_readout_data_bp()
            self.tactile_canvas.circles(
                self.exp_marker_points, radius=0.01, color=(0, 1, 0)
            )
        self.tactile_canvas.circles(
            self.sim_markers_deformed, radius=0.01, color=(1, 0, 0)
        )
        if True:
            self.tactile_canvas.circles(
                self.clock_arm_points,
                radius=0.02,
                per_vertex_color=self.clock_arm_points_per_vertex_color,
            )
        if self.collision2_contact_flat[None] == 1 and 2 in self.collision_ixs:
            self.tactile_canvas.circles(
                self.vein_2d_projection_flat,
                radius=0.01,
                color=(0, 0, 1)
            )
        self.tactile_window.show()

    def visualisation_set_up_gui(self):
        # In headless mode no X server is available, so skip window creation
        # entirely. visualisation_update_gui() returns early for the same reason;
        # training-data collection does not depend on rendering.
        # DIFFTACTILE_SNAPSHOT_DIR renders OFFSCREEN instead of skipping: Taichi
        # GGUI's show_window=False needs no X server but still lets
        # window.save_image() capture frames. That is what makes it possible to
        # verify a trajectory visually on a headless machine - see
        # visualisation_update_gui().
        self.snapshot_dir = os.environ.get("DIFFTACTILE_SNAPSHOT_DIR")
        self.snapshot_every = int(os.environ.get("DIFFTACTILE_SNAPSHOT_EVERY", 20))
        show_window = not HEADLESS
        if HEADLESS and not self.snapshot_dir:
            print("DIFFTACTILE_HEADLESS=1: skipping Taichi GGUI window creation")
            self.window = None
            self.tactile_window = None
            return
        if self.snapshot_dir:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            print(
                f"Rendering offscreen; snapshots every {self.snapshot_every} "
                f"timesteps -> {self.snapshot_dir}"
            )
        self.window = ti.ui.Window("high-level camera", (
            int(SYSTEM_PARAMS.visualisation.window_3d_width),
            int(SYSTEM_PARAMS.visualisation.window_3d_height)
        ), show_window=show_window)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((0, 0, 0))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        x, y, z = self.phantom_centroid_pose[:3]
        self.camera.position(x-SYSTEM_PARAMS.visualisation.camera_offset, y, z)
        self.camera.up(0, 0, 1)
        self.camera.lookat(x, y, z)
        self.camera.fov(3)
        self.tactile_window = ti.ui.Window("tactile readout", (
            int(SYSTEM_PARAMS.visualisation.tactile_readout_width),
            int(SYSTEM_PARAMS.visualisation.tactile_readout_height)
        ), show_window=show_window)
        self.tactile_canvas = self.tactile_window.get_canvas()
        self.bg_image = cv2.imread(self.default_photo)
        self.bg_image = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2RGB)
        self.bg_image = cv2.rotate(self.bg_image, cv2.ROTATE_90_CLOCKWISE)
        clock_arm_points_per_vertex_color_npy = np.array(
            [
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=float,
        )
        self.clock_arm_points_per_vertex_color.from_numpy(
            clock_arm_points_per_vertex_color_npy
        )

    def create_transition_array_vectorized(self, n):
        t = np.linspace(0, 1, n)[:, np.newaxis]
        start = np.array([0, 1, 1])
        end = np.array([1, 0, 0])
        return (1 - t) * start + t * end
    
    def visualisation_update_gui(self, ts):
        # Offscreen rendering keeps going when a snapshot directory is set; only
        # a truly headless run with no snapshots skips the work entirely.
        if HEADLESS and not getattr(self, "snapshot_dir", None):
            return
        if self.window is None:
            return
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        self.visualisation_draw_3d_scene(SYSTEM_PARAMS.contact.num_sub_frames - 1)
        self.visualisation_draw_tactile_readout()
        self.scene.particles(
            self.healthy_tissue_points,
            color=(0.0, 0.0, 1.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
        )
        self.phantom.compute_grid_colours()
        # self.scene.particles(
        #     self.phantom.grid_positions,
        #     per_vertex_color=self.phantom.grid_colours,
        #     radius=SYSTEM_PARAMS.visualisation.particle_size_normal*2.5,
        # )
        if 2 in self.collision_ixs:
            self.scene.particles(
                self.vein.particles_A,
                color=(1.0, 1.0, 0.0),
                radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
            )
        self.scene.particles(
            self.sensor_points,
            color=(0.0, 1.0, 0.0),
            radius=SYSTEM_PARAMS.visualisation.particle_size_normal,
        )
        self.update_vitactip_tip_point()
        self.update_clock_arm_points_3d()
        # self.scene.particles(
        #     self.vitactip_tip_point,
        #     color=(1.0, 0.0, 0.0),
        #     radius=SYSTEM_PARAMS.visualisation.particle_size_keypoint,
        # )
        self.scene.particles(
            self.clock_arm_points_3d,
            per_vertex_color=self.clock_arm_points_per_vertex_color,
            radius=SYSTEM_PARAMS.visualisation.particle_size_keypoint/3,
        )
        self.canvas.scene(self.scene)
        if getattr(self, "snapshot_dir", None):
            # Periodic frames of the 3D scene, named by trajectory and timestep,
            # so a trajectory can be checked by eye afterwards. save_image()
            # works offscreen, which window.show() would not.
            if ts % self.snapshot_every == 0:
                name = self.trajectory_names[self.trajectory_ix[None]]
                # One subdirectory per BO iteration. The filename carries only
                # trajectory and timestep, so without this every iteration would
                # overwrite the previous one's frames and a 10-iteration run
                # would leave only the last. `snapshot_subdir` is set by
                # domain_adaptation(); paths with no iteration (training-data
                # collection) fall back to the flat directory.
                out_dir = os.path.join(
                    self.snapshot_dir, getattr(self, "snapshot_subdir", "") or ""
                )
                os.makedirs(out_dir, exist_ok=True)
                self.window.save_image(
                    os.path.join(out_dir, f"{name}_ts{ts:04d}.png")
                )
        if not HEADLESS:
            self.window.show()

    def forward_pass_common_part(self, ts):
        self.reset_state()
        # self.set_optimisation_params_from_log()
        self.vitactip.set_control_vel(0)
        self.vitactip.set_vel(0)
        # self.vitactip.set_up_system_params_2()
        # self.phantom.set_stiffness()
        for ss in range(SYSTEM_PARAMS.contact.num_sub_frames - 1):
            self.update(ss)

    def randomise_contact_params(self):
        self.normal_stiffness[2] = NP_RNG.uniform(5e3, 5e4)
        self.normal_damping[2] = NP_RNG.uniform(0, 100)
    
    # Sensor<->vein contact coefficients that are FIXED, not searched. Only
    # normal_stiffness[2] is free (it is the vessel-present model's single
    # parameter); the rest are pinned at the values already in
    # system-params.json, which apply_scaling.py does not touch.
    #
    # tangential_stiffness and coulomb_friction_coeff are BOTH 0, so the
    # tangential branch of calculate_contact_force() contributes nothing:
    # min(k_t * v_t, mu * |F_n|) is identically zero. The vessel therefore acts
    # on the sensor purely NORMALLY - it pushes, it never drags. That is the
    # intended model of a smooth subsurface inclusion under a lubricated
    # interface, and it keeps the vessel-present objective a function of one
    # parameter rather than confounding normal and tangential effects.
    VEIN_NORMAL_DAMPING = 100.0        # system-params.json contact.normal_damping[2]
    VEIN_TANGENTIAL_STIFFNESS = 0.0
    VEIN_COULOMB_FRICTION = 0.0

    def set_contact_params(self):
        """Write the contact coefficients the simulator actually uses.

PAIR 0 (sensor<->phantom) depends on the seam (see
        `phantom_contact_enabled`). Disabled - the default - its coefficients are
        forced to -1, a deliberately impossible value so a reader who finds them
        in a dump knows they are inert rather than fitted; nothing consumes them,
        since pair 0 is then absent from `collision_ixs`. Enabled, they are
        restored from system-params.json.

        PAIR 2 (sensor<->vein) takes its damping/tangential/friction constants
        from the class attributes above; `normal_stiffness[2]` is left alone,
        because that is the one quantity the vessel-present BO fits.
        """
        if phantom_contact_enabled():
            # Pair 0 is ON (sanity-check mode): restore its real coefficients
            # from the config, which are the values the project used before the
            # pair was disabled.
            self.normal_stiffness[0] = SYSTEM_PARAMS.contact.normal_stiffness[0]
            self.normal_damping[0] = SYSTEM_PARAMS.contact.normal_damping[0]
            self.tangential_stiffness[0] = (
                SYSTEM_PARAMS.contact.tangential_stiffness[0]
            )
            self.coulomb_friction_coeff[0] = (
                SYSTEM_PARAMS.contact.coulomb_friction_coeff[0]
            )
        else:
            # Pair 0 is OFF and not resolved at all, so these values are never
            # read. -1 is written anyway - a physically impossible negative
            # stiffness - so that anyone who dumps the fields sees they are
            # inert rather than mistaking them for tuned parameters.
            for field in (self.normal_stiffness, self.normal_damping,
                          self.tangential_stiffness,
                          self.coulomb_friction_coeff):
                field[0] = -1.0
        self.normal_damping[2] = self.VEIN_NORMAL_DAMPING
        self.tangential_stiffness[2] = self.VEIN_TANGENTIAL_STIFFNESS
        self.coulomb_friction_coeff[2] = self.VEIN_COULOMB_FRICTION

    def set_contact_params_from_bo(self):
        """Apply the current BO proposal, whichever stage produced it.

        The two stages fit DISJOINT parameter sets, so each key is applied only
        if the proposal contains it:

          vessel-absent  vitactip_youngs_modulus, vitactip_poissons_ratio
          vessel-present normal_stiffness  (of the sensor<->vein pair)

        Pair-0 contact parameters are deliberately absent from both - that pair
        is disabled (see set_contact_params), so fitting its coefficients would
        be fitting noise.
        """
        p = self.bo.params
        if 'vitactip_youngs_modulus' in p:
            self.vitactip.youngs_modulus[None] = p['vitactip_youngs_modulus']
        if 'vitactip_poissons_ratio' in p:
            self.vitactip.poissons_ratio[None] = p['vitactip_poissons_ratio']
        if 'vitactip_youngs_modulus' in p or 'vitactip_poissons_ratio' in p:
            self.vitactip.set_up_system_params_2()
        # The vessel-present model's only free parameter: the sensor<->vein
        # normal stiffness. Note this is pair index 2, NOT 0.
        if 'normal_stiffness' in p:
            self.normal_stiffness[2] = p['normal_stiffness']
        # Re-assert the fixed constants every time, so a stale value cannot
        # survive from a previous stage or a previous iteration.
        self.set_contact_params()
    
    def print_contact_params(self):
        ns = SYSTEM_PARAMS.contact.normal_stiffness
        nd = SYSTEM_PARAMS.contact.normal_damping
        ts = SYSTEM_PARAMS.contact.tangential_stiffness
        cfc = SYSTEM_PARAMS.contact.coulomb_friction_coeff

        print(f"Normal Stiffness: {self.normal_stiffness[None] / ns}")
        print(f"Normal Damping: {self.normal_damping[None] / nd}")
        print(f"Tangential Stiffness: {self.tangential_stiffness[None] / ts}")
        print(f"Coulomb Friction Coefficient: {self.coulomb_friction_coeff[None] / cfc}")
    
    def save_final_params(self):
        results = {
            "vitactip": {
                "youngs_modulus": self.vitactip.youngs_modulus[None],
                "poissons_ratio": self.vitactip.poissons_ratio[None]
            },
            "phantom": {
                "youngs_modulus_0": self.phantom.youngs_modulus[0],
                "youngs_modulus_1": self.phantom.youngs_modulus[1],
                "poissons_ratio_0": self.phantom.poissons_ratio[0],
                "poissons_ratio_1": self.phantom.poissons_ratio[1]
            },
            "contact": {
                "coulomb_friction_coeff": self.coulomb_friction_coeff[None],
                "normal_stiffness": self.normal_stiffness[None],
                "tangential_stiffness": self.tangential_stiffness[None],
                "normal_damping": self.normal_damping[None]
            }
        }
        results = {
            k1: {
                k2: float(v2) 
                for k2, v2 in v1.items()
            }
            for k1, v1 in results.items()
        }
        with open(SYSTEM_PARAMS.files.domain_adaptation_results, "w") as f:
            json.dump(results, f, indent=4, cls=ScientificNotationEncoder)
    
    def set_dt(self, verbose=False):
        # dt = calculate_cfl_timestep(
        #     phantom_healthy_youngs_modulus=self.phantom.youngs_modulus[0],
        #     vitactip_youngs_modulus=self.vitactip.youngs_modulus[None],
        #     courant_number=self.courant_number,
        #     verbose=verbose,
        # )
        dt = SYSTEM_PARAMS.contact.dt_override
        self.dt[None] = dt
        self.phantom.dt[None] = dt
        self.vitactip.dt[None] = dt
        # if verbose:
        #     print(f'dt={dt:0.3e} s')
    
    def get_keypoint_indices_and_validate(self):
        self.vitactip.test_mapping_from_global_space_to_camera_space()
        self.vitactip.extract_markers(0)
        self.compute_mapping_between_experimental_and_sim_markers()
        self.vitactip.save_predicted_markers_to_image()
        self.vitactip.compute_clock_arm_ixs()
        self.vitactip.compute_tip_ix()
        initial_markers = self.vitactip.deformed_markers.to_numpy()
        with open(SYSTEM_PARAMS.files.sim_markers_initial_positions, "wb") as f:
            pickle.dump(initial_markers, f)
        self.vitactip.compute_vertices_E()
        with open(SYSTEM_PARAMS.files.vitactip_points_E, "wb") as f:
            pickle.dump(self.vitactip.vertices_E.to_numpy(), f)

    @ti.kernel
    def log(self):
        if self.interpolation_valid[None] == 1:
            print(0)

    def record_da_trajectories(self, run_dir, fps=None, video_scope=None):
        """Run each of the four DA interactions ONCE and record them to video.

        This is NOT calibration - there is no Bayesian optimisation, no parameter
        proposal and no scoring loop. Every trajectory runs at the parameters
        already in `system-params.json` (Young's modulus, Poisson's ratio, the
        contact coefficients), exactly as loaded by `Contact.__init__` and
        `Vitactip.set_up_system_params()`. `set_contact_params_from_bo()` is
        deliberately never called, so nothing here overrides the JSON.

        Frames are the DEFAULT SIMULATOR CAMERA's view, grabbed straight from the
        GGUI colour buffer with `get_image_buffer_as_numpy()` - the same image the
        window shows, with no separate render pass and no PNG round-trip.

        Writes, under `run_dir`:
            <name>.mp4              one video per trajectory
            all_trajectories.mp4    the four concatenated, in panel order
            recording.json          frame counts, fps and the parameters used

        Args:
            run_dir: output directory (already timestamped by the caller).
            fps: output frame rate; defaults to $DIFFTACTILE_VIDEO_FPS or 30.
            video_scope: "both" (default), "per-trajectory" or "combined".
        """
        import cv2 as _cv2

        fps = fps or _env_int("DIFFTACTILE_VIDEO_FPS", 30)
        video_scope = video_scope or os.environ.get(
            "DIFFTACTILE_VIDEO_SCOPE", "both"
        )
        if video_scope not in ("both", "per-trajectory", "combined"):
            raise ValueError(
                f"video_scope must be both/per-trajectory/combined, got {video_scope!r}"
            )
        want_each = video_scope in ("both", "per-trajectory")
        want_combined = video_scope in ("both", "combined")

        if self.window is None:
            # Nothing to capture: the frames come from the GGUI colour buffer, so
            # a run with no window has no video to record. Fail loudly rather
            # than writing four empty files.
            raise RuntimeError(
                "record_da_trajectories() needs a Taichi GGUI window, but none "
                "was created (headless). Run with a DISPLAY set."
            )

        os.makedirs(run_dir, exist_ok=True)
        # mp4v over H.264: it is the codec always present in an OpenCV build,
        # whereas 'avc1' depends on how the wheel was compiled and fails by
        # writing a 0-byte file rather than raising.
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
        da_max_ts = _env_int(
            "DIFFTACTILE_DA_MAX_TIMESTEPS", DA_MAX_TIMESTEPS_VEIN
        )

        print(
            f"\nRecording {self.trajectories.shape[0]} trajectories at "
            f"{fps} fps -> {run_dir}\n"
        )

        manifest = {"fps": fps, "trajectories": [], "params": {
            "vitactip_youngs_modulus": float(self.vitactip.youngs_modulus[None]),
            "vitactip_poissons_ratio": float(self.vitactip.poissons_ratio[None]),
            "normal_stiffness": float(self.normal_stiffness[0]),
            "tangential_stiffness": float(self.tangential_stiffness[0]),
            "normal_damping": float(self.normal_damping[0]),
            "coulomb_friction_coeff": float(self.coulomb_friction_coeff[0]),
        }}
        combined_writer = None

        # DIFFTACTILE_RECORD_TRAJECTORIES="slide,press" records a subset (all
        # four by default) - used to record the vessel-present slide alone.
        only = os.environ.get("DIFFTACTILE_RECORD_TRAJECTORIES", "")
        only = [n.strip() for n in only.split(",") if n.strip()]
        unknown = [n for n in only if n not in self.trajectory_names]
        if unknown:
            raise ValueError(
                f"DIFFTACTILE_RECORD_TRAJECTORIES names {unknown}, which are not "
                f"in {self.trajectory_names}"
            )

        for i in range(self.trajectories.shape[0]):
            name = self.trajectory_names[i]
            if only and name not in only:
                continue
            self.trajectory_ix[None] = i

            # Same reset sequence as domain_adaptation(): rebuild from the rest
            # pose, then clear the markers/accumulators derived from the old one.
            # extract_markers() accumulates with `+=`, so stale projections would
            # otherwise persist across trajectories.
            self.vitactip.reset_state()
            self.phantom.reset_state()
            self.set_up_initial_positions_state_and_trajectory()
            self.vitactip.reset_state()
            self.phantom.reset_state()
            self.reset_pid_controller()
            self.visualisation_reset_scene()
            self.reset_exp_sim_traj()
            self.vitactip.extract_markers(0)
            self.compute_mapping_between_experimental_and_sim_markers()
            self.set_dt()

            writer = None
            ts = 0
            while self.last_target_reached[None] != 1 and ts < da_max_ts:
                self.pid_controller_1()
                self.pid_controller_2(ts)
                self.pid_controller_3()
                self.vitactip.set_pose_control_1()
                self.vitactip.set_pose_control_2()
                self.vitactip.set_pose_control_3()
                self.forward_pass_common_part(ts)
                self.copy_frame()
                self.vitactip.extract_markers(
                    SYSTEM_PARAMS.contact.num_sub_frames - 1
                )
                self.visualisation_update_gui(ts)

                frame = self._grab_gui_frame()
                if writer is None and want_each:
                    h, w = frame.shape[:2]
                    writer = _cv2.VideoWriter(
                        os.path.join(run_dir, f"{name}.mp4"), fourcc, fps, (w, h)
                    )
                if combined_writer is None and want_combined:
                    h, w = frame.shape[:2]
                    combined_writer = _cv2.VideoWriter(
                        os.path.join(run_dir, "all_trajectories.mp4"),
                        fourcc, fps, (w, h),
                    )
                if writer is not None:
                    writer.write(frame)
                if combined_writer is not None:
                    combined_writer.write(frame)
                ts += 1

            if writer is not None:
                writer.release()
            print(f"    {name:9s} {ts:4d} frames "
                  f"({ts / fps:.1f} s at {fps} fps)")
            manifest["trajectories"].append({"name": name, "frames": ts})

        if combined_writer is not None:
            combined_writer.release()

        with open(os.path.join(run_dir, "recording.json"), "w") as f:
            json.dump(manifest, f, indent=4)

        total = sum(t["frames"] for t in manifest["trajectories"])
        print(f"\n    total {total} frames ({total / fps:.1f} s)")
        return manifest

    def _grab_gui_frame(self):
        """One BGR frame of the default camera view, ready for cv2.VideoWriter.

        Taichi hands back a float RGBA buffer indexed [x, y] with the origin at
        the BOTTOM-left, which is three conventions away from what OpenCV wants:
        transpose to [row, col], flip the rows so the origin is top-left, drop
        alpha, scale to 0-255, and reverse RGB to BGR. Getting any one of these
        wrong yields a video that is silently sideways, upside down or colour
        swapped - and on this scene a red/blue swap would be especially
        misleading, since blue is the phantom and the overlays use red.
        """
        buf = self.window.get_image_buffer_as_numpy()
        rgb = buf[:, :, :3]
        rgb = np.transpose(rgb, (1, 0, 2))
        rgb = np.flipud(rgb)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb[:, :, ::-1]

    def build_da_schedule(self, iters_per_model=None):
        """Which trajectory each BO iteration scores against.

        FOUR INDEPENDENT SURROGATES, one per trajectory, each getting
        `iters_per_model` iterations (default 10 = 4 random + 6
        acquisition-driven), run BLOCK BY BLOCK: all of slide's iterations, then
        all of press's, and so on. Default schedule is 4 * 10 = 40 iterations.

        Returns a list with one entry per iteration, each a single-element list
        of trajectory NAMES. One trajectory per iteration is the point: each
        iteration then costs ONE forward simulation instead of four, and the
        model for that trajectory gets 100% of the signal from it - no averaging
        across objectives on different scales.

        THE ORDER OF THE BLOCKS DOES NOT AFFECT THE RESULT. The four models are
        fitted independently - no model sees another's observations - so running
        them consecutively or interleaved gives each the same data and the same
        posterior. Blocks are simply the clearer arrangement: the log reads as
        four self-contained searches, and each model's 4 random draws are
        immediately followed by its own 6 acquisition-driven ones instead of
        being spread across the whole run.

        `slide` leads only by convention (it is the most informative interaction
        and dominates the summed objective); nothing depends on it.

        NOTE THE SCORES ACROSS ITERATIONS ARE NOT COMPARABLE: a press MAE and a
        slide MAE are different quantities and differ by ~10x. Each is only
        comparable within its own model. The recommendation step
        (`BoGp.recommend`) is what combines them.

        Overridable with DIFFTACTILE_BO_ITERS_PER_MODEL.
        """
        iters_per_model = (
            iters_per_model if iters_per_model is not None
            else _env_int("DIFFTACTILE_BO_ITERS_PER_MODEL", 10)
        )
        # Deliberately NOT self.trajectory_names order (press, twist_z, twist_x,
        # slide): slide's block runs first.
        #
        # DIFFTACTILE_BO_MODELS restricts the run to a subset, e.g.
        # DIFFTACTILE_BO_MODELS=slide fits only the slide surrogate. Useful for
        # iterating on one trajectory without paying for the other three - each
        # model is independent, so a subset run produces exactly the same
        # surrogate for the models it does include. Note the recommendation is
        # then a single-objective one: `recommend()` sums over whichever models
        # have observations, so with one model enabled it minimises that
        # trajectory's predicted MAE alone.
        models = ["slide", "press", "twist_x", "twist_z"]
        selected = os.environ.get("DIFFTACTILE_BO_MODELS")
        if selected:
            wanted = [s.strip() for s in selected.split(",") if s.strip()]
            unknown = [w for w in wanted if w not in models]
            if unknown:
                raise ValueError(
                    f"DIFFTACTILE_BO_MODELS: unknown {unknown}; choose from {models}"
                )
            models = [m for m in models if m in wanted]
        missing = [n for n in models if n not in self.trajectory_names]
        if missing:
            raise ValueError(
                f"schedule names not in trajectory_names: {missing}. "
                f"Available: {self.trajectory_names}"
            )
        schedule = []
        for name in models:
            schedule.extend([[name] for _ in range(iters_per_model)])
        return schedule

    def domain_adaptation(self, num_iterations=None, run_dir=None, schedule=None):
        """Bayesian-optimisation calibration of the simulator's material/contact
        parameters against the real sensor.

        One ITERATION = propose a parameter set, replay all four canonical
        trajectories (press, twist_z, twist_x, slide) with it, and score it by the
        aggregated MAE between simulated and real marker positions at each apex.
        The first `num_random` iterations sample the space at random to seed the
        Gaussian process; the rest are proposed by the acquisition function.

        NO DIFFERENTIABLE SIMULATION. This used to run a Taichi backward pass and
        a gradient-based optimiser over the same objective; that approach was
        abandoned, and everything supporting it has been removed. The objective
        is evaluated by forward simulation only, and BO treats it as a black box -
        which is why it needs no gradients through the simulator at all.

        Writes, under `run_dir`:
            bo_all_params.json / bo_all_targets.json   every configuration tried
            bo_results.json                            best config + full history
            bo_convergence.png                         MAE vs iteration
            da_overlay_<name>.png                      alignment, best config
        """
        # The schedule decides how many iterations there are, and which
        # trajectories each one scores - see build_da_schedule(). An explicit
        # num_iterations still wins, truncating or (by repeating the last entry)
        # extending it, so existing callers and DIFFTACTILE_BO_ITERATIONS behave
        # as before.
        schedule = schedule if schedule is not None else self.build_da_schedule()
        if num_iterations is None:
            num_iterations = _env_int("DIFFTACTILE_BO_ITERATIONS", len(schedule))
        if num_iterations <= len(schedule):
            schedule = schedule[:num_iterations]
        else:
            schedule = schedule + [schedule[-1]] * (num_iterations - len(schedule))
        # RANDOM EXPLORATION IS ONLY THE FIRST 4 ITERATIONS, so with the stock
        # schedule phase 1 is 4 random slide runs followed by 6 acquisition-driven
        # slide runs, and every fine-tuning iteration is acquisition-driven too.
        #
        # The random draws exist ONLY to give the GP something to fit before its
        # acquisition function means anything - a GP with no observations proposes
        # arbitrarily. Four is enough for that, and every iteration beyond it is
        # better spent on a proposal that uses what has been learned: a fully
        # random phase 1 would be a plain random search, improving only by luck.
        num_random = min(
            _env_int("DIFFTACTILE_BO_RANDOM", 4), max(num_iterations - 1, 1)
        )
        run_dir = run_dir or repo_path("difftactile/output")
        os.makedirs(run_dir, exist_ok=True)
        self.bo.set_run_dir(run_dir)

        phases = []
        for names in schedule:
            label = "+".join(names)
            if phases and phases[-1][0] == label:
                phases[-1][1] += 1
            else:
                phases.append([label, 1])
        print(
            f"\nBayesian optimisation: {num_iterations} iterations "
            f"({num_random} random, then acquisition-driven).\n"
            f"Schedule: " + ", ".join(f"{n}x {lbl}" for lbl, n in phases) + "\n"
        )

        history = []
        # Best record per `scored_on` scope - see the note where it is updated.
        best_by_scope = {}
        for it in range(num_iterations):
            # Propose the next parameter set. Random during the exploration
            # phase, to give the GP something to fit before it starts exploiting.
            # The surrogate this iteration proposes from and reports to. One
            # model per trajectory, so "the first 4 iterations are random" is
            # counted PER MODEL, not globally - each model needs its own seed
            # observations before its acquisition function means anything.
            model_name = schedule[it][0]
            seen = len(self.bo.observations.get(model_name, []))
            if seen < num_random:
                self.bo.my_suggest_random(model_name)
                how = "random"
            else:
                self.bo.my_suggest_optimise(model_name)
                how = "acquisition"
            self.set_contact_params_from_bo()

            # Trajectories this iteration scores against, per the schedule.
            iteration_names = schedule[it]
            print(
                f"=== iteration {it + 1}/{num_iterations} ({how}) "
                f"[{'+'.join(iteration_names)}] ==="
            )
            for key, value in self.bo.params.items():
                print(f"    {key:26s} {value:.6g}")

            # Keep this iteration's rendered frames separate from the others'.
            # Read by visualisation_update_gui(), which writes
            # <snapshot_dir>/iterNNN/<trajectory>_ts<NNNN>.png - matching the
            # iterNNN_<name>.npz convention the trajectories/ directory uses, so
            # a frame and the raw markers behind it are easy to line up.
            self.snapshot_subdir = f"iter{it:03d}"

            per_trajectory = {}
            self.da_losses = []
            diverged = False
            for name in iteration_names:
                i = self.trajectory_names.index(name)
                self.trajectory_ix[None] = i
                # Full state reset between parameter sets.
                #
                # A diverged set leaves NaN throughout the sensor state. The
                # stubborn carrier was vertex_velocities[0]: reset_state() used to
                # clear frames 1..N only, but frame 0 is what set_vel(0) writes and
                # the next simulation reads, so 7182 non-finite entries survived
                # every reset. ONE bad parameter set therefore poisoned every
                # iteration after it - 17 of 20 were recorded as "diverged" though
                # each scored normally when re-run in a fresh process.
                #
                # reset_state() now clears frame 0's velocities too (see
                # vitactip.reset_state), and the order below matters:
                # set_up_initial_positions_state_and_trajectory() rebuilds the mesh
                # from the rest pose, then the second reset clears the markers and
                # accumulators derived from the OLD pose. extract_markers()
                # accumulates with `+=`, so a NaN left in those projections would
                # never wash out.
                self.vitactip.reset_state()
                self.phantom.reset_state()
                self.set_up_initial_positions_state_and_trajectory()
                self.vitactip.reset_state()
                self.phantom.reset_state()
                self.reset_pid_controller()
                self.visualisation_reset_scene()
                self.reset_exp_sim_traj()
                self.vitactip.extract_markers(0)
                self.compute_mapping_between_experimental_and_sim_markers()
                self.set_dt()

                # Forward simulation to the trajectory's apex. Bounded so a
                # controller regression cannot spin forever; the PID normally
                # terminates this well before the cap.
                da_max_ts = _env_int(
                    "DIFFTACTILE_DA_MAX_TIMESTEPS_NO_VEIN", DA_MAX_TIMESTEPS_NO_VEIN
                )
                ts = 0
                while self.last_target_reached[None] != 1 and ts < da_max_ts:
                    self.pid_controller_1()
                    self.pid_controller_2(ts)
                    self.pid_controller_3()
                    self.vitactip.set_pose_control_1()
                    self.vitactip.set_pose_control_2()
                    self.vitactip.set_pose_control_3()
                    self.forward_pass_common_part(ts)
                    # Carries the pose reached at the end of this timestep back to
                    # frame 0, which is where the next timestep reads from.
                    # Without it every timestep restarts from the initial pose.
                    self.copy_frame()
                    self.vitactip.extract_markers(
                        SYSTEM_PARAMS.contact.num_sub_frames - 1
                    )
                    self.visualisation_update_gui(ts)
                    ts += 1

                # MAE at the apex, and the alignment overlay for this trajectory.
                #
                # An extreme parameter set can make the FEM solve blow up, and
                # the marker positions come back NaN - which the Hungarian
                # reordering rejects with "matrix contains invalid numeric
                # entries". That is a legitimate answer from the objective ("this
                # configuration is unusable"), not a crash, so it is scored as a
                # divergence and the search continues.
                try:
                    self.compute_da_loss(out_dir=run_dir)
                except (ValueError, IndexError) as exc:
                    print(f"    {name:9s} DIVERGED ({exc})")
                    diverged = True
                    break
                mae_px = self.da_losses[-1]
                if not np.isfinite(mae_px):
                    print(f"    {name:9s} DIVERGED (non-finite MAE)")
                    diverged = True
                    break

                # The collected trajectory itself: simulated marker positions at
                # the apex, beside the real ones they are scored against. Saved
                # per (iteration, trajectory) so a run's raw data survives for
                # re-analysis without re-simulating - and because the run
                # directory is timestamped, repeated runs accumulate rather than
                # overwrite.
                traj_dir = os.path.join(run_dir, "trajectories")
                os.makedirs(traj_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(traj_dir, f"iter{it:03d}_{name}.npz"),
                    sim_markers=self.sim_markers_deformed.to_numpy(),
                    exp_markers=np.load(self.da_npz_paths[name])["points"],
                    mae_px=mae_px,
                    timesteps=ts,
                    params=json.dumps(dict(self.bo.params)),
                )
                per_trajectory[name] = mae_px
                print(
                    f"    {name:9s} MAE = {mae_px:6.2f} px "
                    f"({mae_px * PX_TO_MM:.2f} mm)  [{ts} timesteps]"
                )

            if diverged:
                # Penalise rather than drop it: the GP learns to avoid a region
                # only if it is told the region is bad. A large MAE in the same
                # pixel units the objective already uses - no rescaling, so the
                # penalty reads directly as "far worse than any usable
                # configuration" (real values land well under 300 px).
                aggregated = DIVERGENCE_PENALTY_PX
                print(f"    diverged -> scored as {aggregated:.0f} px (penalty)")
            else:
                aggregated = float(np.mean(self.da_losses))
            self.bo.my_register(aggregated, model_name)
            scored_on = "+".join(iteration_names)
            record = {
                "iteration": it,
                "proposed_by": how,
                # WHICH trajectories this score came from. Load-bearing under a
                # schedule: without it the history is a column of MAEs measured
                # against different objectives with no way to tell them apart.
                "scored_on": scored_on,
                "params": dict(self.bo.params),
                "mae_px": aggregated,
                "mae_mm": aggregated * PX_TO_MM,
                "per_trajectory_px": per_trajectory,
            }
            history.append(record)
            print(
                f"    MAE = {aggregated:.2f} px "
                f"({aggregated * PX_TO_MM:.2f} mm)  [{scored_on}]"
            )

            # "Best" is only meaningful AMONG ITERATIONS SCORED THE SAME WAY -
            # a press MAE and a slide MAE are different quantities and routinely
            # differ by ~10x, so comparing them across the whole history would
            # just pick whichever trajectory happens to score lowest. Best is
            # therefore tracked per `scored_on`, and the headline `best` is the
            # best on the LAST entry in the schedule (slide by default), which is
            # the objective the run finishes on.
            prev = best_by_scope.get(scored_on)
            if not diverged and (prev is None or aggregated < prev["mae_px"]):
                best_by_scope[scored_on] = record
                # Keep the overlays produced by the best configuration so far.
                # Later iterations overwrite da_overlay_*.png, so the winning set
                # is preserved under its own names. Only the trajectories this
                # iteration actually ran have a fresh overlay to copy.
                for name in iteration_names:
                    src = os.path.join(run_dir, f"da_overlay_{name}.png")
                    if os.path.exists(src):
                        shutil.copyfile(
                            src, os.path.join(run_dir, f"best_da_overlay_{name}.png")
                        )
                print(f"    ^ best so far for [{scored_on}]")
            print()

        self.bo.write_to_file()
        # The headline single-trajectory result is the best `slide` observation -
        # the interaction taken to be most informative, and the one that
        # dominates the summed objective. Pinned to slide by NAME rather than to
        # whichever scope the schedule happens to end on, so reordering the cycle
        # cannot silently change which trajectory the headline refers to.
        # Falling back to the global minimum only matters if slide never produced
        # a usable (non-diverged) score.
        best = best_by_scope.get("slide")
        if best is None and best_by_scope:
            best = min(best_by_scope.values(), key=lambda r: r["mae_px"])

        # The multi-model posterior combination (`bo.recommend()`) is NOT run
        # here. With the project relying exclusively on the slide model there is
        # nothing to combine: the best measured slide configuration IS the
        # answer, and a one-model "composite" would only restate it through a
        # surrogate, adding the GP's extrapolation error to a number that is
        # otherwise measured. `recommend()` is kept in bo_gp.py for whoever
        # re-enables the other three trajectory models.
        recommendation = None

        self._report_bo_results(history, best, run_dir, best_by_scope)
        return history, best, recommendation

    def domain_adaptation_vein(self, num_iterations=None, run_dir=None,
                               sensor_name="slide", model_name="slide_vein"):
        """Stage 2: BO on the slide trajectory WITH the subsurface vessel present.

        Objective: MAXIMISE the MAE between the simulated markers and the
        vessel-FREE reference photograph, measured at the moment the projected
        vessel passes under the sensor centre.

        The reasoning, and why the two stages point in opposite directions:

          * stage 1 (vessel-absent, MINIMISE MAE) asks "does the simulator
            reproduce the real sensor?" - low error means high fidelity.
          * stage 2 (vessel-present, MAXIMISE the same MAE) asks "does a
            subsurface vessel actually show up in the markers?" - a large
            departure from the vessel-free appearance means the signal the GNN
            must detect is present at all.

        ONE FREE PARAMETER: the sensor<->vein normal stiffness. Everything else
        about that contact pair is fixed (Contact.VEIN_*: damping at the repo's
        100, tangential stiffness and friction at 0), and the sensor material -
        Young's modulus and Poisson's ratio - is INHERITED from the vessel-absent
        model's best result rather than re-fitted.

        NO FIDELITY CONSTRAINT IS NEEDED, and none is applied. The two stages fit
        disjoint parameters, and a sensor<->vein contact coefficient cannot
        change how the sensor behaves on a phantom with no vein in it. The
        vessel-absent fidelity established in stage 1 is therefore preserved by
        construction, not by policing the search. (An earlier design shared one
        6-D space between the stages and needed rejection sampling to keep stage
        2 faithful; splitting the spaces removed the problem rather than
        managing it.)

        Because the contact force is F_n = -(k_n + c_n*v_n)*d*n_hat, raising the
        vein's normal stiffness raises the force the vessel exerts without
        touching the sensor's own stiffness - which is what allows a stiff sensor
        to still show a pronounced local indentation over the vessel.
        """
        # 4 random + 6 acquisition by default.
        num_iterations = num_iterations or _env_int(
            "DIFFTACTILE_BO_VEIN_ITERATIONS", 10
        )
        num_random = min(_env_int("DIFFTACTILE_BO_RANDOM", 4), num_iterations)
        run_dir = run_dir or repo_path("difftactile/output")
        os.makedirs(run_dir, exist_ok=True)

        # Sensor material from stage 1's best, held fixed for the whole stage.
        sensor_params, sensor_mae = self.bo.best_observed(sensor_name)
        self.vitactip.youngs_modulus[None] = sensor_params[
            "vitactip_youngs_modulus"
        ]
        self.vitactip.poissons_ratio[None] = sensor_params[
            "vitactip_poissons_ratio"
        ]
        self.vitactip.set_up_system_params_2()

        print("\n" + "=" * 70)
        print(" Stage 2: vessel-present slide (MAXIMISING marker disagreement)")
        print("=" * 70)
        print(
            f"\n{num_iterations} iterations ({num_random} random, then "
            f"acquisition-driven).\n"
            f"Free parameter: normal_stiffness of the sensor<->vein pair "
            f"{self.bo.pbounds_vein['normal_stiffness']}\n"
            f"Fixed: sensor E={sensor_params['vitactip_youngs_modulus']:.6g}, "
            f"nu={sensor_params['vitactip_poissons_ratio']:.4g} "
            f"(stage-1 best, {sensor_mae:.2f} px); vein damping="
            f"{self.VEIN_NORMAL_DAMPING:g}, tangential stiffness="
            f"{self.VEIN_TANGENTIAL_STIFFNESS:g}, friction="
            f"{self.VEIN_COULOMB_FRICTION:g}\n"
        )

        # The vessel is what this stage is about, so switch the contact pair on.
        previous_collision_ixs = list(self.collision_ixs)
        self.collision_ixs = active_collision_pairs(True)
        history, best = [], None
        completed = 0
        try:
            for it in range(num_iterations):
                how = "random" if it < num_random else "acquisition"
                self.bo.suggest_for(model_name, force_random=(it < num_random))
                self.set_contact_params_from_bo()
                self.snapshot_subdir = f"vein_iter{it:03d}"

                print(f"=== vein iteration {it + 1}/{num_iterations} ({how}) ===")
                print(f"    normal_stiffness (vein)    "
                      f"{self.bo.params['normal_stiffness']:.6g}")

                mae, triggered, closest, diverged, _ = self._run_vein_slide(
                    run_dir, it
                )
                if diverged:
                    # An unusable configuration earns nothing, the same as one
                    # that is faithful but shows no vessel. Zero rather than a
                    # negative score: a blown-up solve is not "worse than
                    # useless", it is simply useless.
                    score = 0.0
                    print(f"    diverged -> scored {score:.0f}")
                elif mae is None:
                    print("    vessel never reached the sensor centre "
                          f"(closest {closest:.1f} px) -> scored 0")
                    score = 0.0
                else:
                    score = mae
                    print(f"    vessel-present MAE = {mae:.2f} px "
                          f"({mae * PX_TO_MM:.2f} mm)  [vessel {closest:.1f} px "
                          f"from centre]")

                # maximise=True: a LARGER disagreement is better here.
                self.bo.my_register(score, model_name, maximise=True)
                record = {
                    "iteration": it,
                    "proposed_by": how,
                    "params": dict(self.bo.params),
                    "sensor_params": dict(sensor_params),
                    "vein_mae_px": score,
                    "vein_mae_mm": score * PX_TO_MM,
                    "triggered": bool(triggered),
                    "diverged": bool(diverged),
                    "closest_vein_px": None if closest == float("inf") else closest,
                }
                history.append(record)
                completed = it + 1
                # A diverged iteration can never be "best", whatever its score.
                if (not diverged and score > 0
                        and (best is None or score > best["vein_mae_px"])):
                    best = record
                    print("    ^ best so far")
                print()
        except BaseException as exc:
            # A crash here is almost always the simulator blowing up at a high
            # vein stiffness. Report how far the search got and what to do about
            # it BEFORE re-raising, so the message is not buried under a
            # traceback the user has to scroll past.
            self._report_vein_crash(exc, completed, num_iterations,
                                    sensor_params, history, run_dir)
            raise
        finally:
            self.collision_ixs = previous_collision_ixs

        path = os.path.join(run_dir, "bo_vein_results.json")
        with open(path, "w") as f:
            json.dump({"best": best, "history": history,
                       "sensor_params": sensor_params,
                       "stage1_best_mae_px": sensor_mae}, f, indent=4)
        print(f"Stage-2 history: {path}")
        if best is not None:
            print(f"\nBest vessel-present configuration (iteration "
                  f"{best['iteration']}): {best['vein_mae_px']:.2f} px")
            print(f"    normal_stiffness (vein)    "
                  f"{best['params']['normal_stiffness']:.6g}")
        return history, best

    def _report_vein_crash(self, exc, completed, num_iterations, sensor_params,
                           history, run_dir):
        """Explain a stage-2 crash and what the operator should do next.

        The expected cause is an over-stiff CONTACT: the vein's penalty force is
        (k_n + c_n*v_n)*d, and a large k_n at a fixed dt drives the explicit
        solve unstable. The sensor's own stiffness sets how hard it resists that
        force, so halving the sensor's Young's modulus makes the pair solvable
        again at the same vein stiffness.

        Deliberately NOT automatic: halving E changes the sensor the whole
        project is calibrated around, so it is the operator's decision, and the
        halved value should be recorded wherever the adopted parameters live.
        """
        # Salvage whatever was collected before the crash - those iterations are
        # valid observations and re-running them costs simulation time.
        partial = os.path.join(run_dir, "bo_vein_results_partial.json")
        try:
            with open(partial, "w") as f:
                json.dump({"history": history, "sensor_params": sensor_params,
                           "crashed_after": completed}, f, indent=4)
        except Exception:
            partial = None

        e = sensor_params.get("vitactip_youngs_modulus")
        print("\n" + "!" * 70)
        print(" STAGE 2 CRASHED")
        print("!" * 70)
        print(f"\n  {completed}/{num_iterations} iterations completed "
              f"successfully before the failure.")
        print(f"  Error: {type(exc).__name__}: {exc}")
        if partial:
            print(f"  Partial results saved: {partial}")

        # A CODE error is not a simulation failure, and the stiffness advice
        # below would be actively misleading for one - it would send the
        # operator to change a physical parameter over what is a bug. Only
        # numerical failures (or a hard interpreter-level abort) point at the
        # contact being too stiff.
        code_errors = (NameError, AttributeError, TypeError, ImportError,
                       KeyError, IndexError, SyntaxError)
        if isinstance(exc, code_errors):
            print(
                "\n  This is a CODE error, not a simulation failure. The "
                "parameters are not\n  at fault - fix the traceback above and "
                "re-run. Do NOT change the sensor\n  stiffness on account of "
                "this."
            )
            print("!" * 70 + "\n")
            return

        print(
            "\n  Most likely cause: the sensor<->vein contact became too stiff "
            "for the\n  fixed timestep (contact.dt_override = 1e-5 s). The "
            "contact force is\n  (k_n + c_n*v_n) * d, so a large vein "
            "normal_stiffness destabilises the\n  explicit solve."
        )
        if e is not None:
            print(
                f"\n  SUGGESTED MANUAL ACTION: halve the sensor Young's modulus "
                f"assigned by\n  the vessel-absent model, then re-run stage 2:"
                f"\n\n      {e:.6g}  ->  {e / 2:.6g}\n"
                f"\n  A softer sensor absorbs the same contact force with less "
                f"violent\n  acceleration, which is what restores stability. "
                f"Set it in\n  system-params.json (vitactip.single_material."
                f"youngs_modulus) or in the\n  stage-1 observations this stage "
                f"reads, and run domain_adaptation.sh again."
            )
        print("!" * 70 + "\n")

    def domain_adaptation_joint(self, num_iterations=None, run_dir=None,
                                model_name="joint"):
        """ONE BO fitting sensor stiffness and vein contact stiffness together.

        Every iteration runs BOTH trajectories at the same parameter set:

            vessel-ABSENT slide   -> fidelity to the real sensor
            vessel-PRESENT slide  -> how visibly the vessel deforms the sensor

        and scores them with a single objective, so the search can trade a
        little fidelity for a lot of vessel contrast. The two-stage design
        structurally could not do that: stage 1 fixed the sensor before stage 2
        ever saw a vessel, so a sensor that was slightly worse in isolation but
        far more sensitive was unreachable.

        THE OBJECTIVE - penetration minus infidelity, both normalised:

            opt_obj = vpn - van        maximised, on [-1, +1]

            vpn  how far the vessel held the sensor up, on [0, 1]: 1 once
                 the sensor is stopped at the vessel's apex height or above,
                 0 if it sank exactly as far as it would have with no vessel
                 there.
            van  vessel-absent mean marker error / 165 px, 0 = perfect match

        with a single hard rejection: a run whose sensor DIVERGED (NaN or
        otherwise invisible points) scores -1 regardless. There is no fidelity
        gate - `van` already penalises a poor match continuously, so a threshold
        would double-count it and discard the ordering among unfaithful
        configurations.

        The vessel-present marker delta is still measured and logged (see
        SIM-VS-SIM below) but no longer scored: it proved to be ~0.06-0.09 px
        against a 55 px marker spacing, too small to optimise.

        SIM-VS-SIM, NOT SIM-VS-PHOTO. That delta is the effect of the vessel and
        nothing else: the two snapshots differ only by the presence of the
        inclusion, so their difference isolates exactly the deformation the GNN
        has to detect. Measuring the vessel-present markers against the real
        PHOTOGRAPH instead
        (the earlier formulation) conflated two things - how much the vessel
        deforms the sensor, and how badly the simulator already disagrees with
        reality - and the second term is both larger and irrelevant here, since
        the photograph shows a phantom with no vessel in it. Fidelity to the
        photograph is the gate's job; A measures sensitivity alone.

        The comparison is index-wise, which is meaningful because
        `get_graph_connectivity` reorders both snapshots into the same canonical
        marker ordering.

        BOTH SNAPSHOTS ARE TAKEN AT THE SAME TIMESTEP. The vessel-present run
        short-circuits when the vessel passes under the sensor centre, so the
        vessel-free comparison run is stopped at that same timestep rather than
        at its own apex. Comparing a trigger-time snapshot against an apex
        snapshot would measure how far the sensor travelled between the two
        moments - a difference dominated by the slide itself, which has nothing
        to do with the vessel.

        Why a gate rather than a weighted sum. The two quantities are not
        commensurable - fidelity is a constraint to be satisfied, sensitivity a
        quantity to be maximised - and any fixed weighting between them lets a
        big enough sensitivity score buy its way out of being realistic. The
        gate makes that impossible: below the threshold nothing is earned at all,
        so the search cannot trade fidelity away, and above it the score depends
        only on how strongly the vessel shows.

        THE THRESHOLD IS ONE INTER-MARKER SPACING (55 px). That is the sensor's
        own length scale: a mean error of one full spacing means the average
        marker has moved as far as the distance to its neighbour, at which point
        the simulated marker field no longer corresponds to the real one in any
        useful way. It also comfortably admits every realistic configuration
        seen so far (~10-15 px) while excluding the diverged ones (115-380 px).

        A rather than the mean, on the vessel-present side, because a subsurface
        vessel deforms a SMALL PATCH rather than the whole field: the maximum
        captures the local excursion the GNN has to detect, whereas a mean over
        127 markers dilutes it with the many markers the vessel never reaches.

        The vessel-absent run therefore serves as the gate rather than as a
        reported aside; both numbers are still logged.
        """
        num_iterations = num_iterations or _env_int(
            "DIFFTACTILE_BO_JOINT_ITERATIONS", 10
        )
        num_random = min(_env_int("DIFFTACTILE_BO_JOINT_RANDOM", 5),
                         num_iterations)
        run_dir = run_dir or repo_path("difftactile/output")
        os.makedirs(run_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print(" Joint BO: sensor stiffness + vein contact, both trajectories")
        print("=" * 70)
        print(
            f"\n{num_iterations} iterations ({num_random} random, then "
            f"acquisition-driven).\n"
            f"Free: vitactip_youngs_modulus "
            f"{self.bo.pbounds_joint['vitactip_youngs_modulus']}, "
            f"normal_stiffness (vein) "
            f"{self.bo.pbounds_joint['normal_stiffness']}\n"
            f"Objective: MAXIMISE 3*A - B, A = max marker error, "
            f"B = mean marker error (vessel-present)\n"
        )

        # Live per-iteration log. Written to a shared folder under a
        # run-timestamped name, so `tail -f` follows the search in real time and
        # runs sit side by side for comparison afterwards.
        log_path = os.path.join(
            repo_path("difftactile/output/bo_logs"),
            f"joint_{time.strftime('%Y%m%d-%H%M%S')}.csv",
        )
        iteration_log = IterationLog(log_path, [
            "iteration", "proposed_by",
            "vitactip_youngs_modulus", "normal_stiffness",
            "vpn", "van", "objective",
            "q_z_vessel_present", "q_z_vessel_free", "z_vein_max",
            "vessel_absent_mae_px", "sensor_nodes_over_vein",
            "marker_delta_max_px", "marker_delta_mean_px",
            "trigger_ts", "diverged",
        ])
        print(f"Live iteration log: {log_path}\n")

        history, best = [], None
        completed = 0
        previous_collision_ixs = list(self.collision_ixs)
        try:
            for it in range(num_iterations):
                how = "random" if it < num_random else "acquisition"
                self.bo.suggest_for(model_name, force_random=(it < num_random))
                self.set_contact_params_from_bo()
                self.snapshot_subdir = f"joint_iter{it:03d}"

                print(f"=== joint iteration {it + 1}/{num_iterations} ({how}) ===")
                for key, value in self.bo.params.items():
                    print(f"    {key:26s} {value:.6g}")

                # 1. VESSEL-PRESENT first, to learn WHEN the vessel passes
                #    under the sensor. That timestep is what the vessel-free
                #    comparison run has to be stopped at, so it must be known
                #    before the second run starts.
                self.collision_ixs = active_collision_pairs(True)
                vein_mae, triggered, closest, vein_diverged, trigger_ts = (
                    self._run_vein_slide(run_dir, it)
                )
                vein_points = (
                    None if vein_diverged or vein_mae is None
                    else self.da_last_sim_points.copy()
                )
                # Which surface node the vessel-present run measured; the
                # vessel-free run must report that SAME node.
                _, _, vein_node_index = getattr(
                    self, "da_last_q_z", (float("nan"), 0, None)
                )

                # 2. VESSEL-FREE, stopped at the SAME timestep. Three jobs:
                #    the fidelity term `van` (against the photograph), the
                #    comparison markers (against the vessel-present ones), and
                #    the REFERENCE DEPTH the penetration scale is measured from.
                self.collision_ixs = active_collision_pairs(False)
                no_vein_mae = None
                no_vein_diverged = False
                no_vein_points = None
                no_vein_q_z = float("nan")
                try:
                    self.da_losses = []
                    _, no_vein_diverged = self._run_plain_slide(
                        stop_at_ts=trigger_ts
                    )
                    # Depth WITHOUT the vessel, at the same moment AND on the
                    # SAME NODE the vessel-present run measured. Pinning the
                    # index is essential: an independent argmin would pick a
                    # different physical point, and the difference would then be
                    # mostly "which node won" rather than the vessel's effect.
                    if vein_node_index is None:
                        # The vessel-present run found no node over the vein, so
                        # there is nothing to compare against; leave the
                        # reference undefined and let vpn fall back to 0.
                        no_vein_q_z = float("nan")
                    else:
                        no_vein_q_z, _, _ = self.node_nearest_vein_apex(
                            node_index=vein_node_index
                        )
                    if no_vein_diverged:
                        no_vein_mae = DIVERGED_MEAN_MARKER_ERROR_PX
                    else:
                        self.compute_da_loss(out_dir=run_dir)
                        no_vein_mae = float(self.da_losses[-1])
                        no_vein_points = self.da_last_sim_points.copy()
                except (ValueError, IndexError) as exc:
                    no_vein_diverged = True
                    no_vein_mae = DIVERGED_MEAN_MARKER_ERROR_PX
                    print(f"    vessel-absent run failed: {exc}")

                diverged = vein_diverged or no_vein_diverged
                print(f"    vessel-absent mean    = {no_vein_mae:6.2f} px")

                # 3. SCORE - two normalised terms, differenced:
                #
                #        opt_obj = vpn - van
                #
                #    vpn in [0, 1]  penetration: how far the sensor pressed
                #                   toward the vein, 1 = deepest
                #    van in [0, 1]  infidelity: vessel-absent MAE clamped to
                #                   [0, 55] px and scaled, 0 = perfect match
                #
                #    So the objective rewards pressing into the vessel and
                #    penalises disagreeing with the real sensor, on a common
                #    scale where one unit of each is worth the same. Range is
                #    [-1, +1].
                #
                #    CONTINUOUS, unlike the binary predecessor. That test asked
                #    only "does the sensor reach the vein's depth", which every
                #    gate-passing configuration satisfied identically (q_z
                #    0.06031-0.06037 across a 1.7x range of Young's modulus), so
                #    the objective was constant and BO had nothing to optimise.
                #    A graded penetration gives the GP a gradient to follow.
                #
                #    THE ONLY HARD REJECTION IS DIVERGENCE. There is no
                #    fidelity gate: `van` already penalises a poor match to the
                #    real sensor, continuously and in proportion, so a threshold
                #    on top of it would double-count the same quantity and throw
                #    away the ordering among unfaithful configurations. A
                #    diverged run is different in kind - its numbers are not
                #    measurements of anything - so it is rejected outright at -1.
                a_px = b_px = 0.0
                # Penetration measured against the VESSEL-FREE depth at the same
                # timestep, so it reports the vessel's own contribution rather
                # than how far a soft sensor happened to compress.
                #
                # Both depths are STORED values, not re-measured here: the mesh
                # currently holds the vessel-FREE pose (that run came second), so
                # asking the sensor for its geometry now would compare the
                # vessel-free run against itself and give vpn = 0 always.
                q_z, n_nodes, _ = getattr(
                    self, "da_last_q_z", (float("nan"), 0, None)
                )
                vpn = normalised_penetration(
                    q_z, no_vein_q_z, self.vein.max_z()
                )
                van = min(max(no_vein_mae, 0.0), VAN_CLAMP_PX) / VAN_CLAMP_PX
                if diverged:
                    score = -1.0
                    print(f"    objective             = {score:+.4f} "
                          f"(sensor diverged - NaN/invisible points)")
                else:
                    if vein_points is not None and no_vein_points is not None:
                        d = np.linalg.norm(vein_points - no_vein_points, axis=1)
                        a_px = float(np.max(d))
                        b_px = float(np.mean(d))
                        print(f"    marker delta (sim-vs-sim): max {a_px:.3f} px, "
                              f"mean {b_px:.3f} px")
                    score = vpn - van
                    print(f"    sensor nodes over vein   = {n_nodes}")
                    print(f"    q_z vessel-present       = {q_z:.5f}")
                    print(f"    q_z vessel-FREE (ref)    = {no_vein_q_z:.5f} "
                          f"-> vpn = {vpn:.4f}")
                    print(f"    vessel-absent {no_vein_mae:6.2f} px "
                          f"-> van = {van:.4f}")
                    print(f"    objective (vpn - van) = {score:+.4f}")

                self.bo.my_register(score, model_name, maximise=True)
                record = {
                    "iteration": it,
                    "proposed_by": how,
                    "params": dict(self.bo.params),
                    "max_marker_error_px": a_px,
                    "mean_marker_error_px": b_px,
                    "objective": score,
                    "vessel_absent_mae_px": no_vein_mae,
                    "van_clamp_px": VAN_CLAMP_PX,
                    "z_vein_max": self.vein.max_z(),
                    "sensor_nodes_over_vein": n_nodes,
                    "q_z": None if not np.isfinite(q_z) else q_z,
                    "q_z_vessel_free": (
                        None if not np.isfinite(no_vein_q_z) else no_vein_q_z
                    ),
                    "q_z_node_index": vein_node_index,
                    "vpn": vpn,
                    "van": van,
                    "triggered": bool(triggered),
                    "diverged": bool(vein_diverged),
                    "closest_vein_px": None if closest == float("inf") else closest,
                }
                history.append(record)
                completed = it + 1
                # Flushed immediately, so the file is complete up to this
                # iteration even if the run is stopped or crashes next.
                iteration_log.append({
                    "iteration": it,
                    "proposed_by": how,
                    "vitactip_youngs_modulus": self.bo.params.get(
                        "vitactip_youngs_modulus"
                    ),
                    "normal_stiffness": self.bo.params.get("normal_stiffness"),
                    "vpn": vpn,
                    "van": van,
                    "objective": score,
                    "q_z_vessel_present": None if not np.isfinite(q_z) else q_z,
                    "q_z_vessel_free": (
                        None if not np.isfinite(no_vein_q_z) else no_vein_q_z
                    ),
                    "z_vein_max": self.vein.max_z(),
                    "vessel_absent_mae_px": no_vein_mae,
                    "sensor_nodes_over_vein": n_nodes,
                    "marker_delta_max_px": a_px,
                    "marker_delta_mean_px": b_px,
                    "trigger_ts": trigger_ts,
                    "diverged": int(bool(diverged)),
                })
                # A diverged iteration can never be "best", whatever its score.
                # `score > 0` is NOT a validity test: vpn - van is signed, so
                # a genuinely usable configuration can score negative (poor
                # fidelity outweighing shallow penetration). The only invalid
                # outcome is divergence; among the rest the largest score wins.
                if (not diverged
                        and (best is None or score > best["objective"])):
                    best = record
                    print("    ^ best so far")
                print()
        except BaseException as exc:
            self._report_vein_crash(exc, completed, num_iterations,
                                    dict(self.bo.params), history, run_dir)
            raise
        finally:
            self.collision_ixs = previous_collision_ixs

        path = os.path.join(run_dir, "bo_joint_results.json")
        with open(path, "w") as f:
            json.dump({"best": best, "history": history}, f, indent=4)
        print(f"Joint history: {path}")
        if best is not None:
            print(f"\nBest joint configuration (iteration {best['iteration']}): "
                  f"objective {best['objective']:.2f}")
            print(f"    A = {best['max_marker_error_px']:.2f} px, "
                  f"B = {best['mean_marker_error_px']:.2f} px, "
                  f"vessel-absent = {best['vessel_absent_mae_px']}")
            for key, value in best["params"].items():
                print(f"    {key:26s} {value:.6g}")
        return history, best

    def _run_plain_slide(self, stop_at_ts=None):
        """Run the slide trajectory with the CURRENT parameters, vessel-free.

        No scoring - the caller decides what to measure afterwards.

        `stop_at_ts` stops the run after that many timesteps instead of at the
        trajectory's apex, so a snapshot can be taken at the SAME moment as a
        vessel-present run's trigger. Without it the two snapshots would be at
        different poses and their difference would be dominated by the sensor's
        travel rather than by the vessel.

        Returns (timesteps, diverged). Health-checked every
        HEALTH_CHECK_EVERY_TS timesteps, and short-circuits on failure: an apex
        reached by a blown-up solve is not an apex.
        """
        self.trajectory_ix[None] = self.trajectory_names.index("slide")
        self.vitactip.reset_state()
        self.phantom.reset_state()
        self.set_up_initial_positions_state_and_trajectory()
        self.vitactip.reset_state()
        self.phantom.reset_state()
        self.reset_pid_controller()
        self.visualisation_reset_scene()
        self.reset_exp_sim_traj()
        self.vitactip.extract_markers(0)
        self.compute_mapping_between_experimental_and_sim_markers()
        self.set_dt()
        da_max_ts = _env_int(
            "DIFFTACTILE_DA_MAX_TIMESTEPS_NO_VEIN", DA_MAX_TIMESTEPS_NO_VEIN
        )
        if stop_at_ts is not None:
            da_max_ts = min(da_max_ts, stop_at_ts)
        ts = 0
        while self.last_target_reached[None] != 1 and ts < da_max_ts:
            self.pid_controller_1()
            self.pid_controller_2(ts)
            self.pid_controller_3()
            self.vitactip.set_pose_control_1()
            self.vitactip.set_pose_control_2()
            self.vitactip.set_pose_control_3()
            self.forward_pass_common_part(ts)
            self.copy_frame()
            self.vitactip.extract_markers(SYSTEM_PARAMS.contact.num_sub_frames - 1)
            self.visualisation_update_gui(ts)
            if ts % HEALTH_CHECK_EVERY_TS == 0:
                healthy, reason = self.sensor_state_is_healthy()
                if not healthy:
                    print(f"    SENSOR DIVERGED at ts={ts}: {reason}")
                    return ts, True
            ts += 1
        return ts, False

    def validate_final_params(self, params, run_dir, label="final"):
        """Run all four trajectories ONCE at `params` and report the alignment.

        The confirmation step: everything before this is a model-based claim
        about parameters, and this is the only part that MEASURES the chosen
        configuration on every interaction. Runs the vessel-FREE phantom, since
        that is what the four reference photographs show.

        Writes one alignment overlay per trajectory (`da_overlay_<name>.png`,
        simulated markers RED against real markers GREEN - manuscript Fig. 5)
        plus a JSON of the per-trajectory MAEs.
        """
        print("\n" + "=" * 70)
        print(f" Final validation: 4 trajectories at the {label} parameters")
        print("=" * 70)
        print("\n  parameters:")
        for key, value in params.items():
            print(f"    {key:26s} {value:.6g}")
        print()

        # Adopt the configuration, then run vessel-free to match the
        # photographs. An EMPTY collision list, not [0]: pair 0 is disabled
        # project-wide (the phantom is pinned, so it never resisted anyway), so
        # "vessel-free" means no contact pairs resolved at all.
        self.bo.params = dict(params)
        self.set_contact_params_from_bo()
        previous_collision_ixs = list(self.collision_ixs)
        self.collision_ixs = active_collision_pairs(False)
        results = {}
        try:
            for i in range(self.trajectories.shape[0]):
                name = self.trajectory_names[i]
                self.trajectory_ix[None] = i
                self.snapshot_subdir = f"{label}_{name}"
                self.vitactip.reset_state()
                self.phantom.reset_state()
                self.set_up_initial_positions_state_and_trajectory()
                self.vitactip.reset_state()
                self.phantom.reset_state()
                self.reset_pid_controller()
                self.visualisation_reset_scene()
                self.reset_exp_sim_traj()
                self.vitactip.extract_markers(0)
                self.compute_mapping_between_experimental_and_sim_markers()
                self.set_dt()

                da_max_ts = _env_int(
                    "DIFFTACTILE_DA_MAX_TIMESTEPS_NO_VEIN", DA_MAX_TIMESTEPS_NO_VEIN
                )
                ts = 0
                diverged = None
                while self.last_target_reached[None] != 1 and ts < da_max_ts:
                    self.pid_controller_1()
                    self.pid_controller_2(ts)
                    self.pid_controller_3()
                    self.vitactip.set_pose_control_1()
                    self.vitactip.set_pose_control_2()
                    self.vitactip.set_pose_control_3()
                    self.forward_pass_common_part(ts)
                    self.copy_frame()
                    self.vitactip.extract_markers(
                        SYSTEM_PARAMS.contact.num_sub_frames - 1
                    )
                    self.visualisation_update_gui(ts)
                    if ts % HEALTH_CHECK_EVERY_TS == 0:
                        healthy, reason = self.sensor_state_is_healthy()
                        if not healthy:
                            diverged = reason
                            break
                    ts += 1
                if diverged:
                    # Reported as a failure rather than scored: a validation
                    # number from a blown-up solve would be meaningless, and
                    # silently recording it would misrepresent the parameters.
                    print(f"    {name:9s} DIVERGED at ts={ts} ({diverged})")
                    results[name] = None
                    continue

                self.da_losses = []
                try:
                    self.compute_da_loss(out_dir=run_dir)
                    mae = self.da_losses[-1]
                except (ValueError, IndexError) as exc:
                    print(f"    {name:9s} FAILED ({exc})")
                    results[name] = None
                    continue
                results[name] = float(mae)
                print(f"    {name:9s} MAE = {mae:6.2f} px "
                      f"({mae * PX_TO_MM:.2f} mm)  [{ts} timesteps]")
        finally:
            self.collision_ixs = previous_collision_ixs

        usable = [v for v in results.values() if v is not None]
        summary = {
            "label": label,
            "params": dict(params),
            "per_trajectory_mae_px": results,
            "per_trajectory_mae_mm": {
                k: (v * PX_TO_MM if v is not None else None)
                for k, v in results.items()
            },
            "mean_mae_px": float(np.mean(usable)) if usable else None,
            "total_mae_px": float(np.sum(usable)) if usable else None,
        }
        if usable:
            print(f"\n    {'mean':9s} {np.mean(usable):6.2f} px "
                  f"({np.mean(usable) * PX_TO_MM:.2f} mm)")
            print(f"    {'total':9s} {np.sum(usable):6.2f} px "
                  f"({np.sum(usable) * PX_TO_MM:.2f} mm)")
        path = os.path.join(run_dir, f"{label}_validation.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\n  Statistics: {path}")
        print(f"  Alignment figures (Fig. 5 panels): "
              f"{os.path.join(run_dir, 'da_overlay_<name>.png')}")
        return summary

    def score_current_configuration(self, run_dir=None):
        """One joint-objective evaluation of the CURRENT parameters.

        The same measurement `domain_adaptation_joint` performs per iteration -
        vessel-present slide, then vessel-free slide at the matched timestep and
        node - but with no proposal step, so it scores whatever is already
        loaded. Shares `_run_vein_slide`, `compute_da_loss` and
        `normalised_penetration` with the search, which is what makes the number
        comparable to a BO iteration's rather than merely similar.

        Writes `score.json` and returns the same dict.
        """
        run_dir = run_dir or repo_path("difftactile/output")
        os.makedirs(run_dir, exist_ok=True)
        previous_collision_ixs = list(self.collision_ixs)
        try:
            # 1. VESSEL-PRESENT.
            self.collision_ixs = active_collision_pairs(True)
            self.snapshot_subdir = "score_vein"
            vein_mae, triggered, closest, vein_diverged, trigger_ts = (
                self._run_vein_slide(run_dir, 0)
            )
            vein_points = (
                None if vein_diverged or vein_mae is None
                else self.da_last_sim_points.copy()
            )
            q_z, n_nodes, vein_node_index = getattr(
                self, "da_last_q_z", (float("nan"), 0, None)
            )

            # 2. VESSEL-FREE, matched timestep and node.
            self.collision_ixs = active_collision_pairs(False)
            self.snapshot_subdir = "score_no_vein"
            no_vein_mae, no_vein_diverged = None, False
            no_vein_points, no_vein_q_z = None, float("nan")
            try:
                self.da_losses = []
                _, no_vein_diverged = self._run_plain_slide(
                    stop_at_ts=trigger_ts
                )
                if vein_node_index is not None:
                    no_vein_q_z, _, _ = self.node_nearest_vein_apex(
                        node_index=vein_node_index
                    )
                if no_vein_diverged:
                    no_vein_mae = DIVERGED_MEAN_MARKER_ERROR_PX
                else:
                    self.compute_da_loss(out_dir=run_dir)
                    no_vein_mae = float(self.da_losses[-1])
                    no_vein_points = self.da_last_sim_points.copy()
            except (ValueError, IndexError) as exc:
                no_vein_diverged = True
                no_vein_mae = DIVERGED_MEAN_MARKER_ERROR_PX
                print(f"    vessel-absent run failed: {exc}")

            diverged = vein_diverged or no_vein_diverged
            van = min(max(no_vein_mae, 0.0), VAN_CLAMP_PX) / VAN_CLAMP_PX
            vpn = normalised_penetration(q_z, no_vein_q_z, self.vein.max_z())
            a_px = b_px = 0.0
            if vein_points is not None and no_vein_points is not None:
                d = np.linalg.norm(vein_points - no_vein_points, axis=1)
                a_px, b_px = float(np.max(d)), float(np.mean(d))
            score = -1.0 if diverged else vpn - van

            print("\n" + "-" * 70)
            print(" Result")
            print("-" * 70)
            if diverged:
                print(f"\n  SENSOR DIVERGED -> objective {score:+.4f}")
            else:
                print(f"\n  sensor nodes over vein   = {n_nodes}")
                print(f"  q_z vessel-present       = {q_z:.5f}")
                print(f"  q_z vessel-FREE (ref)    = {no_vein_q_z:.5f}")
                print(f"  z_vein_max (apex)        = {self.vein.max_z():.5f}")
                print(f"  marker delta (sim-vs-sim): max {a_px:.3f} px, "
                      f"mean {b_px:.3f} px")
                print(f"\n  vpn (penetration)        = {vpn:.4f}  "
                      f"[0, {VPN_WEIGHT}]")
                print(f"  van (infidelity)         = {van:.4f}  "
                      f"(vessel-absent MAE {no_vein_mae:.2f} px / "
                      f"{VAN_CLAMP_PX:.0f})")
                print(f"  objective (vpn - van)    = {score:+.4f}")

            result = {
                "phantom_contact_enabled": phantom_contact_enabled(),
                "collision_pairs_with_vein": active_collision_pairs(True),
                "params": {
                    "vitactip_youngs_modulus": float(
                        self.vitactip.youngs_modulus[None]
                    ),
                    "vitactip_poissons_ratio": float(
                        self.vitactip.poissons_ratio[None]
                    ),
                    "normal_stiffness_vein": float(self.normal_stiffness[2]),
                    "normal_damping_vein": float(self.normal_damping[2]),
                },
                "vpn": vpn,
                "van": van,
                "objective": score,
                "q_z_vessel_present": None if not np.isfinite(q_z) else q_z,
                "q_z_vessel_free": (
                    None if not np.isfinite(no_vein_q_z) else no_vein_q_z
                ),
                "z_vein_max": self.vein.max_z(),
                "vessel_absent_mae_px": no_vein_mae,
                "sensor_nodes_over_vein": n_nodes,
                "marker_delta_max_px": a_px,
                "marker_delta_mean_px": b_px,
                "trigger_ts": trigger_ts,
                "diverged": bool(diverged),
            }
            path = os.path.join(run_dir, "score.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"\n  Written to: {path}")
            return result
        finally:
            self.collision_ixs = previous_collision_ixs

    def _run_vein_slide(self, run_dir, it):
        """One vessel-present slide, scored when the vessel reaches the centre.

        Returns (mae, triggered, closest_px, diverged, trigger_ts). `mae` is
        None if the vessel never came within the trigger radius; `diverged` is
        True if the sensor's FEM state failed a health check, which the caller
        must score separately - a diverged sensor produces huge marker errors
        that would otherwise LOOK like excellent vessel sensitivity.
        `trigger_ts` is the timestep the snapshot was taken at, so a vessel-free
        run can be stopped at the same pose for comparison.

        The state is checked every HEALTH_CHECK_EVERY_TS timesteps and the
        trajectory short-circuits on failure: once a solve blows up it does not
        recover, so the remaining timesteps only waste compute.

        SHORT-CIRCUITS at the first trigger: once the snapshot is taken there is
        nothing left to measure, so the remaining timesteps are pure cost. The
        vessel typically passes under the sensor early in the slide, so this cuts
        most of a ~398-timestep trajectory - the dominant expense of the whole
        stage.

        Nothing downstream depends on the trajectory finishing: the score is the
        MAE at the trigger instant, and `last_target_reached` is only used to end
        the loop. (This differs from the vessel-ABSENT objective, which is
        measured at the trajectory's apex and therefore must run to the end.)
        """
        name = "slide"
        self.trajectory_ix[None] = self.trajectory_names.index(name)
        self.vitactip.reset_state()
        self.phantom.reset_state()
        self.set_up_initial_positions_state_and_trajectory()
        self.vitactip.reset_state()
        self.phantom.reset_state()
        self.reset_pid_controller()
        self.visualisation_reset_scene()
        self.reset_exp_sim_traj()
        self.vitactip.extract_markers(0)
        self.compute_mapping_between_experimental_and_sim_markers()
        self.set_dt()

        da_max_ts = _env_int(
            "DIFFTACTILE_DA_MAX_TIMESTEPS_VEIN", DA_MAX_TIMESTEPS_VEIN
        )
        mae, closest, trigger_ts = None, float("inf"), None
        # Depth geometry at the trigger; (reaches, z_s, z_v). NaN until a
        # snapshot is actually taken.
        self.da_last_depth = (False, float("nan"), float("nan"), 0)
        self.da_last_q_z = (float("nan"), 0, None)
        ts = 0
        while self.last_target_reached[None] != 1 and ts < da_max_ts:
            self.pid_controller_1()
            self.pid_controller_2(ts)
            self.pid_controller_3()
            self.vitactip.set_pose_control_1()
            self.vitactip.set_pose_control_2()
            self.vitactip.set_pose_control_3()
            self.forward_pass_common_part(ts)
            self.copy_frame()
            self.vitactip.extract_markers(SYSTEM_PARAMS.contact.num_sub_frames - 1)
            self.visualisation_update_gui(ts)
            if ts % HEALTH_CHECK_EVERY_TS == 0:
                healthy, reason = self.sensor_state_is_healthy()
                if not healthy:
                    print(f"    SENSOR DIVERGED at ts={ts}: {reason}")
                    return None, False, closest, True, ts
            # The projection must be refreshed before the proximity test:
            # visualisation_update_gui() only fills it when a window is drawn.
            self.visualisation_project_vein_2d()
            self.visualisation_prepare_tactile_readout_data_fp()
            triggered, d = self.vein_over_sensor_centre()
            closest = min(closest, d)
            if triggered and mae is None:
                try:
                    mae = self.compute_vein_da_loss(out_dir=run_dir,
                                                    tag=f"vein_iter{it:03d}")
                except (ValueError, IndexError) as exc:
                    print(f"    scoring failed at ts={ts}: {exc}")
                    return None, True, closest, False, ts
                # Depth geometry AT THE TRIGGER, while the vessel-present state
                # is still live. Reading it after the vessel-free comparison run
                # would report that run's geometry instead - the mesh fields are
                # overwritten in place.
                self.da_last_depth = self.sensor_reaches_vein_depth()
                # Raw depth only. `vpn` cannot be formed yet: its reference is
                # the VESSEL-FREE q_z at this same moment, which has not been
                # simulated at this point in the iteration.
                self.da_last_q_z = self.node_nearest_vein_apex()
                # Snapshot taken - stop here rather than simulating the rest of
                # the slide for a measurement that has already been made.
                trigger_ts = ts
                ts += 1
                break
            ts += 1
        return mae, mae is not None, closest, False, trigger_ts

    def _report_recommendation(self, rec, run_dir):
        """Print the combined recommendation and write it beside the history."""
        print("\n" + "=" * 70)
        print(" Combined recommendation (all four models)")
        print("=" * 70)
        print(
            "\nMinimises the SUM of the four models' predicted pixel errors, "
            "unscaled -\nNOT an average of the four best-observed points.\n"
        )
        print("  parameters:")
        for key, value in rec["params"].items():
            print(f"    {key:26s} {value:.6g}")
        total = rec["predicted_total_mae_px"]
        print("\n  predicted MAE at this configuration (px):")
        for name, mae in sorted(rec["predicted_mae_px"].items()):
            sigma = rec["predicted_sigma_px"][name]
            print(f"    {name:9s} {mae:7.2f} +/- {sigma:.2f}")
        print(f"    {'TOTAL':9s} {total:7.2f} px ({total * PX_TO_MM:.2f} mm)")
        print("\n  each model's own best OBSERVED point, for comparison:")
        for name, d in sorted(rec["per_model_best_observed"].items()):
            print(f"    {name:9s} {d['mae_px']:7.2f} px")
        spread = rec["argmin_spread"]
        print(f"\n  argmin spread across models: {spread:.3f}")
        print(
            "    (0 = the four models agree on where the optimum is; ~0.29 is "
            "as\n     scattered as random points. A large value means the "
            "objectives\n     genuinely conflict and a single recommendation is "
            "hiding a trade-off.)"
        )
        print(
            "\n  NOT YET MEASURED: this is a model prediction. Confirm it by "
            "running\n  all four trajectories at these parameters before "
            "adopting them."
        )
        path = os.path.join(run_dir, "bo_recommendation.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=4)
        print(f"\n  Written to: {path}")

    def _report_bo_results(self, history, best, run_dir, best_by_scope=None):
        """Print the BO outcome and write it to JSON and a convergence figure."""
        print("=" * 70)
        print(" Bayesian optimisation results")
        print("=" * 70)
        print(f"\nEvery configuration tried ({len(history)} total), worst to best:\n")
        print(f"  {'iter':>4}  {'MAE px':>8}  {'MAE mm':>7}  {'by':>12}  {'scored on':>10}")
        for r in sorted(history, key=lambda r: -r["mae_px"]):
            print(
                f"  {r['iteration']:>4}  {r['mae_px']:>8.2f}  "
                f"{r['mae_mm']:>7.3f}  {r['proposed_by']:>12}  "
                f"{r.get('scored_on', '?'):>10}"
            )

        # Under a schedule the MAEs above are NOT mutually comparable - each is
        # measured against whichever trajectories that iteration ran. Report the
        # best within each scope so the ranking means something.
        if best_by_scope:
            print("\nBest per trajectory scope (these ARE comparable within a row):\n")
            print(f"  {'scored on':>10}  {'iter':>4}  {'MAE px':>8}  {'MAE mm':>7}")
            for scope, r in sorted(best_by_scope.items()):
                print(
                    f"  {scope:>10}  {r['iteration']:>4}  {r['mae_px']:>8.2f}  "
                    f"{r['mae_mm']:>7.3f}"
                )

        if best is not None:
            print(f"\nBest configuration (iteration {best['iteration']}, "
                  f"scored on {best.get('scored_on', '?')}):")
            print(f"  MAE = {best['mae_px']:.2f} px "
                  f"({best['mae_mm']:.3f} mm)")
            for name, mae in best["per_trajectory_px"].items():
                print(f"    {name:9s} {mae:6.2f} px ({mae * PX_TO_MM:.2f} mm)")
            print("\n  parameters:")
            for key, value in best["params"].items():
                print(f"    {key:26s} {value:.6g}")
            print(
                "\n  To adopt these, copy them into system-params.json "
                "(contact.* and the Young's modulus / Poisson's ratio entries)."
            )

        results_path = os.path.join(run_dir, "bo_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "best": best,
                    "best_by_scope": best_by_scope or {},
                    "history": history,
                },
                f,
                indent=4,
            )
        print(f"\nFull history: {results_path}")

        # Convergence, PLOTTED PER SCOPE. A single running minimum over the whole
        # history would be meaningless under a schedule: it would step down every
        # time the objective changed to a trajectory that simply scores lower,
        # which reads as progress but is not. One series per `scored_on` keeps
        # each curve a like-for-like comparison, and the running best is taken
        # within a series.
        plt.figure(figsize=(10, 6))
        scopes = []
        for r in history:
            s = r.get("scored_on", "?")
            if s not in scopes:
                scopes.append(s)
        for scope in scopes:
            pts = [(r["iteration"], r["mae_px"]) for r in history
                   if r.get("scored_on", "?") == scope]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            line, = plt.plot(xs, ys, "o-", label=f"{scope}")
            if len(ys) > 1:
                plt.plot(xs, np.minimum.accumulate(ys), "--", linewidth=2,
                         color=line.get_color(), alpha=0.6,
                         label=f"{scope} (best so far)")
        plt.xlabel("Bayesian optimisation iteration")
        plt.ylabel("MAE (px)")
        plt.title("Domain adaptation: marker alignment error, by trajectory scope")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        finish_plot(plt, os.path.join(run_dir, "bo_convergence.png"))
        print(f"Convergence figure: {os.path.join(run_dir, 'bo_convergence.png')}")
        print(f"Alignment overlays (best config): "
              f"{os.path.join(run_dir, 'best_da_overlay_<name>.png')}")

    def collect_training_data(self, num_loops=None, substeps=(0, 1),
                              trajectory_ixs=None, with_vein=COLLECT_VEIN_PAIR):
        """Run the randomised trajectories and write one .npz per trial.

        Defaults reproduce the training-data collection exactly; the keyword
        arguments exist for `vessel_map_trajectory_main()`, which wants a
        single vein-present slide rather than the whole loop:

          num_loops       outer loops (default DIFFTACTILE_NUM_LOOPS / config)
          substeps        which of the two substeps to run; substep 0 is the
                          vein-present one when `with_vein`
          trajectory_ixs  trajectory types to execute (default
                          DIFFTACTILE_TRAJECTORIES)
          with_vein       enable the sensor<->vein contact pair on substep 0
        """
        # self.clear_temp_images()
        # self.clear_npz()
        file_num = 0
        # Number of outer loops; each contributes 2 substeps x 4 trajectories.
        # DIFFTACTILE_NUM_LOOPS=1 gives an 8-trial smoke test instead of the full run.
        if num_loops is None:
            num_loops = _env_int(
                "DIFFTACTILE_NUM_LOOPS", SYSTEM_PARAMS.contact.num_training_trajectories
            )
        if num_loops != SYSTEM_PARAMS.contact.num_training_trajectories:
            print(
                f"DIFFTACTILE_NUM_LOOPS={num_loops} (full run would be "
                f"{SYSTEM_PARAMS.contact.num_training_trajectories}); "
                f"expecting {num_loops * 8} trials"
            )
        if trajectory_ixs is None:
            trajectory_ixs = list(_trajectory_indices())
        for i in range(num_loops):
            for j in substeps:
                print(f"training trajectory: {i}/{num_loops - 1}; substep: {j}/5")
                self.generate_trajectories()
                # Pair 0 is disabled project-wide (see __init__), so a
                # "vein-absent" substep resolves NO contact pairs at all.
                if with_vein and j < 1:
                    self.collision_ixs = active_collision_pairs(True)
                else:
                    self.collision_ixs = active_collision_pairs(False)
                print(
                    f"  substep {j}: collision_ixs={self.collision_ixs} "
                    f"({'WITH vein' if 2 in self.collision_ixs else 'no vein'})"
                )
                self.randomise_contact_params()
                # Which of the four trajectory types to execute. The PUBLISHED
                # dataset (pickle_20250901_220921*) is entirely type 3,
                # "slide (vein)" — it was collected when this loop read
                # `range(3, 4)`; a later commit (0e7280a) widened it to all four,
                # which is why a default run now also produces types 0/1/2.
                # Type 0 ("press (no vein)") terminates in ~36 timesteps, below
                # the `ts > 80` threshold at which recording starts, so it yields
                # empty arrays by design.
                # Set DIFFTACTILE_TRAJECTORIES=3 to reproduce the published
                # dataset; the default keeps the current all-four behaviour.
                for traj_ix in trajectory_ixs:
                    self.trajectory_ix[None] = traj_ix
                    trajectory_name = self.trajectory_names[self.trajectory_ix[None]]
                    # print(f'executing trajectory: {trajectory_name}')
                    self.set_up_initial_positions_state_and_trajectory()
                    # self.vein_sparse_to_dense()
                    self.reset_pid_controller()
                    self.visualisation_reset_scene()
                    self.reset_exp_sim_traj()
                    self.vitactip.extract_markers(0)
                    # self.compute_mapping_between_experimental_and_sim_markers()
                    self.set_dt(verbose=True)
                    self.fp()
                    # self.print_contact_params()
                    for ts in range(SYSTEM_PARAMS.meta.max_timesteps_per_trajectory):
                        self.pid_controller_1()
                        self.pid_controller_2(ts)
                        self.pid_controller_3()
                        self.vitactip.set_pose_control_1()
                        self.vitactip.set_pose_control_2()
                        self.vitactip.set_pose_control_3()
                        self.forward_pass_common_part(ts)
                        self.copy_frame()
                        self.vitactip.extract_markers(
                            SYSTEM_PARAMS.contact.num_sub_frames - 1
                        )
                        self.vitactip.mark_surface_nodes_in_contact(
                            SYSTEM_PARAMS.contact.num_sub_frames - 1
                        )
                        self.visualisation_update_gui(ts)
                        if ts % 10 == 0:
                            self.record_vitactip_mesh()
                        # target = self.current_target_idx[None]
                        if (
                            ts > 80
                        ):
                            self.record_training_data_point()
                        # if ts % 100 == 0:
                        #     self.save_sensor_mesh_to_npz()
                            # print(f"ts={ts}; sensor mesh saved")
                        if self.last_target_reached[None] == 1:
                            break
                    self.write_training_data_to_file(file_num)
                    file_num += 1
                    self.write_vitactip_mesh_to_file()
                    print(
                        f"training trajectory: {i}/{num_loops - 1}; substep: {j}/5 done"
                    )
        print("training data collection done")
        print("all done")


def vessel_map_trajectory_main():
    """Simulate ONE vein-present slide, with sensor poses, for the Sim->Sim map.

    The published simulated dataset records marker pixels and the vein's image
    projection but NOT the sensor pose (pose recording was added later, see
    `record_training_data_point`), so none of its trajectories can be
    reprojected onto the phantom plane the way the real datasets are. Rather
    than regenerate 500 trajectories (~5 h GPU) for one figure, this simulates a
    single slide from the same generator (`get_slide_trajectory`, vein contact
    pair enabled, same randomised contact parameters), under its own seed so it
    is a fresh draw from the training distribution and not one of the training
    files. Its .npz carries `T_BA` per frame and the vein's world centreline.

    Output: difftactile/output/vessel_map_sim/raw/trajectory_0000.npz, then
    Hungarian-reordered into .../raw_reordered_dense/ (the layout cnn/dataset.py
    reads) by pre_process_sim_data. Both are small and shipped in the data
    bundle, so the map can be regenerated without a simulator; run this again
    only to draw a different trajectory (change DIFFTACTILE_SEED).

    Seed: DIFFTACTILE_SEED, default 2026 here - deliberately NOT the dataset's
    42, whose first slide is trajectory_0000 of the published TRAINING split.
    """
    os.environ.setdefault("DIFFTACTILE_SEED", "2026")
    seed = seed_everything(deterministic_torch=False)
    print(f"Seed: {seed} (override with DIFFTACTILE_SEED)")

    out_root = repo_path("difftactile/output/vessel_map_sim")
    raw_dir = os.path.join(out_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for stale in os.listdir(raw_dir):
        os.remove(os.path.join(raw_dir, stale))

    if RUN_ON_LAB_MACHINE:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR,
                arch=ti.cuda,
                device_memory_GB=float(os.environ.get("TI_DEVICE_MEMORY_GB", 9)))
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)
    contact_model = Contact()
    contact_model.visualisation_set_up_gui()
    contact_model.save_tactile_sensor_mesh_node_mapping_to_pickle()
    contact_model.trajectory_ix[None] = 0
    contact_model.set_up_initial_positions_state_and_trajectory()
    contact_model.reset_pid_controller()
    contact_model.reset_exp_sim_traj()
    contact_model.get_keypoint_indices_and_validate()
    contact_model.training_data_dir_override = raw_dir

    # One loop, the vein-present substep only, the slide only. Type 3 is
    # "slide" in the list generate_trajectories() installs at the top of the
    # collection loop (press, twist_z, twist_x, slide) - the same index the
    # published dataset was collected with (DIFFTACTILE_TRAJECTORIES=3).
    slide_ix = 3
    contact_model.collect_training_data(
        num_loops=1, substeps=(0,), trajectory_ixs=[slide_ix], with_vein=True
    )

    # Reorder into the base-graph marker order the GNN expects.
    from difftactile.data_analysis.sim.pre_process_sim_data import PreProcessSimData
    os.environ["DIFFTACTILE_SIM_RAW_DIR"] = raw_dir
    PreProcessSimData.sim_marker_tracker()
    print(f"Sim vessel-map trajectory written under {out_root}")
    print("all done")


def alignment_figures_main():
    """Entrypoint: the four alignment panels for the CURRENT configuration.

    Runs each of press / twist_z / twist_x / slide ONCE at the parameters in
    system-params.json, caches the marker positions, and draws the manuscript
    figures - simulated markers red, real green, a blue segment joining each
    corresponding pair, on white.

    USES THE CACHE WHEN IT EXISTS. Each trajectory is minutes of simulation and
    the figures are pure styling on top of two point sets, so re-simulating to
    change a colour would be wasted time. DIFFTACTILE_ALIGNMENT_SOURCE points at
    an existing `markers_*.npz` set (e.g. a published BO run's directory);
    DIFFTACTILE_ALIGNMENT_FORCE=1 re-simulates even when a cache is present.
    """
    from difftactile.data_analysis.experiment.alignment_figures import (
        TRAJECTORY_ORDER, generate_alignment_figures,
    )

    source = os.environ.get("DIFFTACTILE_ALIGNMENT_SOURCE")
    force = os.environ.get("DIFFTACTILE_ALIGNMENT_FORCE", "0") == "1"
    if source and not force:
        cached = all(
            os.path.exists(os.path.join(source, f"markers_{n}.npz"))
            for n in TRAJECTORY_ORDER
        )
        if cached:
            print(f"Using cached marker positions from {source}")
            generate_alignment_figures(source)
            return

    run_dir = repo_path(
        f"difftactile/output/alignment_figures/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print("No usable cache - running the four trajectories.")

    seed = seed_everything(deterministic_torch=False)
    print(f"Seed: {seed} (override with DIFFTACTILE_SEED)")
    if RUN_ON_LAB_MACHINE:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR,
                arch=ti.cuda,
                device_memory_GB=float(os.environ.get("TI_DEVICE_MEMORY_GB", 9)))
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)

    from difftactile.data_analysis.experiment.domain_adaptation import (
        extract_real_marker_positions,
    )
    extract_real_marker_positions()

    contact_model = Contact()
    contact_model.visualisation_set_up_gui()
    contact_model.generate_trajectories()
    contact_model.trajectory_ix[None] = 0
    contact_model.set_up_initial_positions_state_and_trajectory()
    contact_model.reset_pid_controller()
    contact_model.reset_exp_sim_traj()
    contact_model.get_keypoint_indices_and_validate()

    params = {
        "vitactip_youngs_modulus": float(
            contact_model.vitactip.youngs_modulus[None]
        ),
        "vitactip_poissons_ratio": float(
            contact_model.vitactip.poissons_ratio[None]
        ),
    }
    print("\n  configuration under test:")
    for key, value in params.items():
        print(f"    {key:26s} {value:.6g}")
    print(f"    {'normal_stiffness (vein)':26s} "
          f"{contact_model.normal_stiffness[2]:.6g}")

    # validate_final_params runs all four vessel-free and, via compute_da_loss,
    # writes both the photo overlays and the markers_*.npz this needs.
    contact_model.validate_final_params(params, run_dir, label="alignment")

    print("\nDrawing the manuscript panels:")
    generate_alignment_figures(run_dir)
    print(f"\nAll artifacts: {run_dir}")


def score_current_params_main():
    """Entrypoint: score the CONFIG AS IT STANDS, with no optimisation.

    Runs exactly one iteration of the joint objective - the vessel-present slide
    and the vessel-free slide - at whatever parameters system-params.json
    currently holds, and reports vpn, van and vpn - van.

    A SANITY CHECK, not a search. `domain_adaptation.sh` proposes parameters and
    hunts for good ones; this one proposes nothing, so the number it prints is
    attributable entirely to the configuration on disk. That makes it the tool
    for questions like "does contact resolve at all with the phantom pair
    switched on", where a search would confound the answer with its own
    exploration.

    Honours the pair-0 seam, so `DIFFTACTILE_PHANTOM_CONTACT=1` scores the same
    configuration with sensor<->phantom contact enabled.
    """
    run_dir = repo_path(
        f"difftactile/output/score_params/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if is_headless():
        print("Headless: no GGUI window will be created.")
    seed = seed_everything(deterministic_torch=False)
    print(f"Seed: {seed} (override with DIFFTACTILE_SEED)")

    if RUN_ON_LAB_MACHINE:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR,
                arch=ti.cuda,
                device_memory_GB=float(os.environ.get("TI_DEVICE_MEMORY_GB", 9)))
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)

    from difftactile.data_analysis.experiment.domain_adaptation import (
        extract_real_marker_positions,
    )
    extract_real_marker_positions()

    contact_model = Contact()
    contact_model.visualisation_set_up_gui()
    contact_model.generate_trajectories()
    contact_model.trajectory_ix[None] = 0
    contact_model.set_up_initial_positions_state_and_trajectory()
    contact_model.reset_pid_controller()
    contact_model.reset_exp_sim_traj()
    contact_model.get_keypoint_indices_and_validate()

    print("\n" + "=" * 70)
    print(" Scoring the configuration in system-params.json (no optimisation)")
    print("=" * 70)
    on = phantom_contact_enabled()
    print(f"\n  sensor<->phantom contact pair: "
          f"{'ENABLED' if on else 'disabled'}")
    if on:
        c = SYSTEM_PARAMS.contact
        print(f"    normal_stiffness       {c.normal_stiffness[0]}")
        print(f"    tangential_stiffness   {c.tangential_stiffness[0]}")
        print(f"    normal_damping         {c.normal_damping[0]}")
        print(f"    coulomb_friction_coeff {c.coulomb_friction_coeff[0]}")
    print(f"  sensor<->vein contact pair:    ENABLED")
    print(f"    normal_stiffness       {SYSTEM_PARAMS.contact.normal_stiffness[2]}")
    print(f"    normal_damping         {SYSTEM_PARAMS.contact.normal_damping[2]}")
    print(f"\n  sensor material (as loaded):")
    print(f"    youngs_modulus         "
          f"{contact_model.vitactip.youngs_modulus[None]:.6g}")
    print(f"    poissons_ratio         "
          f"{contact_model.vitactip.poissons_ratio[None]:.6g}")

    start = time.perf_counter()
    result = contact_model.score_current_configuration(run_dir=run_dir)
    print(f"\nScoring took {time.perf_counter() - start:.2f} s")
    print(f"All artifacts: {run_dir}")
    return result


def domain_adaptation_main():
    """Entrypoint: calibrate the simulator against the real sensor, via BO.

    Default (DIFFTACTILE_DA_MODE=joint): one Bayesian optimisation over sensor
    stiffness and sensor<->vein contact stiffness, scoring each proposal on a
    vessel-ABSENT slide (fidelity to the real photograph) and a vessel-PRESENT
    slide (how visibly the vessel deforms the sensor) together - see
    `domain_adaptation_joint()`. The chosen configuration is then validated on
    all four canonical interactions (press, twist about z, twist about x,
    slide), which is what the manuscript's Fig. 5 shows. `staged` keeps the
    older sequential design.

    NOT DIFFERENTIABLE. An earlier design backpropagated through the Taichi
    simulation; that was abandoned, and all of the machinery supporting it has
    been removed. BO treats the simulator as a black box, so only forward
    simulation is needed.

    Every run writes into its OWN timestamped directory under
    difftactile/output/domain_adaptation/, so repeated runs accumulate instead
    of overwriting - the same convention the training pipeline uses.
    """
    run_dir = repo_path(
        f"difftactile/output/domain_adaptation/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # SNAPSHOTS ARE ALWAYS ON for domain adaptation, written inside this run's
    # own timestamped directory (snapshots/iterNNN/<trajectory>_ts<NNNN>.png).
    # A DA run is a calibration whose whole output is "how well do simulated and
    # real markers line up", and the rendered frames are the only way to see
    # WHY a configuration scored as it did - a numeric MAE cannot show that a
    # trajectory slid the wrong way or that an inclusion was present. They were
    # previously opt-in via DIFFTACTILE_SNAPSHOT_DIR, so the frames existed only
    # for runs where someone had thought to ask in advance, which is exactly
    # when they are least likely to be needed.
    #
    # setdefault, not assignment: an explicit DIFFTACTILE_SNAPSHOT_DIR still
    # wins, so a caller can redirect them elsewhere.
    #
    # ONLY WHEN A DISPLAY IS AVAILABLE. Taichi GGUI's offscreen path
    # (show_window=False) SEGFAULTS in this image - it dies partway through the
    # first trajectory with a core dump, which is why snapshots were opt-in in
    # the first place. Defaulting them on unconditionally would therefore turn
    # every headless DA run into a crash. With a real display GGUI has a window
    # and save_image() is reliable, so the frames come for free alongside the
    # live view.
    #
    # A headless run consequently still produces no snapshots. That is a known
    # limitation of the Taichi build, not something to work around here; run DA
    # with a display when the frames are wanted.
    if not is_headless():
        os.environ.setdefault(
            "DIFFTACTILE_SNAPSHOT_DIR", os.path.join(run_dir, "snapshots")
        )
    else:
        print(
            "Headless: skipping snapshots (Taichi GGUI segfaults offscreen in "
            "this image). Run with a DISPLAY to capture them."
        )

    # Seed BEFORE anything stochastic runs, so a DA run is reproducible.
    #
    # `NP_RNG` (main/constants.py) is `default_rng()` with no seed, and this path
    # draws from it in three places: the random phase of the BO search
    # (`BoGp.my_suggest_random`), the tangential_stiffness fraction drawn every
    # iteration, and generate_trajectories(), which randomises the four DA
    # interactions themselves. Without this call each run explored a different
    # search space AND scored it against differently-randomised trajectories, so
    # two runs of the same code were not comparable.
    #
    # `deterministic_torch=False`: there is no torch in this path (the simulator
    # is Taichi), so the deterministic-kernel switches would only cost the
    # startup work and set CUBLAS_WORKSPACE_CONFIG for no reason. Taichi's own
    # kernels are unaffected by torch's determinism flags either way.
    #
    # The GP is separately deterministic already - BayesianOptimization is
    # constructed with random_state=1 in bo_gp.py.
    seed = seed_everything(deterministic_torch=False)
    print(f"Seed: {seed} (override with DIFFTACTILE_SEED)")

    if RUN_ON_LAB_MACHINE:
        ti.init(
            debug=False,
            offline_cache=False,
            log_level=ti.ERROR,
            arch=ti.cuda,
            device_memory_GB=float(os.environ.get("TI_DEVICE_MEMORY_GB", 9)),
        )
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)

    # The REAL marker positions the simulation is measured against, extracted
    # from the four reference photographs. compute_da_loss() loads these, and
    # nothing else generates them - the photographs ship with the repository but
    # the .npz files did not, so DA used to die on a missing file.
    from difftactile.data_analysis.experiment.domain_adaptation import (
        extract_real_marker_positions,
    )
    print("Real marker positions (from the reference photographs):")
    extract_real_marker_positions()

    contact_model = Contact()

    # NO SUBSURFACE VEIN during domain adaptation.
    #
    # `Contact.__init__` defaults to `[0, 2]` - pair 0 is sensor<->phantom and
    # pair 2 is sensor<->vein - and `collect_training_data()` overrides it per
    # substep, but this path never did, so every DA simulation ran with a vein
    # embedded in the phantom. The REAL sensor photographs these are scored
    # against were taken on a plain phantom with no inclusion, so the simulation
    # was being fitted to a physically different object: the vein stiffens the
    # region under the sensor, so the BO absorbed that mismatch into the
    # material and contact parameters it was calibrating.
    #
    # This is deliberately NOT a change to `collect_training_data()`, which must
    # keep collecting both kinds of trial - dataset A needs vein-present and
    # vein-absent trajectories, since the GNN's whole task is telling them apart.
    # That path's per-substep choice (DIFFTACTILE_VEIN_PAIR) is untouched.
    #
    # Empty, not [0]: pair 0 (sensor<->phantom) is disabled project-wide, so
    # "vessel-free" here means no contact pairs are resolved at all.
    contact_model.collision_ixs = active_collision_pairs(False)

    contact_model.visualisation_set_up_gui()
    # The four DA interactions, which are NOT the four training trajectory types
    # set up in set_up_trajectories_and_phantom_states(). This call replaces
    # them with press / twist_z / twist_x / slide, in the order the figure's
    # panels (a)-(d) expect.
    contact_model.generate_trajectories()
    contact_model.trajectory_ix[None] = 0
    contact_model.set_up_initial_positions_state_and_trajectory()
    contact_model.reset_pid_controller()
    contact_model.reset_exp_sim_traj()
    contact_model.get_keypoint_indices_and_validate()

    start = time.perf_counter()

    # TWO MODES, selected by DIFFTACTILE_DA_MODE:
    #
    #   joint (default)  ONE BO over sensor E + vein contact stiffness, running
    #                    BOTH trajectories per iteration and scoring 3A - B.
    #                    Fidelity and vessel sensitivity are traded off inside a
    #                    single search.
    #   staged           the older sequential design: vessel-absent BO first,
    #                    then vessel-present BO at the sensor it chose. Kept
    #                    because it is what the published slide_only_bo run used.
    #
    # Both finish with the same validation: all four trajectories once, at the
    # chosen configuration, vessel-free (which is what the reference
    # photographs show).
    mode = os.environ.get("DIFFTACTILE_DA_MODE", "joint")
    final_params, label = None, None

    if mode == "joint":
        _, joint_best = contact_model.domain_adaptation_joint(run_dir=run_dir)
        if joint_best is not None:
            final_params, label = dict(joint_best["params"]), "final_joint"
    elif mode == "staged":
        # STAGE 1 IS LOADED FROM DISK when a completed run is available, rather
        # than re-simulated. Its search is seeded and deterministic, so
        # re-running it reproduces the same observations at a cost of ~30 minutes
        # of simulation; replaying the saved ones gives the same surrogate.
        # DIFFTACTILE_STAGE1_OBSERVATIONS points elsewhere, or "" forces a fresh
        # run.
        stage1_default = repo_path(
            "difftactile/output/domain_adaptation_published/slide_only_bo/"
            "bo_observations.json"
        )
        stage1_path = os.environ.get(
            "DIFFTACTILE_STAGE1_OBSERVATIONS", stage1_default
        )
        if stage1_path and os.path.exists(stage1_path):
            loaded = contact_model.bo.load_observations(
                stage1_path, names={"slide"}
            )
            print(f"\nStage 1 loaded from {stage1_path}: {loaded} "
                  f"(not re-simulated - the search is seeded and deterministic)")
            _, best_mae = contact_model.bo.best_observed("slide")
            print(f"  best vessel-absent MAE: {best_mae:.2f} px")
        else:
            contact_model.domain_adaptation(run_dir=run_dir)

        sensor_params, _ = contact_model.bo.best_observed("slide")
        if os.environ.get("DIFFTACTILE_SKIP_VEIN_BO", "0") != "1":
            _, vein_best = contact_model.domain_adaptation_vein(run_dir=run_dir)
            if vein_best is not None:
                # MERGE, do not replace. The vein stage fits only the contact
                # stiffness, so its parameter dict alone does not describe the
                # configuration that ran - the sensor material came from stage 1.
                # Validating the vein dict by itself recorded a provenance that
                # could not be reproduced from the JSON.
                final_params = dict(sensor_params)
                final_params.update(vein_best["params"])
                label = "final_vein"
        if final_params is None:
            final_params, label = dict(sensor_params), "final_slide"
    else:
        raise SystemExit(
            f"DIFFTACTILE_DA_MODE={mode!r} is not recognised; "
            f"use 'joint' or 'staged'."
        )

    if final_params is None:
        print("\nNo usable configuration was found - skipping validation.")
    else:
        contact_model.validate_final_params(final_params, run_dir, label=label)

    print(f"\nDomain adaptation took {time.perf_counter() - start:.2f} s")
    print(f"All artifacts: {run_dir}")


def record_da_trajectories_main():
    """Entrypoint: run the four DA interactions once each and record them.

    A VISUALISATION entrypoint, not a calibration one. It runs press / twist_z /
    twist_x / slide a single time at the parameters already in
    `system-params.json` and records the default camera's view to .mp4 - use it
    to see what the simulator is actually doing. `domain_adaptation_main()` is
    the one that searches for parameters.

    Every run writes to its OWN timestamped directory under
    difftactile/output/da_recordings/<YYYYmmdd-HHMMSS>/, so runs accumulate
    rather than overwrite - the same convention the BO runs use.

    NEEDS A DISPLAY. Frames are read from the GGUI colour buffer, and Taichi's
    offscreen path segfaults in this image, so there is no headless route.

    Environment:
        DIFFTACTILE_VIDEO_FPS    output frame rate (default 30)
        DIFFTACTILE_VIDEO_SCOPE  both (default) / per-trajectory / combined
        DIFFTACTILE_DA_MAX_TIMESTEPS  per-trajectory cap (default 400)
        DIFFTACTILE_VEIN         1 to embed the subsurface vein (default off,
                                 matching domain adaptation - the reference
                                 photographs are of a plain phantom)
        DIFFTACTILE_RECORD_TRAJECTORIES
                                 comma-separated subset of press / twist_z /
                                 twist_x / slide to record (default all four)
    """
    run_dir = repo_path(
        f"difftactile/output/da_recordings/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if is_headless():
        raise SystemExit(
            "This entrypoint records the GGUI window, which needs a display.\n"
            "DISPLAY is unset (or DIFFTACTILE_HEADLESS=1), and Taichi GGUI "
            "segfaults offscreen in this image, so there is no headless route."
        )

    # Reproducible: generate_trajectories() randomises the interactions.
    seed = seed_everything(deterministic_torch=False)
    print(f"Seed: {seed} (override with DIFFTACTILE_SEED)")

    if RUN_ON_LAB_MACHINE:
        ti.init(
            debug=False,
            offline_cache=False,
            log_level=ti.ERROR,
            arch=ti.cuda,
            device_memory_GB=float(os.environ.get("TI_DEVICE_MEMORY_GB", 9)),
        )
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)

    contact_model = Contact()

    # No vein by default, matching domain_adaptation_main(): these are the same
    # four interactions, and the real sensor photographs they correspond to were
    # taken on a plain phantom. DIFFTACTILE_VEIN=1 puts it back, which is useful
    # for SEEING what the inclusion does to the deformation - it is drawn yellow.
    if os.environ.get("DIFFTACTILE_VEIN", "0") == "1":
        contact_model.collision_ixs = active_collision_pairs(True)
        print("Vein ENABLED (drawn yellow)")
    else:
        contact_model.collision_ixs = active_collision_pairs(False)

    contact_model.visualisation_set_up_gui()
    # Replaces the four TRAINING trajectory types with the four DA interactions,
    # in the order the manuscript's panels (a)-(d) expect.
    contact_model.generate_trajectories()
    contact_model.trajectory_ix[None] = 0
    contact_model.set_up_initial_positions_state_and_trajectory()
    contact_model.reset_pid_controller()
    contact_model.reset_exp_sim_traj()
    contact_model.get_keypoint_indices_and_validate()

    start = time.perf_counter()
    contact_model.record_da_trajectories(run_dir=run_dir)
    print(f"\nRecording took {time.perf_counter() - start:.2f} s")
    print(f"All artifacts: {run_dir}")


def main():
    # Seed BEFORE anything stochastic runs, so a collected dataset is reproducible.
    #
    # Training-data collection is heavily randomised by design - that randomness
    # IS the dataset's variety - but it was drawn from an unseeded `NP_RNG`
    # (main/constants.py), so two runs of the same command produced different
    # trajectories and neither could be regenerated. What this reaches:
    #
    #   * generate_trajectories() / generate_random_state_dicts() - the sensor
    #     poses, press depths, slide directions and rotations of every trial.
    #   * randomise_contact_params() - the per-trial sensor<->vein contact
    #     stiffness and damping (pair index 2).
    #   * the NP_RNG.permutation() over collision_ixs in the contact loop.
    #
    # Collection takes ~2 h 45 m for the full 800 trials, so being unable to
    # reproduce a dataset was expensive: any question about a published trial
    # ("which parameters produced this?") could only be answered by keeping the
    # .npz files forever. With a fixed seed the run itself is the record.
    #
    # NOTE this changes which dataset a default run produces - it is a different
    # draw, not the previously-shipped one. The PUBLISHED dataset predates this
    # and cannot be regenerated by any seed; restore it from the Zenodo bundle
    # rather than expecting `DIFFTACTILE_SEED=42` to rebuild it.
    #
    # `deterministic_torch=False`: no torch in this path (the simulator is
    # Taichi), so the deterministic-kernel switches would cost startup work and
    # set CUBLAS_WORKSPACE_CONFIG for nothing.
    seed = seed_everything(deterministic_torch=False)
    print(f"Seed: {seed} (override with DIFFTACTILE_SEED)")

    if RUN_ON_LAB_MACHINE:
        ti.init(
            debug=False,
            offline_cache=False,
            log_level=ti.ERROR,
            arch=ti.cuda,
            # Honour TI_DEVICE_MEMORY_GB so a smaller card can be accommodated;
            # an explicit kwarg would otherwise override the env var silently.
            device_memory_GB=float(os.environ.get("TI_DEVICE_MEMORY_GB", 9)),
        )
    else:
        ti.init(debug=False, offline_cache=False, log_level=ti.ERROR, arch=ti.cpu)
    contact_model = Contact()
    contact_model.visualisation_set_up_gui()
    contact_model.save_tactile_sensor_mesh_node_mapping_to_pickle()

    contact_model.trajectory_ix[None] = 0
    contact_model.set_up_initial_positions_state_and_trajectory()
    contact_model.reset_pid_controller()
    contact_model.reset_exp_sim_traj()
    contact_model.get_keypoint_indices_and_validate()
    # contact_model.set_up_torch_params()
    
    start_time = time.perf_counter()
    contact_model.collect_training_data()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training data collection took {elapsed_time:.2f} seconds")
    
    contact_model.bo.write_to_file()
    if False:
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            contact_model.collect_training_data()
            profiler.disable()
        finally:
            profiler.dump_stats("profile.out")

