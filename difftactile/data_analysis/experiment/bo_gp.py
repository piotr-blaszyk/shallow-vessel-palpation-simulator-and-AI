import json

import IPython
from bayes_opt import BayesianOptimization, acquisition

from difftactile.main.constants import *


class BoGp:
    """Bayesian optimisation over the simulator's material and contact parameters.

    The GP never sees a raw parameter value. Every parameter is mapped to the
    unit cube first, and SCALE-LIKE parameters are mapped through their
    LOGARITHM on the way (see `LOG_SCALED` below) - "log-transform then
    min-max", never log-then-min-max-then-standardise.
    """

    # Which parameters are scale-like, i.e. a fixed MULTIPLICATIVE change matters
    # equally anywhere in the range, so they are searched in log space.
    #
    # The test is how you would naturally state the range. "1e4 to 4.8e5" or
    # "1x to 100x" is multiplicative; "0.3 to 0.5" is not. Young's modulus and
    # the three contact stiffness/damping coefficients are physical scale
    # parameters spanning one to two decades - under a linear map their bottom
    # decade is squeezed into a few percent of [0, 1], which is finer than a GP
    # with an O(1) lengthscale (or the acquisition optimiser's absolute
    # tolerances) can resolve. Poisson's ratio and the friction coefficient are
    # bounded ratios on a natural additive scale, so they stay linear.
    # THE PHANTOM IS NOT SEARCHED. Its Young's modulus and Poisson's ratio are
    # deliberately absent here: the phantom's particles are pinned in place for
    # simulated dataset collection, so it does not deform and its material
    # parameters have no effect on anything the objective measures. Searching
    # them would waste iterations on two dimensions that cannot change the score.
    #
    # They REMAIN in system-params.json (phantom.silicone / phantom.hard_plastic)
    # and are still read by phantom.py - do not delete them from the codebase.
    # They are kept so the phantom can be made deformable again by whoever
    # extends this work; at that point add them back to the dicts below (and to
    # LOG_SCALED, since a modulus is scale-like) and re-enable the corresponding
    # lines in main.py::set_contact_params_from_bo().
    LOG_SCALED = {
        'vitactip_youngs_modulus',
        'normal_stiffness',
        'tangential_stiffness',
        'normal_damping',
    }

    def __init__(self):
        # Search bounds in RAW physical units. Nominal values from
        # system-params.json for reference: normal_stiffness 44.8,
        # tangential_stiffness 3.9, normal_damping 34.0,
        # coulomb_friction_coeff 0.66.
        #
        # The log-scaled entries need a STRICTLY POSITIVE lower bound - log(0) is
        # -inf, so a bound of 0 has no image in log space at all. The three
        # contact coefficients previously started at exactly 0, which is a
        # degenerate configuration anyway (zero normal stiffness means the sensor
        # does not touch the phantom), so each is floored at 5e-2.
        #
        # The UPPER bounds were also raised from 5e1 to 5e2. The adopted
        # configuration in bo-gp.json sits at normal_stiffness 44.8 and
        # normal_damping 34.0 - i.e. hard against the old ceiling of 50, with
        # under 12% of the range above it. A boundary hit like that is exactly
        # the case for extending by a DECADE rather than by a fixed increment, so
        # the search can actually explore the stiffer side of the working point
        # instead of being clipped at it. The result is four decades from 5e-2 to
        # 5e2, with the adopted values near the middle rather than at the edge.
        # THE SENSOR RANGES ARE WIDE AND INCLUDE VERY SOFT CONFIGURATIONS.
        # E in [1e4, 2e6] Pa, nu in [1e-2, 0.45].
        #
        # This is an EXPLICIT CHOICE to let the search decide, reverting an
        # earlier box that pinned the sensor stiff by construction
        # (E in [1e6, 2e6]). The upper ends are unchanged and remain the validated
        # ones - E = 2e6 with nu = 0.45 is the stiffest corner, at C = 0.357, and
        # it is verified both by CFL and by running all four trajectories.
        #
        # WHAT THE SOFT END COSTS, so a future reader does not rediscover it the
        # hard way. Both effects were measured on this simulator, not assumed:
        #
        #   * TIP LAG. The base layer is a hard kinematic constraint
        #     (update_external_forces() OVERWRITES its velocity with
        #     vertex_control_velocities) while the interior only catches up as
        #     shear waves travel down the mesh, at c_s = sqrt(mu/rho). At the
        #     nominal E = 1.39e5 that takes ~460 timesteps against a `slide` of
        #     ~398, so the tip visibly trails the base for the whole trajectory.
        #     At E = 2e6 it is ~120 timesteps and the lag is minimal.
        #   * DEGENERATE ELEMENTS NEAR THE BASE. On `twist_x`, elements straddling
        #     the fixed-layer boundary absorb the mismatch between the imposed
        #     rigid rotation and a soft interior that cannot follow. Measured
        #     minimum Jacobian J = det(F) at the end of the trajectory:
        #
        #         E=1e4   nu=0.497   min J = 0.449   (55% volume compression;
        #                                             37/40 worst elements touch
        #                                             the fixed layer, vs a 32.6%
        #                                             baseline)
        #         E=2e6   nu=0.45    min J = 0.963   (4% compression, 24/40)
        #
        #     No element INVERTS in either case (J stays positive), so this is
        #     bad conditioning rather than a solver failure - but the soft case
        #     looks visibly degenerate in the GGUI view.
        #
        # Neither is a crash: softer is always CFL-stable, so nothing here can
        # blow the solve up. The risk is silent - the search can buy a good MAE
        # with a sensor that deforms by the wrong mechanism (draping and dragging
        # rather than pressing), which is not what the real ViTacTip does, since
        # the physical sensor/phantom interface is well lubricated. Note also
        # that the range is log-scaled, so ~87% of it lies below E = 1e6 and ~50%
        # below the nominal 139300: a log-uniform initial design will sample the
        # soft regime heavily. (The floor has been raised twice, 1e2 -> 1e3 -> 1e4,
        # trimming the softest decades - 1e2 was 1400x softer than nominal and of
        # no physical interest. The range is still 2.3 decades wide, and its
        # geometric midpoint is now 1.41e5, essentially the nominal 139300, so a
        # typical random draw is no longer systematically softer than nominal.)
        #
        # If the sensor ever needs pinning stiff again, restore the floor to
        # E = 1e6 - that box is validated and documented in the git history.
        #
        # WHY POISSON'S RATIO IS CAPPED AT 0.45, AWAY FROM INCOMPRESSIBLE.
        # The visible symptom was the sensor TIP lagging the BASE during `slide`,
        # with no contact involved: the base layer is a hard kinematic constraint
        # (update_external_forces() OVERWRITES its velocity with
        # vertex_control_velocities), while the tip is free and only catches up as
        # elastic waves carry the motion down the mesh. That lag is a SHEAR
        # response, so it is governed by the shear wave speed
        # c_s = sqrt(mu/rho), mu = E/(2(1+nu)) - NOT by the pressure wave.
        #
        # The two speeds pull in opposite directions as nu -> 0.5:
        #   * c_p = sqrt((lam + 2 mu)/rho) DIVERGES, and c_p is what sets the CFL
        #     limit, so near-incompressible material forces a smaller dt.
        #   * c_s barely moves (mu changes by ~5% from nu=0.49 to nu=0.35), so
        #     incompressibility buys almost NO extra rigidity against this lag.
        #
        # Near-incompressibility was therefore paying the entire CFL cost while
        # delivering nothing for the actual problem. Capping nu at 0.45 collapses
        # c_p and frees the CFL headroom that makes E = 2e6 affordable, and it is
        # E that raises c_s. Net effect on the lag, over a ~30 mm sensor at
        # dt = 1e-5:
        #
        #     E=1.393e5 nu=0.497 (nominal)   c_s =  6.5 m/s   ~460 timesteps
        #     E=2.5e5   nu=0.49              c_s =  8.7 m/s   ~344 timesteps
        #     E=2.0e6   nu=0.45              c_s = 25.0 m/s   ~120 timesteps
        #
        # i.e. at the stiff end the tip catches up ~4x faster than at the nominal
        # parameters. THE TRADE IS DELIBERATE: the sensor is less incompressible
        # than real silicone, in exchange for moving much more like a rigid body.
        #
        # 0.45 IS THE RIGHT PLACE TO STOP, checked rather than assumed. Going
        # further to nu = 0.40 was measured and rejected: c_s rises only
        # 25.0 -> 25.5 m/s (tip lag 120 -> 118 timesteps) and twist_x's min J
        # barely moves (0.963 -> 0.955), while the bulk-to-shear ratio K/mu HALVES
        # from 9.7 to 4.7 - i.e. it buys ~2% rigidity for twice the volumetric
        # squashiness. mu = E/(2(1 + nu)) is nearly flat in nu, so past this point
        # only E moves the needle. Lower nu is still permitted by the range below
        # (the floor is 1e-2), but it is not where the useful configurations are.
        #
        # CONFIRMED VISUALLY, which is the only test that settles it: the lag is a
        # rendered-motion artefact, and no scalar in the objective measures it. The
        # stiffest corner was watched through a full `slide` in the GGUI window and
        # the tip lag is minimal - the sensor tracks its kinematically-driven base
        # instead of trailing behind it for the whole trajectory. Do not "restore"
        # nu towards 0.5 on the grounds that silicone is nearly incompressible:
        # that is what caused the lag, and it would also cut the affordable E by
        # ~4x at this dt.
        #
        # THE UPPER BOUNDS ARE SET BY THE TIMESTEP, NOT BY TASTE. This is an
        # explicit MPM/FEM solve at a FIXED dt (contact.dt_override = 1e-5 s), and
        # stiffness raises the elastic wave speed c = sqrt((lam + 2 mu)/rho). The
        # CFL condition needs dt <= dx/c, so too stiff a sensor blows the solve up
        # - that is the crash mode to avoid, and the alternative (shrinking dt)
        # would slow every simulation down, which is exactly what must not happen.
        #
        # E and nu are NOT independent here: nu enters c_p through
        # (1 - nu)/((1 + nu)(1 - 2 nu)), which diverges as nu -> 0.5. So the
        # incompressibility bound and the stiffness bound have to be chosen
        # TOGETHER, and the binding case is the joint upper corner. This is the
        # same coupling exploited above: backing nu off to 0.45 is precisely what
        # makes E = 2.0e6 affordable. At nu = 0.49 that same E would sit at
        # C = 0.537, and at nu = 0.497 the ceiling falls below 5.2e5.
        #
        # Measured Courant number C = dt * c_p / dx over this box
        # (dx = 2.323e-3 m, rho = 1100 kg/m^3):
        #
        #     E=1.0e3 nu=0.01  ->  C = 0.004     (softest corner, trivially stable)
        #     E=1.0e3 nu=0.45  ->  C = 0.008
        #     E=1.0e6 nu=0.45  ->  C = 0.253
        #     E=2.0e6 nu=0.01  ->  C = 0.184
        #     E=2.0e6 nu=0.45  ->  C = 0.357     (stiffest corner, the binding one)
        #
        # The nominal configuration already runs at C = 0.363, so EVERY point in
        # this space - including the stiffest corner - is below what the simulator
        # does today. dt therefore does not need to change, which is the whole
        # point. Only the UPPER corner can threaten CFL; the wide soft end costs
        # nothing in stability (see the tip-lag / element-quality note above for
        # what it does cost).
        #
        # If these bounds are ever widened, recompute C at the new joint upper
        # corner FIRST - a crash at some parameter set means the sensor got too
        # stiff for the fixed dt, not that the optimiser misbehaved.
        #
        # CFL IS NECESSARY, NOT SUFFICIENT. It bounds the linear wave speed; the
        # contact solve and the MPM transfer can still blow up inside a
        # CFL-satisfying box, so the stiffest corner is checked EMPIRICALLY rather
        # than trusted. An earlier box (E <= 5.0e5, nu <= 0.49, C = 0.380) was
        # verified this way - all four trajectories completed with every sensor
        # vertex finite - and the current corner sits at a LOWER C than that one.
        #
        # RESPONSE TO A CRASH, if one ever appears here: halve the offending
        # side of each stiffness-controlling bound, one side per parameter, and
        # re-test the corner. Which side depends on which direction stiffens the
        # sensor:
        #
        #   vitactip_youngs_modulus   UPPER bound  /2   (higher E = stiffer)
        #   vitactip_poissons_ratio   UPPER bound  ->   halve its DISTANCE TO 0.5
        #                                              (nu -> 0.5 is the singular,
        #                                              incompressible limit, so
        #                                              "halve the bound" is
        #                                              meaningless; 0.49 -> 0.495
        #                                              would be STIFFER, the wrong
        #                                              way. Halving the gap gives
        #                                              0.49 -> 0.48.)
        #   normal_stiffness          UPPER bound  /2
        #   tangential_stiffness      UPPER bound  /2
        #   normal_damping            UPPER bound  /2
        #
        # The LOWER bounds are never the crash cause - a softer sensor is always
        # stable - so they stay put.
        # TWO DISJOINT SEARCH SPACES, one per stage. They share no parameter, so
        # the two models cannot interfere and the vessel-present search needs NO
        # fidelity constraint: varying the sensor<->vein contact stiffness cannot
        # change how the sensor behaves on a phantom that has no vein in it.
        #
        # The vessel-ABSENT model fits the sensor MATERIAL only. Its old contact
        # dimensions (pair 0) are gone because that pair is disabled - the
        # phantom is pinned, so those four coefficients could not move the score
        # and the search was fitting noise across them. Dropping them takes the
        # problem from 6 dimensions to 2, which is far better posed for the ~10
        # observations a run affords.
        #
        # E's bounds MUST match `pbounds_joint` below: the two spaces share the
        # key, and `_bounds_for_key` refuses to guess between conflicting
        # ranges - a parameter that normalised differently depending on which
        # model proposed it would make the two models' observations silently
        # incomparable.
        self.pbounds_no_vein = {
            'vitactip_youngs_modulus': (1e5, 1e6),
            'vitactip_poissons_ratio': (1e-2, 0.45),
        }
        # The vessel-PRESENT model fits ONE parameter: the sensor<->vein normal
        # stiffness. Everything else about that pair is fixed (damping at the
        # repo's 100, tangential stiffness and friction at 0 - see
        # Contact.VEIN_* in main.py), and the sensor material is inherited from
        # the vessel-absent model's best result rather than re-fitted.
        #
        # The upper bound is 1e5, twice the repo's own pair-2 value of 5e4 (the
        # top of the range `randomise_contact_params()` samples). Raised beyond
        # the known-tolerated regime deliberately: a 10-iteration search found
        # the vessel barely deformed the stiff sensor at all, so the useful
        # region - if there is one - lies at higher contact stiffness than the
        # simulator has previously been run at. Log-scaled, spanning six decades.
        #
        # This is the parameter most likely to destabilise the solve: the vein's
        # penalty force is (k_n + c_n*v_n)*d on a FIXED dt, and the damping-side
        # stability limit scales as 1/c_n. A crash here is expected to be the
        # contact, not the material - `_report_vein_crash()` says as much and
        # tells the operator to halve the sensor's Young's modulus.
        # Matches `pbounds_joint`'s range for the same reason as above.
        self.pbounds_vein = {
            'normal_stiffness': (1e4, 1e5),
        }
        # The JOINT space: one model fitting the sensor material and the vein
        # contact TOGETHER, scoring both trajectories in every iteration. This is
        # what makes the fidelity/sensitivity trade-off a single optimisation
        # rather than two sequential ones - the objective can trade a little
        # vessel-absent accuracy for a lot of vessel contrast, which the
        # two-stage design structurally cannot do.
        #
        # Poisson's ratio is NOT searched here: it is held at whatever the sensor
        # already has. Only E and the vein's normal stiffness are free, which
        # keeps a 10-iteration budget over 2 dimensions rather than 3.
        # NARROWED to the region where BOTH halves of the objective are
        # achievable, which the wider box demonstrably was not:
        #
        #   E in [1e5, 1e6]   the soft end (1e4-1e5) diverged repeatedly -
        #                     inverted elements at 18k-108k - and the very stiff
        #                     end (>1e6) diverged too, with NaN at 1.5e6-1.76e6.
        #                     What survived sat in between.
        #   k_n in [1e4, 1e5] the vessel only actually stops the sensor at high
        #                     contact stiffness: at the shipped 5e4 it lifts the
        #                     sensor 8.4 mm (vpn 0.99, 13.7 px marker signal),
        #                     while every k_n ~0.2 the search picked gave vpn
        #                     ~0.001 and no signal at all. Below ~1e4 there is
        #                     nothing to find.
        #
        # Both remain log-scaled, spanning one decade each.
        self.pbounds_joint = {
            'vitactip_youngs_modulus': (1e5, 1e6),
            'normal_stiffness': (1e4, 1e5),
        }
        # Default space, used by anything that does not name a stage.
        self.pbounds = dict(self.pbounds_no_vein)
        self._validate_bounds()
        # The GP's own search space is the unit cube for every parameter, log
        # scaled or not. Defining it here rather than transforming after the fact
        # means the acquisition optimiser, its lengthscales and the random
        # initial design all operate in log coordinates automatically - sampling
        # raw values and taking the log afterwards would silently reintroduce the
        # linear spacing this is meant to remove.
        self.pbounds_normalised = {k: (0.0, 1.0) for k in self.pbounds}
        # Which space each model searches. A name not listed here falls back to
        # the vessel-absent space.
        self.model_spaces = {
            "slide_vein": self.pbounds_vein,
            "joint": self.pbounds_joint,
        }
        # ONE INDEPENDENT SURROGATE PER TRAJECTORY, not one over an aggregate.
        #
        # Each trajectory is its own objective g_i(x), on its own scale (a slide
        # MAE runs ~10x a press MAE). Fitting one GP per objective and INDUCING
        # the composite - rather than fitting a single GP to the mean of the four
        # - keeps that structure instead of averaging it away, and it means an
        # iteration can evaluate ONE trajectory rather than all four: 4x more
        # proposals for the same wall-clock, which is the whole point here since
        # each trajectory is a full forward simulation.
        #
        # It also removes the need to know each objective's true range. Every GP
        # learns its own output scale, so no rescaling of the raw MAE is required
        # (and none is done - see my_register).
        #
        # `optimisers` is keyed by trajectory name and populated lazily by
        # _optimiser_for(), so this class does not need to know the trajectory
        # names up front - main.py owns those.
        self.optimisers = {}
        self.params_path = SYSTEM_PARAMS.files.bo_gp_json
        self.target_path = SYSTEM_PARAMS.files.bo_gp_target_json
        self.all_params_path = SYSTEM_PARAMS.files.bo_all_params
        self.all_targets_path = SYSTEM_PARAMS.files.bo_all_targets
        self.all_params = []
        self.all_targets = []
        # Per-objective observation log, for the recommendation step and so a run
        # is reconstructible without refitting: {name: [(params_dict, mae), ...]}
        self.observations = {}

    def _optimiser_for(self, name):
        """The `BayesianOptimization` for one trajectory, created on first use.

        Each gets its OWN `random_state`, derived from the name, so the four
        models do not draw identical "random" sequences - with a shared seed
        every model would propose the same points and the four surrogates would
        be redundant.
        """
        if name not in self.optimisers:
            acq = acquisition.ExpectedImprovement(xi=0.01)
            space = self.space_for(name)
            self.optimisers[name] = BayesianOptimization(
                f=None,
                acquisition_function=acq,
                pbounds={k: (0.0, 1.0) for k in space},
                # Duplicates are legitimate here, in two ways. Projecting an
                # older run onto a smaller space collapses points that differed
                # only in dimensions no longer searched - they were the same
                # experiment, repeated. And on a 1-D or 2-D space the
                # acquisition optimiser will genuinely revisit a converged
                # optimum. Both should be recorded, not raised on: the repeats
                # also give the GP a read on observation noise.
                allow_duplicate_points=True,
                verbose=0,
                # Stable across runs and distinct per trajectory. Deliberately
                # not hash(): Python salts str hashes per process, so it would
                # not be reproducible.
                random_state=1 + (sum(ord(c) for c in name) % 1000),
            )
            self.observations[name] = []
        return self.optimisers[name]
    
    def set_run_dir(self, run_dir):
        """Redirect the per-run outputs into `run_dir`.

        `bo_gp_json` deliberately stays where it is: it is the handoff file the
        simulator reads back for the CURRENT proposal, not a record of the run.
        The two history files are records, so a timestamped run keeps its own
        rather than overwriting the last one's.
        """
        import os
        os.makedirs(run_dir, exist_ok=True)
        self.all_params_path = os.path.join(run_dir, "bo_all_params.json")
        self.all_targets_path = os.path.join(run_dir, "bo_all_targets.json")

    def space_for(self, name):
        """The bounds dict `name`'s model searches."""
        return self.model_spaces.get(name, self.pbounds_no_vein)

    def _bounds_for_key(self, key):
        """(lo, hi) for `key`, from whichever space defines it.

        Lets normalise/unnormalise work on a parameter dict without being told
        which stage produced it. The two STAGE spaces are disjoint, and the joint
        space reuses their keys with IDENTICAL bounds - which is what keeps this
        lookup well-defined. That is asserted rather than assumed: if the joint
        space ever disagreed, a parameter would normalise differently depending
        on which model produced it, and observations from the two would silently
        stop being comparable.
        """
        found = None
        for space in (self.pbounds_no_vein, self.pbounds_vein,
                      self.pbounds_joint):
            if key not in space:
                continue
            if found is not None and space[key] != found:
                raise ValueError(
                    f"Parameter {key} has conflicting bounds across search "
                    f"spaces: {found} vs {space[key]}"
                )
            found = space[key]
        if found is None:
            raise KeyError(
                f"Parameter {key} is in no search space "
                f"({sorted(self.pbounds_no_vein)} / "
                f"{sorted(self.pbounds_vein)} / {sorted(self.pbounds_joint)})"
            )
        return found

    def _validate_bounds(self):
        """Fail loudly on bounds a log map cannot represent.

        A log-scaled parameter needs lo > 0: log(0) is -inf and a negative value
        has no log at all, either of which would put a NaN into the GP's training
        data, where it would poison every subsequent proposal rather than raising.
        """
        for space in (self.pbounds_no_vein, self.pbounds_vein,
                      self.pbounds_joint):
            for key, (lo, hi) in space.items():
                if hi <= lo:
                    raise ValueError(
                        f"{key}: upper bound {hi} must exceed lower {lo}"
                    )
                if key in self.LOG_SCALED and lo <= 0:
                    raise ValueError(
                        f"{key} is log-scaled, so its lower bound must be "
                        f"strictly positive (got {lo}). Use a small positive "
                        f"floor, or drop the parameter from LOG_SCALED."
                    )

    def normalise_dict(self, input_dict):
        """Raw physical units -> the unit cube the GP searches.

        Log-scaled parameters are mapped through their logarithm first:

            u = (log x - log lo) / (log hi - log lo)

        so each decade of the range occupies an equal share of [0, 1]. The base
        cancels in the ratio, so natural log is used. Linear parameters get the
        plain min-max map. Nothing is standardised on top of either.
        """
        normalized = {}
        for key, value in input_dict.items():
            min_val, max_val = self._bounds_for_key(key)
            if key in self.LOG_SCALED:
                # Clamp before the log: a value at or below zero cannot be
                # represented, and floating-point drift through the round trip can
                # otherwise leave a proposal a hair under the lower bound.
                value = max(float(value), min_val)
                normalized[key] = (
                    (np.log(value) - np.log(min_val))
                    / (np.log(max_val) - np.log(min_val))
                )
            else:
                normalized[key] = (value - min_val) / (max_val - min_val)
        return normalized

    def unnormalise_dict(self, normalized_dict):
        """The unit cube -> raw physical units; inverse of `normalise_dict`.

        For log-scaled parameters this is the geometric interpolation
        x = lo * (hi / lo) ** u, so u = 0.5 lands on the geometric mean of the
        bounds rather than the arithmetic one.
        """
        unnormalized = {}
        for key, value in normalized_dict.items():
            min_val, max_val = self._bounds_for_key(key)
            if key in self.LOG_SCALED:
                log_min, log_max = np.log(min_val), np.log(max_val)
                unnormalized[key] = float(
                    np.exp(value * (log_max - log_min) + log_min)
                )
            else:
                unnormalized[key] = value * (max_val - min_val) + min_val
        return unnormalized

    @staticmethod
    def black_box_function(*args, **kwargs):
        return 0

    def my_suggest_optimise(self, name):
        """Acquisition-driven proposal from the surrogate for `name`."""
        params = self._optimiser_for(name).suggest()
        self.my_suggest_optimise_helper(params)

    def my_suggest_random(self, name=None):
        """Seed the GP with a uniform draw from the unit cube.

        Drawn in NORMALISED coordinates, which is what makes the initial design
        log-uniform for the log-scaled parameters - each decade gets an equal
        share of the draws. Sampling raw values and taking the log afterwards
        would concentrate the design in the top decade, which is the trap this
        whole transform exists to avoid.

        `name` is accepted for symmetry with `my_suggest_optimise` and to make
        the surrogate exist from iteration 0; the draw itself is model-free.
        """
        if name is not None:
            self._optimiser_for(name)
        space = self.space_for(name)
        params = {k: NP_RNG.uniform(0.0, 1.0) for k in space}
        self.my_suggest_optimise_helper(params)

    def my_suggest_optimise_helper(self, params):
        """Turn one proposal in the unit cube into raw parameters for the sim.

        `tangential_stiffness` is not free: it is tied to `normal_stiffness` as a
        random fraction of it, which is a relationship between PHYSICAL values.
        It is therefore imposed after `unnormalise_dict`, on the raw quantities.
        Applying it to the normalised coordinates (as this used to) silently
        became exponentiation once `normal_stiffness` went log-scaled -
        u_t = f * u_n maps to x_t = lo * (hi/lo) ** (f * u_n), i.e. a fractional
        POWER of the ratio rather than a fraction of the stiffness.

        The result is clipped back into `pbounds`: the fraction can push the
        product below the 5e-2 floor, and a value outside the bounds would be
        rejected by `optimiser.register` in `my_register`.
        """
        self._commit_params(self._finalise_params(params))

    def _finalise_params(self, params):
        """Unit-cube proposal -> raw parameters.

        Split out of `my_suggest_optimise_helper` so a proposal can be built and
        INSPECTED before being committed as the current parameter set.

        The old `tangential_stiffness = U(0, 0.3) * normal_stiffness` coupling is
        gone with the parameter itself: tangential stiffness is now a fixed 0 on
        the only live contact pair (see Contact.VEIN_TANGENTIAL_STIFFNESS), so
        there is nothing to couple. That also removes a hidden source of
        stochasticity - the coupling redrew a random fraction on every proposal,
        so the same unit-cube point did not map to the same raw parameters twice.
        """
        return self.unnormalise_dict(params)

    def _commit_params(self, params, verbose=True):
        """Adopt `params` as the current proposal and write the handoff file."""
        if verbose:
            print(params)
        self.params = params
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=4)
    
    def my_register(self, target, name, maximise=False):
        """Record one evaluated configuration and its MAE, against `name`'s model.

        The MAE is handed to the GP RAW, only negated - `bayes_opt` maximises,
        and a lower alignment error is better. There is deliberately no rescaling
        of the target: it is already in meaningful units (pixels), and mapping it
        into some other range only obscures what the optimiser is working with.

        Note the PARAMETERS are still normalised to the unit cube by
        `normalise_dict`. That is a different thing and is load-bearing: Young's
        modulus spans ~1e5 while the friction coefficient spans ~1, and a GP
        kernel comparing them on their raw scales would be dominated by the
        largest-magnitude parameter. For the scale-like parameters that map runs
        through a LOG first, so the GP measures distance multiplicatively.

        The target is deliberately NOT log-transformed. It spans well under a
        decade in practice (~11-14 px for usable configurations) and the
        divergence penalty is a deliberate constant in the same pixel units, so
        there is nothing here for a log to fix.
        """
        # `bayes_opt` MAXIMISES. Most objectives here are errors to be minimised,
        # so the target is negated; the vessel-present model is the exception -
        # it maximises its MAE (a large disagreement with the vessel-free
        # photograph means the vessel is visibly deforming the sensor), so its
        # target is registered as-is.
        signed = float(target) if maximise else -float(target)
        self._optimiser_for(name).register(
            params=self.normalise_dict(self.params),
            target=signed,
        )
        self.observations[name].append((dict(self.params), float(target)))
        self.all_params.append(self.params)
        self.all_targets.append(target)
    
    def _fitted_gp(self, name):
        """The model's GP, FITTED to everything registered so far.

        `bayes_opt` fits `._gp` lazily inside `suggest()`, so a GP that has only
        ever been `register()`ed to is untrained: `predict` then returns the
        PRIOR - mean 0, std 1 - regardless of the data. That silently made the
        fidelity constraint admit every candidate (predicted 'MAE' of -0.00 px
        against a 20.54 px threshold), which is a constraint that does nothing.
        Fitting explicitly here is what makes the prediction reflect the
        observations.
        """
        opt = self.optimisers[name]
        gp = opt._gp
        gp.fit(opt.space.params, opt.space.target)
        return gp

    def predict_mae(self, name, params):
        """This model's predicted MAE (px) at `params`, with its uncertainty.

        `params` is in RAW units. Returns (mu, sigma) in PIXELS.

        Assumes `name` is a MINIMISING model, i.e. one registered with
        `maximise=False`, whose GP therefore holds negated MAE - the mean is
        flipped back to "lower is better" here. Do not call this on the
        vessel-present model, whose targets are stored unnegated; the sign would
        come out backwards.
        """
        gp = self._fitted_gp(name)
        keys = list(self.space_for(name))
        norm = self.normalise_dict(params)
        x = np.array([[norm[k] for k in keys]])
        mu, sigma = gp.predict(x, return_std=True)
        # np.atleast_1d: sklearn returns a 0-d array for a single query point in
        # some versions, which is not indexable.
        return float(-np.atleast_1d(mu)[0]), float(np.atleast_1d(sigma)[0])

    def load_observations(self, path, names=None, maximise_names=()):
        """Refit surrogates from a previous run's `bo_observations.json`.

        Every observation is replayed through `my_register`, so the resulting
        GPs are exactly those the original run ended with - the search is
        deterministic given its data, and nothing about a model depends on
        WHEN its observations arrived.

        This is what makes a stage re-usable: re-running a completed
        vessel-absent search to obtain a model that is already on disk costs ~30
        minutes of simulation and, being seeded, reproduces the same numbers.

        `names` restricts which models are loaded; `maximise_names` marks those
        whose targets are maximised (see `my_register`), which must match how
        they were originally registered or the GP's sign will be inverted.
        """
        with open(path) as f:
            data = json.load(f)
        loaded = {}
        for name, records in data.items():
            if names is not None and name not in names:
                continue
            space = self.space_for(name)
            for rec in records:
                # PROJECT ONTO THE CURRENT SPACE. Older runs recorded parameters
                # that are no longer searched - the four sensor<->phantom contact
                # coefficients, dropped once that pair was disabled. Keeping only
                # the keys this model still searches lets those runs be reused:
                # the dropped dimensions had no effect on the measured MAE (the
                # pair was inert even then), so the surviving observation is
                # still a valid (x, y) pair for the smaller space.
                #
                # A record MISSING a searched key is a different matter - that
                # is a genuinely incompatible run, and it raises.
                missing = [k for k in space if k not in rec["params"]]
                if missing:
                    raise ValueError(
                        f"{path}: model {name!r} record lacks {missing}, which "
                        f"the current search space requires"
                    )
                projected = {k: rec["params"][k] for k in space}
                # _commit_params, not a suggest: these parameters are given, and
                # my_register reads self.params.
                self._commit_params(projected, verbose=False)
                self.my_register(
                    rec["mae_px"], name, maximise=(name in maximise_names)
                )
            loaded[name] = len(records)
        return loaded

    def best_observed(self, name):
        """The lowest MAE this model has actually measured, and its parameters."""
        obs = self.observations.get(name)
        if not obs:
            raise RuntimeError(f"model {name!r} has no observations")
        params, mae = min(obs, key=lambda t: t[1])
        return dict(params), float(mae)

    def suggest_for(self, name, force_random=False):
        """Propose the next parameter set for `name`, unconstrained.

        NO FIDELITY CONSTRAINT, and none is needed. The vessel-present model
        searches only the sensor<->vein normal stiffness, while the
        vessel-absent model searches only the sensor material - disjoint sets.
        A vein contact coefficient cannot change how the sensor behaves on a
        phantom that contains no vein, so the vessel-present search CANNOT
        degrade vessel-absent fidelity, whatever value it picks.

        The earlier rejection-sampling constraint existed because both stages
        shared one 6-D space and stage 2 could have wandered into parameter sets
        that no longer matched the real sensor. Splitting the spaces removes the
        problem at its source rather than policing it, which is why that
        machinery is gone.

        `force_random` draws uniformly instead of consulting the acquisition
        function, for the seeding phase where the GP has too little data to say
        anything useful.
        """
        if force_random:
            space = self.space_for(name)
            self._optimiser_for(name)
            proposal = {k: NP_RNG.uniform(0.0, 1.0) for k in space}
        else:
            proposal = self._optimiser_for(name).suggest()
        raw = self._finalise_params(proposal)
        self._commit_params(raw)
        return raw

    def recommend(self, num_candidates=20000, beta=1.0, seed=0):
        """Combine the per-trajectory posteriors into ONE recommended parameter set.

        THE ARGMIN OF A SUM IS NOT THE SUM OF THE ARGMINS. Each model records its
        own best-observed configuration, and averaging those four points - or
        picking whichever scored lowest - has no justification: if press is best
        at one corner of the space and slide at another, the composite optimum
        can lie somewhere neither of them visited, or outside them entirely. So
        the four RECORDED OPTIMA are used only as a diagnostic (see
        `argmin_spread` in the returned dict); the recommendation comes from the
        POSTERIORS, which are defined everywhere in the domain rather than only
        where a trajectory happened to be evaluated. That matters here because
        under the schedule no single configuration is measured on all four
        trajectories.

        Method: draw `num_candidates` points from the unit cube, ask every GP for
        its posterior mean and standard deviation there, SUM the four predicted
        MAEs in raw pixels, and return the best point under a lower confidence
        bound.

        NO RESCALING OF THE OBJECTIVES. The composite is the plain sum of the
        four predicted pixel errors:

            f(x) = sum_i mu_i(x)          [px]

        This is deliberate and is what "weight the four equally" means here: one
        pixel of error counts the same whichever trajectory it came from. The
        total is itself a meaningful quantity - the aggregate marker
        misalignment in pixels - and it stays directly comparable to the
        per-trajectory MAEs the run prints and to `PX_TO_MM`.

        The consequence to be aware of: a trajectory whose error is LARGER in
        absolute pixels therefore has more influence on the recommendation.
        Measured on this simulator, `slide` runs ~10x the others (~300 px vs
        ~7-30 px), so it dominates the sum, and the recommendation is
        substantially "whatever makes slide align". That is a legitimate reading
        of equal weighting - a pixel is a pixel - and slide is also the
        interaction taken to be most informative, so the bias points the right
        way. It is NOT the same as each trajectory having equal *leverage* over
        the answer; if that is ever wanted, standardise each mu_i by its spread
        over the candidate set before summing (never by the observed range,
        which would let the sampling history set the weighting).

        `beta` is the risk aversion: the score is mean + beta*sigma of the
        composite (this MINIMISES error, so uncertainty is ADDED, not subtracted
        as it would be when maximising). Coverage is uneven across objectives
        under a decoupled schedule, so a candidate that is only good because
        nobody sampled near it is penalised. beta=0 recovers the plain
        posterior-mean recommendation.

        Returns a dict with the recommended raw parameters, the per-trajectory
        predicted MAEs there, and diagnostics. The caller should CONFIRM it by
        simulating all four trajectories at the recommendation - that is the only
        step that turns a model-based claim into a measured one.
        """
        names = [n for n in sorted(self.optimisers) if self.observations.get(n)]
        if not names:
            raise RuntimeError("recommend() needs at least one fitted model")

        # All models passed here must share one space; with the stock setup that
        # means the vessel-absent models only. The vessel-present model searches
        # a different, one-parameter space and cannot be combined with them.
        keys = list(self.space_for(names[0]))
        for n in names[1:]:
            if list(self.space_for(n)) != keys:
                raise ValueError(
                    f"recommend() needs models over one space; {n!r} differs "
                    f"from {names[0]!r}"
                )
        rng = np.random.default_rng(seed)
        # Uniform in NORMALISED coordinates, which is log-uniform in raw units for
        # the log-scaled parameters - the same measure the search itself uses.
        candidates = rng.uniform(0.0, 1.0, size=(num_candidates, len(keys)))

        mus, sigmas, per_name = [], [], {}
        for name in names:
            # Fitted explicitly: bayes_opt trains `._gp` inside suggest(), so a
            # model that was only registered to would answer from the prior.
            gp = self._fitted_gp(name)
            mu, sigma = gp.predict(candidates, return_std=True)
            # The GP was fitted on NEGATED MAE (bayes_opt maximises), so flip back
            # to "MAE, lower is better" before combining. No other transform is
            # applied - the values stay in raw pixels.
            mu = -mu
            per_name[name] = (mu, sigma)
            mus.append(mu)
            sigmas.append(sigma)

        # Equal weighting = a plain SUM of the raw per-trajectory pixel errors.
        # Unit weights, not 1/n: the composite is then the total misalignment in
        # pixels rather than the mean, which keeps it on the same footing as the
        # per-trajectory numbers. (Scaling all weights by a constant cannot move
        # the argmin anyway; it only changes what the printed score means.)
        weights = np.ones(len(names))
        composite_mu = np.sum(mus, axis=0)
        # Independent GPs, so variances add in quadrature.
        composite_sigma = np.sqrt(np.sum([s ** 2 for s in sigmas], axis=0))
        score = composite_mu + beta * composite_sigma
        best_ix = int(np.argmin(score))

        best_norm = {k: float(candidates[best_ix, j]) for j, k in enumerate(keys)}
        best_raw = self.unnormalise_dict(best_norm)

        # Diagnostic: how far apart are the four models' own best-observed points?
        # Tightly clustered means the objectives broadly agree and a single
        # recommendation is meaningful; scattered means they genuinely conflict
        # and this scalarisation is hiding a trade-off.
        argmins = {}
        for name in names:
            obs = self.observations[name]
            best_params = min(obs, key=lambda t: t[1])[0]
            argmins[name] = self.normalise_dict(best_params)
        spread = 0.0
        if len(argmins) > 1:
            pts = np.array([[a[k] for k in keys] for a in argmins.values()])
            spread = float(np.mean(np.std(pts, axis=0)))

        return {
            "params": best_raw,
            "params_normalised": best_norm,
            "predicted_mae_px": {
                n: float(per_name[n][0][best_ix]) for n in names
            },
            "predicted_sigma_px": {
                n: float(per_name[n][1][best_ix]) for n in names
            },
            "weights": {n: float(w) for n, w in zip(names, weights)},
            "beta": beta,
            "num_candidates": num_candidates,
            # Sum of the four predicted MAEs at the recommendation, in pixels.
            "predicted_total_mae_px": float(composite_mu[best_ix]),
            # ...and the same with the beta*sigma risk penalty added, which is
            # what was actually minimised.
            "composite_score": float(score[best_ix]),
            "per_model_best_observed": {
                n: {
                    "params": min(self.observations[n], key=lambda t: t[1])[0],
                    "mae_px": min(self.observations[n], key=lambda t: t[1])[1],
                }
                for n in names
            },
            # Mean per-dimension std of the four argmins, in unit-cube coords.
            # 0 = identical, ~0.29 = as scattered as uniform random points.
            "argmin_spread": spread,
        }

    def write_to_file(self):
        # print("writing to file!")
        with open(self.all_params_path, "w") as f:
            json.dump(self.all_params, f, indent=4)
        with open(self.all_targets_path, "w") as f:
            json.dump(self.all_targets, f, indent=4)
        # Per-objective observations, so the four surrogates can be refitted
        # without re-simulating.
        obs_path = os.path.join(
            os.path.dirname(self.all_params_path), "bo_observations.json"
        )
        with open(obs_path, "w") as f:
            json.dump(
                {n: [{"params": p, "mae_px": t} for p, t in obs]
                 for n, obs in self.observations.items()},
                f, indent=4,
            )

def main():
    return
    b = BoGp()
    IPython.embed()
    b.write_to_file()
