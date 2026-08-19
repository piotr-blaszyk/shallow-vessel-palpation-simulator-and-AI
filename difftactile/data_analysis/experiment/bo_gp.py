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
    # the contact stiffness/damping coefficients are physical scale
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
    # extends this work; at that point add them to `pbounds_joint` below (and to
    # LOG_SCALED, since a modulus is scale-like) and re-enable the corresponding
    # lines in main.py::set_contact_params_from_bo().
    LOG_SCALED = {
        'vitactip_youngs_modulus',
        'normal_stiffness',
        'tangential_stiffness',
        'normal_damping',
    }

    def __init__(self):
        # THE SEARCH SPACE, in RAW physical units. ONE joint space: the only
        # search mode (`main.py::domain_adaptation_joint`) fits the sensor's
        # Young's modulus and the sensor<->vein contact normal stiffness
        # TOGETHER, scoring a vessel-absent slide (fidelity to the real
        # photograph) and a vessel-present slide (how far the vessel holds the
        # sensor up) in every iteration. The older two-stage design (a
        # vessel-absent BO over the sensor material per trajectory, then a
        # vessel-present BO at the sensor it chose) is OBSOLETE and its spaces
        # were removed: it could not trade a little fidelity for a lot of vessel
        # contrast, which the joint objective does by construction.
        #
        # Everything else is FIXED during the search:
        #   * sensor Poisson's ratio - held at its system-params.json value.
        #   * the sensor<->vein pair's other coefficients - damping 100,
        #     tangential stiffness 0, friction 0 (Contact.VEIN_* in main.py).
        #   * the sensor<->phantom pair (index 0) - DISABLED project-wide: the
        #     phantom's particles are pinned, so it does not deform and its
        #     four contact coefficients could not move any score. Searching
        #     them (as the earliest 6-D design did) was fitting noise.
        #   * the phantom and vein materials - see the LOG_SCALED note above.
        #
        # Both parameters are log-scaled, spanning one decade each, NARROWED to
        # the region where BOTH halves of the objective are achievable, which a
        # wider box demonstrably was not:
        #
        #   E   in [1e5, 1e6]  the soft end (1e4-1e5) diverged repeatedly -
        #                      inverted elements at 18k-108k - and the very stiff
        #                      end (>1e6) diverged too, with NaN at 1.5e6-1.76e6.
        #                      What survived sat in between.
        #   k_n in [1e4, 1e5]  the vessel only actually stops the sensor at high
        #                      contact stiffness: at the shipped 5e4 it lifts the
        #                      sensor 8.4 mm (vpn 0.99, 13.7 px marker signal),
        #                      while every k_n ~0.2 an earlier search picked gave
        #                      vpn ~0.001 and no signal at all. Below ~1e4 there
        #                      is nothing to find. 1e5 is twice the repo's own
        #                      pair-2 value and the top of the range
        #                      `randomise_contact_params()` samples (5e4).
        #
        # THE UPPER BOUNDS ARE SET BY THE TIMESTEP, NOT BY TASTE. This is an
        # explicit solve at a FIXED dt (contact.dt_override = 1e-5 s): stiffness
        # raises the elastic wave speed c_p = sqrt((lam + 2 mu)/rho) and CFL needs
        # dt <= dx/c_p, so an over-stiff SENSOR blows the solve up. (Measured
        # Courant numbers: E=1e6/nu=0.45 -> C = 0.253; E=2e6/nu=0.45 -> 0.357;
        # the nominal configuration runs at 0.363, so this box is safe.) Note nu
        # enters c_p through (1 - nu)/((1 + nu)(1 - 2 nu)), which diverges as
        # nu -> 0.5: a more incompressible sensor cuts the affordable E, and
        # since nu is NOT searched here the E ceiling must be re-checked if nu
        # is ever changed in system-params.json. CFL is necessary, not
        # sufficient - the contact solve can still blow up inside a
        # CFL-satisfying box, which is why the k_n bound above was found
        # empirically. The vein's penalty force is (k_n + c_n*v_n)*d on a fixed
        # dt, so k_n is the parameter most likely to destabilise the solve; a
        # crash is expected to be the contact, not the material
        # (`_report_vein_crash()` says as much).
        #
        # RESPONSE TO A CRASH, if one ever appears here: halve the UPPER bound
        # of the offending parameter (higher E = stiffer; higher k_n = stiffer
        # contact) and re-test. The LOWER bounds are never the crash cause -
        # softer is always stable - so they stay put.
        #
        # A log-scaled parameter needs a STRICTLY POSITIVE lower bound - log(0)
        # is -inf, so a bound of 0 has no image in log space at all;
        # `_validate_bounds()` raises on that.
        self.pbounds_joint = {
            'vitactip_youngs_modulus': (1e5, 1e6),
            'normal_stiffness': (1e4, 1e5),
        }
        # Default space, used by anything that does not name a model.
        self.pbounds = dict(self.pbounds_joint)
        # The GP's own search space is the unit cube for every parameter, log
        # scaled or not. Defining it here rather than transforming after the fact
        # means the acquisition optimiser, its lengthscales and the random
        # initial design all operate in log coordinates automatically - sampling
        # raw values and taking the log afterwards would silently reintroduce the
        # linear spacing this is meant to remove.
        self.pbounds_normalised = {k: (0.0, 1.0) for k in self.pbounds}
        # Which space each model searches. A name not listed here falls back to
        # the joint space.
        self.model_spaces = {
            "joint": self.pbounds_joint,
        }
        self._validate_bounds()
        # ONE SURROGATE PER MODEL NAME, keyed by name and populated lazily by
        # _optimiser_for(), so this class does not need to know the names up
        # front - main.py owns those. (The joint search uses a single model,
        # "joint".) Each GP learns its own output scale, so no rescaling of
        # the raw objective is required (and none is done - see my_register).
        self.optimisers = {}
        self.params_path = SYSTEM_PARAMS.files.bo_gp_json
        self.target_path = SYSTEM_PARAMS.files.bo_gp_target_json
        self.all_params_path = SYSTEM_PARAMS.files.bo_all_params
        self.all_targets_path = SYSTEM_PARAMS.files.bo_all_targets
        self.all_params = []
        self.all_targets = []
        # Per-model observation log, so a run is reconstructible without
        # refitting: {name: [(params_dict, objective), ...]}
        self.observations = {}

    def _optimiser_for(self, name):
        """The `BayesianOptimization` for one model name, created on first use.

        Each gets its OWN `random_state`, derived from the name, so two models
        would not draw identical "random" sequences (the joint search has one
        model, "joint"; the seed derivation is kept so an extra model can be
        added without changing the existing one's draws).
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
                # Stable across runs and distinct per model. Deliberately
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
        return self.model_spaces.get(name, self.pbounds_joint)

    def _bounds_for_key(self, key):
        """(lo, hi) for `key`, from whichever space defines it.

        Lets normalise/unnormalise work on a parameter dict without being told
        which model produced it. If several spaces ever define the same key
        they must agree on its bounds - asserted rather than assumed, since a
        parameter that normalised differently depending on the model would make
        observations silently incomparable.
        """
        found = None
        for space in self.model_spaces.values():
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
                f"({sorted(self.pbounds_joint)})"
            )
        return found

    def _validate_bounds(self):
        """Fail loudly on bounds a log map cannot represent.

        A log-scaled parameter needs lo > 0: log(0) is -inf and a negative value
        has no log at all, either of which would put a NaN into the GP's training
        data, where it would poison every subsequent proposal rather than raising.
        """
        for space in self.model_spaces.values():
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

    def _finalise_params(self, params):
        """Unit-cube proposal -> raw parameters.

        Split out of `suggest_for` so a proposal can be built and INSPECTED
        before being committed as the current parameter set.

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
        # `bayes_opt` MAXIMISES. An error-like objective (minimised) is negated;
        # the joint objective (vpn - van, maximised) is registered as-is via
        # `maximise=True`.
        signed = float(target) if maximise else -float(target)
        self._optimiser_for(name).register(
            params=self.normalise_dict(self.params),
            target=signed,
        )
        self.observations[name].append((dict(self.params), float(target)))
        self.all_params.append(self.params)
        self.all_targets.append(target)
    
    def suggest_for(self, name, force_random=False):
        """Propose the next parameter set for `name`, unconstrained.

        No fidelity constraint: the joint objective already penalises a poor
        vessel-absent match continuously (see `domain_adaptation_joint`), so
        nothing has to be rejected here.

        `force_random` draws uniformly in the unit cube instead of consulting
        the acquisition function, for the seeding phase where the GP has too
        little data to say anything useful. Drawing in NORMALISED coordinates
        is what makes the initial design log-uniform for the log-scaled
        parameters - each decade gets an equal share of the draws.
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

    def write_to_file(self):
        # print("writing to file!")
        with open(self.all_params_path, "w") as f:
            json.dump(self.all_params, f, indent=4)
        with open(self.all_targets_path, "w") as f:
            json.dump(self.all_targets, f, indent=4)
        # Per-model observations, so a surrogate can be refitted without
        # re-simulating.
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
