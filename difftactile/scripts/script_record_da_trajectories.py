"""Record the four domain-adaptation interactions to video.

    python -m difftactile.scripts.script_record_da_trajectories

Runs press, twist about z, twist about x and slide ONCE each, at the parameters
already in system-params.json, and records the default simulator camera's view
to .mp4 files under difftactile/output/da_recordings/<timestamp>/.

This is a visualisation tool, NOT calibration: it proposes no parameters and
scores nothing. Use ./docker/domain_adaptation.sh for the Bayesian-optimisation
run. Needs a display.

Prefer ./docker/record_da_trajectories.sh, which wraps this.
"""

from difftactile.main.main import record_da_trajectories_main

if __name__ == "__main__":
    record_da_trajectories_main()
