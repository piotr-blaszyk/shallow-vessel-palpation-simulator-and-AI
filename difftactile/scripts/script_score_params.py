"""Score the current system-params.json configuration - no optimisation.

    python -m difftactile.scripts.script_score_params

Runs ONE joint-objective evaluation (vessel-present slide + vessel-free slide)
at whatever parameters are already in system-params.json, and reports vpn, van
and the composite vpn - van.

This is the sanity-check counterpart to ./docker/domain_adaptation.sh, which
searches for parameters. Here nothing is proposed, so the score belongs entirely
to the configuration on disk.

Prefer ./docker/score_params.sh, which wraps this.
"""

from difftactile.main.main import score_current_params_main

if __name__ == "__main__":
    score_current_params_main()
