"""Compute AUROC for all six (train -> test) x (pretrained | retrained) scenarios.

    python -m difftactile.scripts.script_auroc_all_scenarios              # all six
    python -m difftactile.scripts.script_auroc_all_scenarios A-to-B       # one config
    python -m difftactile.scripts.script_auroc_all_scenarios --pretrained # published weights only

Writes AUROC_RESULTS.md and one ROC PDF per scenario under
difftactile/output/roc_curves/.
"""

from difftactile.cnn.auroc_all_scenarios import run_from_cli

if __name__ == "__main__":
    run_from_cli()
