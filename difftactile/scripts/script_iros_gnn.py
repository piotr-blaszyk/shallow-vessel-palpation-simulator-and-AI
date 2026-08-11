"""GNN entrypoint for the three transfer scenarios.

    python -m difftactile.scripts.script_iros_gnn sim-to-silicone
    python -m difftactile.scripts.script_iros_gnn sim-to-meat
    python -m difftactile.scripts.script_iros_gnn silicone-to-meat

The scenario may also be given as DIFFTACTILE_SCENARIO; it defaults to
sim-to-silicone (evaluation + ROC curve, no training). See run_scenario() in
difftactile/cnn/iros_gnn.py for what each scenario does.
"""

from difftactile.cnn.iros_gnn import *

if __name__ == '__main__':
    run_scenario()
