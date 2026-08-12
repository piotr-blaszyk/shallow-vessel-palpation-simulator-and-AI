"""GNN entrypoint for the three transfer configurations reported in the paper.

Three datasets are used: (A) simulated, (B) real silicone phantom with shallow
veins, (C) real meat phantom with veins at varying depths. Each configuration
can be trained from scratch or evaluated from the published checkpoint:

    python -m difftactile.scripts.script_segmentation_gnn A-to-B --train
    python -m difftactile.scripts.script_segmentation_gnn A-to-B --eval
    python -m difftactile.scripts.script_segmentation_gnn C-to-B --train
    python -m difftactile.scripts.script_segmentation_gnn A-to-C --eval

Omitting the mode uses that configuration's default (evaluation where a
published checkpoint exists). The configuration may also be given as
DIFFTACTILE_SCENARIO and the mode as DIFFTACTILE_MODE. The older names
(sim-to-silicone, sim-to-meat, silicone-to-meat) are still accepted as aliases.

See run_scenario() in difftactile/cnn/segmentation_gnn.py for what each one does.
"""

from difftactile.cnn.segmentation_gnn import *

if __name__ == '__main__':
    run_scenario()
