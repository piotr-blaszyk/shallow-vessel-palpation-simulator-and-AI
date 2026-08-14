"""Manuscript alignment panels: simulated vs real markers, on white.

    python -m difftactile.scripts.script_alignment_figures

Runs press / twist_z / twist_x / slide once each at the parameters in
system-params.json and draws one figure per interaction - simulated markers red,
real markers green, a blue segment joining each corresponding pair.

Uses cached marker positions when they exist, so restyling the figures costs
nothing. Prefer ./docker/alignment_figures.sh, which wraps this.
"""

from difftactile.main.main import alignment_figures_main

if __name__ == "__main__":
    alignment_figures_main()
