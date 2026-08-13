"""Domain-adaptation alignment overlays (manuscript Fig. 5).

    python -m difftactile.scripts.script_domain_adaptation

Runs the four canonical interactions - press, twist about z, twist about x and
slide - through the simulator and overlays the simulated marker positions (red)
on the real ones (green) photographed from the physical sensor. Writes one PNG
per interaction to difftactile/output/da_overlay_<name>.png.

Prefer ./docker/domain_adaptation.sh, which wraps this.
"""

from difftactile.main.main import domain_adaptation_main

if __name__ == "__main__":
    domain_adaptation_main()
