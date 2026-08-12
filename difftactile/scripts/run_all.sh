#!/usr/bin/env zsh
set -e
set -x
python -m difftactile.scripts.script_apply_scaling
python -m difftactile.scripts.script_pre_main
# NB: no CFL / contact-parameter estimation step here. That diagnostic writes scalar contact
# params where main.py expects 3-element lists, which breaks script_main, so its entrypoint
# wrapper was removed; difftactile/main/cfl_and_contact_params_estimation.py remains as a
# library that main.py imports.
python -m difftactile.scripts.script_main
