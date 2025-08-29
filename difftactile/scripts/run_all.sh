#!/usr/bin/env zsh
set -e
set -x
python -m difftactile.scripts.script_apply_scaling
python -m difftactile.scripts.script_pre_main
# python -m difftactile.scripts.script_cfl_and_contact_params_estimation
python -m difftactile.scripts.script_main
