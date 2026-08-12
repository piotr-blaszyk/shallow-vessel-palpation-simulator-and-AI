"""Review the meat-phantom marker annotations frame by frame.

    DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_browse_meat_annotations

Keys: m / n next / previous trial, k / j next / previous frame, q quit.
"""

from difftactile.data_analysis.experiment.preprocess_meat_data import browse

if __name__ == "__main__":
    browse()
