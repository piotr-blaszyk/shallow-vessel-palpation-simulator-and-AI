"""Build the compressed per-trial meat videos that ship in the data bundle.

    python -m difftactile.scripts.script_make_meat_clean_videos

Author-side: reads meat_training_data/raw/ and writes
meat_training_data/clean/<trial_id>/frames.mp4.
"""

from difftactile.data_analysis.experiment.make_meat_clean_videos import main

if __name__ == "__main__":
    main()
