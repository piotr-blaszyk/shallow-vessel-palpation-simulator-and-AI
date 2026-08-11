"""Build the phantom ground-truth segmentation mask from the VGG annotations.

    python -m difftactile.scripts.script_annotate

Reads `files.exp_phantom_labels` (phantom-labels-vgg.json) plus the undistorted
phantom photo, and writes `files.phantom_ground_truth_segmentation_mask` —
the artifact `predict_exp.py` consumes. This module previously had no wrapper,
so it could not be run the documented way even though it is the only producer
of that mask.
"""

from difftactile.data_analysis.experiment.annotate import main

if __name__ == '__main__':
    main()
