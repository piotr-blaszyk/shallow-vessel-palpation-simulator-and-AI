# Video-frame-space metrics (best-of-five instance per model, pooled once)

Per-marker predictions of every central frame of every clip of every trial, pooled
into one set and scored once at threshold 0.5; AP is threshold-free.
No averaging over trials, clips or seeds. The instance is the best-of-five by AP
(cnn/model_selection.py) - the same one the top-view maps use.

| Model | n | positives | TP | FP | FN | TN | MCC | F1 | Prec. | Rec. | FG IoU | BG IoU | AP | chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sim→Sim (A-to-A) | 85725 | 3607 | 3573 | 15316 | 34 | 66802 | 0.39 | 0.32 | 0.19 | 0.99 | 0.19 | 0.81 | 0.50 | 0.042 |
| Sim→Silicone (A-to-B) | 15240 | 1727 | 909 | 2070 | 818 | 11443 | 0.30 | 0.39 | 0.31 | 0.53 | 0.24 | 0.80 | 0.33 | 0.113 |
| Sim→Meat (A-to-C) | 27813 | 1810 | 1493 | 7134 | 317 | 18869 | 0.29 | 0.29 | 0.17 | 0.82 | 0.17 | 0.72 | 0.23 | 0.065 |
| Meat→Silicone (C-to-B) | 15240 | 1727 | 1600 | 8448 | 127 | 5065 | 0.20 | 0.27 | 0.16 | 0.93 | 0.16 | 0.37 | 0.35 | 0.113 |

Instances:
- A-to-A best-of-5 seeds (seed 1, AP 0.4999, AUROC 0.9592) from sweep 20260815-130143
- A-to-B best-of-5 seeds (seed 2, AP 0.3255, AUROC 0.7832) from sweep 20260815-130143
- A-to-C best-of-5 seeds (seed 2, AP 0.2279, AUROC 0.8405) from sweep 20260815-130143
- C-to-B best-of-5 seeds (seed 4, AP 0.3481, AUROC 0.7874) from sweep 20260815-130143
