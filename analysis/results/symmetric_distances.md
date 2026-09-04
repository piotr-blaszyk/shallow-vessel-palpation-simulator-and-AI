# Symmetric nearest-neighbour distances (poster table)

Definition used on the poster (`symmetric_all`): every scored element (marker of a central
video frame / pixel of a top-view map) measures the distance to the nearest ground-truth
element carrying the label the model PREDICTED for it (predicted vessel -> nearest true
vessel; predicted background -> nearest true background). TP and TN contribute 0, FP and FN
contribute > 0. All distances of a model are pooled and averaged once (mean); infinite
distances (no ground-truth element of that label in the frame/map) are dropped.
Frame space: threshold 0.5, markers in 1920x1080 px, 55 px = 2 mm. Map space: 1 px = 1 mm,
the manuscript's runs and operating points; the recomputed one-directional means match
Table 4 exactly.

| Model | space | FG IoU | symmetric, all elements (mm) | symmetric, FP+FN only (mm) | one-directional, Table 4 (mm) | ASSD (mm) | n elements |
|---|---|---|---|---|---|---|---|
| Sim→Sim | video frame (marker) | 0.19 | 0.66 | 3.74 | 3.03 | 2.54 | 85725 |
| Sim→Sim | top-view map (pixel) | 0.42 | 0.13 | 1.86 | 1.05 | 0.75 | 3102 |
| Sim→Silicone | video frame (marker) | 0.24 | 0.52 | 2.82 | 2.16 | 2.13 | 15118 |
| Sim→Silicone | top-view map (pixel) | 0.28 | 0.13 | 1.60 | 1.21 | 1.49 | 18000 |
| Sim→Meat | video frame (marker) | 0.17 | 1.21 | 4.95 | 4.12 | 3.44 | 26919 |
| Sim→Meat | top-view map (pixel) | 0.21 | 0.43 | 6.87 | 5.49 | 4.36 | 74981 |
| Meat→Silicone | video frame (marker) | 0.16 | 2.46 | 4.67 | 3.87 | 3.27 | 14040 |
| Meat→Silicone | top-view map (pixel) | 0.31 | 0.14 | 1.75 | 1.31 | 1.50 | 18000 |

Map space, symmetric over swept pixels only (pixels the sensor passed over): Sim→Sim 0.28 mm, Sim→Silicone 0.27 mm, Sim→Meat 1.54 mm, Meat→Silicone 0.28 mm
