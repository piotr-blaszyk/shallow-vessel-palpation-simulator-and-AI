# Vessel centreline / centroid panels on the project page

Written by `docker/website_centreline_panels.sh`, which runs
`analysis/scripts/workflow_centreline_panels.py --all-axes` and converts each panel to
lossless WebP. `--all-axes` is the only difference from the panels the poster prints: the
poster draws the axes on the left column and the bottom row only, so its sixteen cells butt
up flush inside one figure, whereas a web grid scales every cell to one width and would then
render the plots at visibly different sizes.

Every object drawn here comes from exactly the extraction that
`analysis/scripts/detection_ospa.py` scores — same peak finding, same near-duplicate
filtering, same Hungarian matching — so the pictures and the numbers cannot disagree.

**Drawing conventions.** Markers and map pixels carry the project's confusion colours: green =
both say vessel, red = missed vessel (truth only), blue = false alarm (prediction only), grey
or black = neither. **Magenta** is ground truth and **orange** is the prediction; each object
is a straight centreline (total-least-squares fit) with its centroid marked. An index printed
in both colours means the two were matched by the Hungarian assignment; `k*` means matched but
further apart than the 5 mm tolerance; `(k)` in parentheses means unmatched — an invented
vessel in orange, a missed one in magenta.

## Frame-space panels (16)

`cl_grid_<n>_<model>.webp`, one per (ground-truth vessel count `n` = 0…3) × model. Video-frame
space: the sensor's 127 tracked markers, in millimetres on the sensor surface, at the
frame-space operating point (threshold 0.5). All sixteen share one set of axis limits.

Each cell is the frame of that model's test set that is **most representative** of that
difficulty, not the best one: the frame whose OSPA and F₁ both sit closest to the median over
that model's frames *with the same ground-truth vessel count*, in units of the robust spread of
each metric. Stratifying by count matters — 379 of Sim→Sim's 675 frames hold no vessel at all,
and an unstratified median would be theirs.

| n | model | file | frames in stratum | chosen frame | OSPA (mm) | F₁ |
|---|---|---|---:|---|---:|---:|
| 0 | Sim→Sim | `cl_grid_0_sim_to_sim.webp` | 379 | `frame_0000` | 0.00 | 1.00 |
| 0 | Sim→Silicone | `cl_grid_0_sim_to_silicone.webp` | 26 | `frame_0096` | 0.00 | 1.00 |
| 0 | Sim→Meat | `cl_grid_0_sim_to_meat.webp` | 90 | `frame_0064` | 10.00 | 0.00 |
| 0 | Meat→Silicone | `cl_grid_0_meat_to_silicone.webp` | 26 | `frame_0005` | 10.00 | 0.00 |
| 1 | Sim→Sim | `cl_grid_1_sim_to_sim.webp` | 270 | `frame_0395` | 1.05 | 1.00 |
| 1 | Sim→Silicone | `cl_grid_1_sim_to_silicone.webp` | 43 | `frame_0052` | 5.19 | 0.67 |
| 1 | Sim→Meat | `cl_grid_1_sim_to_meat.webp` | 103 | `frame_0048` | 5.55 | 0.67 |
| 1 | Meat→Silicone | `cl_grid_1_meat_to_silicone.webp` | 43 | `frame_0017` | 6.82 | 0.50 |
| 2 | Sim→Sim | `cl_grid_2_sim_to_sim.webp` | 26 | `frame_0335` | 4.09 | 0.67 |
| 2 | Sim→Silicone | `cl_grid_2_sim_to_silicone.webp` | 51 | `frame_0034` | 1.59 | 1.00 |
| 2 | Sim→Meat | `cl_grid_2_sim_to_meat.webp` | 17 | `frame_0180` | 4.59 | 0.50 |
| 2 | Meat→Silicone | `cl_grid_2_meat_to_silicone.webp` | 51 | `frame_0070` | 4.39 | 0.67 |
| 3 | Sim→Sim | `cl_grid_3_sim_to_sim.webp` | 0 | — (placeholder) | — | — |
| 3 | Sim→Silicone | `cl_grid_3_sim_to_silicone.webp` | 0 | — (placeholder) | — | — |
| 3 | Sim→Meat | `cl_grid_3_sim_to_meat.webp` | 9 | `frame_0186` | 4.02 | 0.50 |
| 3 | Meat→Silicone | `cl_grid_3_meat_to_silicone.webp` | 0 | — (placeholder) | — | — |

Three of the sixteen are an explicit "no 3-vessel frame in this test set" placeholder rather
than a blank: only the Meat phantom was built with three straws under one slide.

## Top-view panels (4)

`cl_map_<model>.webp`, one per model, at each map's own aspect ratio. Top-view space: 1 px =
1 mm on the phantom, at each model's own chosen operating point (the vessel-map rule: pixel
precision ≥ 0.9 within 3 mm, recall maximised). Objects are connected components — no peak
finding, because the map builder saves thresholded masks and discards the probability field.

| model | file | map shown | GT vessels | predicted | matched | OSPA (mm) |
|---|---|---|---:|---:|---:|---:|
| Sim→Sim | `cl_map_sim_to_sim.webp` | the single dedicated simulated slide | 1 | 1 | 1 | 0.36 |
| Sim→Silicone | `cl_map_sim_to_silicone.webp` | the whole silicone phantom, all ten sweeps | 10 | 13 | 10 | 4.76 |
| Sim→Meat | `cl_map_sim_to_meat.webp` | `trial_03_1-metal-straw-beneath-3-steaks-20260228-233031`, the median-OSPA trial of ten | 1 | 1 | 1 | 5.00 |
| Meat→Silicone | `cl_map_meat_to_silicone.webp` | the whole silicone phantom, all ten sweeps | 10 | 14 | 10 | 5.06 |

Sim→Meat has ten trial maps rather than one stitched map, so the same representativeness rule
picks among them: the trial whose OSPA is closest to the median over that model's ten.

The recovered ground-truth counts are the physical ones — 1 vessel on the Sim map, 10 on each
silicone whole-phantom map (the ten sweeps the run's `run.json` records) and 1/1/1/1/1/1/1/2/3/0
across the ten meat trials, matching every trial's name including the `no-straw` control.

## Regenerating

```bash
./docker/reproduce_analysis.sh --only 1,4     # per-marker predictions, then the OSPA scoring
./docker/website_centreline_panels.sh         # these images
```

Source numbers: `analysis/results/workflow_panel_choice.json` (which frame each cell shows and
why) and `analysis/results/detection_ospa.json` (the pooled metrics).
