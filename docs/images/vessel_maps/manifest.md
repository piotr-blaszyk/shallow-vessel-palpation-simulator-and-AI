# Bird's-eye vessel maps on the project page

Written by `docker/website_vessel_maps.sh` (`website_vessel_maps.py`). Each image is a run's `confusion_r00.png` (1 px = 1 mm, ground truth not grown) upscaled x5 nearest-neighbour and saved as lossless WebP. Colours: green = both say vessel, red = missed vessel (truth only), blue = false alarm (prediction only), black = neither.

## A-to-A (sim -> sim), ground truth from simulator

Run: `difftactile/output/vessel_maps/sim-to-sim-test-trajectories_gt-simulator/20260817-131622`  
Model: A-to-A best-of-5 seeds (seed 1, AP 0.4977, AUROC 0.9589) from sweep 20260815-194045  
Threshold: precision >= 0.9 (within 3 mm) reached; recall maximised at threshold 0.6222 (precision 0.900, recall 0.622, 2412 predicted pixels)

| # | file | map (run subfolder) | description | size (mm, w x h) |
|---|---|---|---|---|
| 1 | `sim_to_sim_map_01_trajectory_0474_vessel-present.webp` | `map_01_trajectory_0474_vessel-present` | simulated slide trajectory_0474 (vessel-present), map 1 of 10 in the prediction video's order | 66 x 50 |
| 2 | `sim_to_sim_map_02_trajectory_0478_vessel-present.webp` | `map_02_trajectory_0478_vessel-present` | simulated slide trajectory_0478 (vessel-present), map 2 of 10 in the prediction video's order | 66 x 50 |
| 3 | `sim_to_sim_map_03_trajectory_0463_vessel-absent.webp` | `map_03_trajectory_0463_vessel-absent` | simulated slide trajectory_0463 (vessel-absent), map 3 of 10 in the prediction video's order | 66 x 50 |
| 4 | `sim_to_sim_map_04_trajectory_0430_vessel-present.webp` | `map_04_trajectory_0430_vessel-present` | simulated slide trajectory_0430 (vessel-present), map 4 of 10 in the prediction video's order | 66 x 50 |
| 5 | `sim_to_sim_map_05_trajectory_0458_vessel-present.webp` | `map_05_trajectory_0458_vessel-present` | simulated slide trajectory_0458 (vessel-present), map 5 of 10 in the prediction video's order | 66 x 50 |
| 6 | `sim_to_sim_map_06_trajectory_0485_vessel-absent.webp` | `map_06_trajectory_0485_vessel-absent` | simulated slide trajectory_0485 (vessel-absent), map 6 of 10 in the prediction video's order | 66 x 50 |
| 7 | `sim_to_sim_map_07_trajectory_0490_vessel-present.webp` | `map_07_trajectory_0490_vessel-present` | simulated slide trajectory_0490 (vessel-present), map 7 of 10 in the prediction video's order | 66 x 50 |
| 8 | `sim_to_sim_map_08_trajectory_0488_vessel-present.webp` | `map_08_trajectory_0488_vessel-present` | simulated slide trajectory_0488 (vessel-present), map 8 of 10 in the prediction video's order | 66 x 50 |
| 9 | `sim_to_sim_map_09_trajectory_0469_vessel-absent.webp` | `map_09_trajectory_0469_vessel-absent` | simulated slide trajectory_0469 (vessel-absent), map 9 of 10 in the prediction video's order | 66 x 50 |
| 10 | `sim_to_sim_map_10_trajectory_0450_vessel-present.webp` | `map_10_trajectory_0450_vessel-present` | simulated slide trajectory_0450 (vessel-present), map 10 of 10 in the prediction video's order | 66 x 50 |

## A-to-B (sim -> silicone), ground truth from video

Run: `difftactile/output/vessel_maps/sim-to-silicone_gt-video/20260817-131640`  
Model: A-to-B best-of-5 seeds (seed 2, AP 0.3255, AUROC 0.7832) from sweep 20260815-194045  
Threshold: precision >= 0.9 (within 3 mm) reached; recall maximised at threshold 0.5739 (precision 0.900, recall 0.234, 1342 predicted pixels)

| # | file | map (run subfolder) | description | size (mm, w x h) |
|---|---|---|---|---|
| 1 | `sim_to_silicone.webp` | `silicone` | silicone phantom, ten sweeps, 180 x 100 mm workspace | 180 x 100 |

## A-to-C (sim -> meat), ground truth from video

Run: `difftactile/output/vessel_maps/sim-to-meat_gt-video/20260817-131643`  
Model: A-to-C best-of-5 seeds (seed 2, AP 0.2279, AUROC 0.8405) from sweep 20260815-194045  
Threshold: PRECISION TARGET NOT REACHED: no threshold gives precision >= 0.9 (within 3 mm) with at least 20 predicted pixels on this run (max reachable precision 0.873); FALLING BACK to the F1-optimal threshold 0.5530 (precision 0.550, recall 0.359, F1 0.434, 6202 predicted pixels)

| # | file | map (run subfolder) | description | size (mm, w x h) |
|---|---|---|---|---|
| 1 | `sim_to_meat_trial_01_1-metal-straw-beneath-1-steak-20260228-232013.webp` | `trial_01_1-metal-straw-beneath-1-steak-20260228-232013` | 1 metal straw beneath 1 steak | 157 x 48 |
| 2 | `sim_to_meat_trial_02_1-metal-straw-beneath-2-steaks-20260228-232632.webp` | `trial_02_1-metal-straw-beneath-2-steaks-20260228-232632` | 1 metal straw beneath 2 steaks | 157 x 48 |
| 3 | `sim_to_meat_trial_03_1-metal-straw-beneath-3-steaks-20260228-233031.webp` | `trial_03_1-metal-straw-beneath-3-steaks-20260228-233031` | 1 metal straw beneath 3 steaks | 157 x 48 |
| 4 | `sim_to_meat_trial_04_1-metal-straw-beneath-4-steaks-20260228-233454.webp` | `trial_04_1-metal-straw-beneath-4-steaks-20260228-233454` | 1 metal straw beneath 4 steaks | 157 x 48 |
| 5 | `sim_to_meat_trial_05_1-metal-straw-beneath-6-steaks-20260228-234337.webp` | `trial_05_1-metal-straw-beneath-6-steaks-20260228-234337` | 1 metal straw beneath 6 steaks | 157 x 48 |
| 6 | `sim_to_meat_trial_06_1-metal-straw-on-top-20260228-230937.webp` | `trial_06_1-metal-straw-on-top-20260228-230937` | 1 metal straw on top | 157 x 48 |
| 7 | `sim_to_meat_trial_07_1-silicone-straw-beneath-2-steaks-20260301-001457.webp` | `trial_07_1-silicone-straw-beneath-2-steaks-20260301-001457` | 1 silicone straw beneath 2 steaks | 157 x 48 |
| 8 | `sim_to_meat_trial_08_2-metal-straws-beneath-2-steaks-20260228-235749.webp` | `trial_08_2-metal-straws-beneath-2-steaks-20260228-235749` | 2 metal straws beneath 2 steaks | 157 x 48 |
| 9 | `sim_to_meat_trial_09_3-metal-straws-beneath-2-steaks-20260301-000849.webp` | `trial_09_3-metal-straws-beneath-2-steaks-20260301-000849` | 3 metal straws beneath 2 steaks | 157 x 48 |
| 10 | `sim_to_meat_trial_10_no-straw-20260228-234824.webp` | `trial_10_no-straw-20260228-234824` | no straw | 157 x 48 |

## C-to-B (meat -> silicone), ground truth from video

Run: `difftactile/output/vessel_maps/meat-to-silicone_gt-video/20260817-131654`  
Model: C-to-B best-of-5 seeds (seed 4, AP 0.3481, AUROC 0.7874) from sweep 20260815-194045  
Threshold: precision >= 0.9 (within 3 mm) reached; recall maximised at threshold 0.6129 (precision 0.900, recall 0.232, 1326 predicted pixels)

| # | file | map (run subfolder) | description | size (mm, w x h) |
|---|---|---|---|---|
| 1 | `meat_to_silicone.webp` | `silicone` | silicone phantom, ten sweeps, 180 x 100 mm workspace | 180 x 100 |
