# Sim-to-Real Subsurface Feature Localisation with a Soft Optical Tactile Sensor

A differentiable-simulation pipeline for locating features **hidden beneath a soft surface** —
blood vessels in a silicone phantom, or plastic straws buried under layers of steak — from the
deformation of markers on a ViTacTip optical tactile sensor.

The core question: **can a model trained entirely in simulation localise subsurface structure in
the real world?** A graph neural network is trained only on synthetic marker displacements
produced by a differentiable FEM simulation, then evaluated on video from a physical sensor
pressing on real tissue.

This is a masters project. It is a fork of
[DiffTactile](https://difftactile.github.io/) (Si et al., ICLR 2024); the upstream
manipulation tasks have been removed and the differentiable Taichi FEM core repurposed for
subsurface sensing. Upstream's original README is preserved at the `upstream-difftactile` tag.

### Project repositories

This work spans two repositories, submitted together to an **ECCV 2026 workshop** as
*"Sim-to-Real Subsurface Feature Localisation with a Soft Optical Tactile Sensor"*:

| Repository | Role |
|---|---|
| **[shallow-vessel-palpation-simulator-and-AI](https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI)** (this one) | **Main repository.** Simulation, dataset generation, GNN training and evaluation — everything needed to reproduce the published results. |
| [shallow-vessel-palpation-robot-control](https://github.com/piotr-blaszyk/shallow-vessel-palpation-robot-control) | Robot control. Drives the DOBOT Magician E6 arm that collected the real tactile recordings for both phantoms. Needed only to *gather new* data, not to reproduce results. |

Data and trained model weights are published on Zenodo as **shallow-vessel-palpation-dataset**
([10.5281/zenodo.21900934](https://doi.org/10.5281/zenodo.21900934)) — see
[Quickstart](#quickstart-docker) below.

```
   Taichi FEM simulation            Real sensor
   sensor + phantom + vein          video of pressing
            │                             │
            ▼                             ▼
   synthetic marker displacements   tracked marker displacements
            │                             │
            └──────────► GNN ◄────────────┘
                     train on sim      evaluate on real
                          │
                          ▼
                 subsurface feature map (+ ROC / IoU)
```

---

## Quickstart (Docker)

**Docker is the only officially supported way to run this repository.** The image pins the
whole stack (CUDA 12.6, Taichi, PyTorch 2.8 + PyTorch Geometric) on a single Python
interpreter, and passes the host X display through so the Taichi GGUI simulator windows work.

**Requirements:** Ubuntu 20.04/22.04/24.04/26.04, an NVIDIA GPU (≥10 GB VRAM),
the NVIDIA driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

> **Use the `main` branch** — it is the only branch, and a plain `git clone` already puts you
> there. See [Branches and tags](#branches-and-tags).

```bash
# 1. Clone (main is the only branch)
git clone https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI.git
cd shallow-vessel-palpation-simulator-and-AI

# 2. Fetch the data bundle from Zenodo (~190 MB) and unpack it into place
#    (datasets + trained checkpoints; see data/MANIFEST.md for what is inside)
#    Zenodo record "shallow-vessel-palpation-dataset", DOI 10.5281/zenodo.21900934
#    -> https://doi.org/10.5281/zenodo.21900934
wget https://zenodo.org/records/21900934/files/shallow-vessel-palpation-data.tar.gz
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz

# 3. Build the image (~10-30 min, downloads several GB), start the container,
#    and open a shell inside it
cd docker
./docker-build.sh
./docker-run.sh
./docker-connect.sh

# 4. You are now INSIDE the container. Verify GPU, dependencies and data
cd docker
./run_pipeline.sh check
```

> **Everything from step 4 onwards runs *inside* the container, from the `docker/` directory.**
> `./docker-connect.sh` puts you there; `cd docker` is what makes the `./run_pipeline.sh` form
> below work. The two exceptions are the viewer scripts — `view_predictions.sh` and
> `annotate_data_docker.sh` — which are launched from the **host** and `docker exec` into the
> container themselves; each says so where it appears.
>
> If you prefer not to keep a shell open, every in-container command below is equivalent to
> `docker exec -it vessel-palpation ./docker/<script> <args>` run from the host.

### Which script produces which figure or table in the paper

Every script below maps onto a specific artifact in the manuscript:

| Script | Produces | In the paper |
|---|---|---|
| `./domain_adaptation.sh` | simulated (red) vs real (green) marker alignment, four interactions | **Fig. 5** (a) press, (b) twist-z, (c) twist-x, (d) slide |
| `./score_all_scenarios.sh` | foreground / background IoU per model | **Table 3** |
| `./score_all_scenarios.sh` | ROC curves per scenario | **Fig. 6** |
| `./view_predictions.sh` | per-frame prediction panels | **Fig. 7** |
| `./vessel_map.sh` | bird's-eye vessel map and confusion overlays | **Fig. 8** (a), (c), (d) |
| `./run_pipeline.sh <config>` | the same IoU / ROC numbers, one configuration at a time | Table 3, Fig. 6 |

Two things in the manuscript are **not** reproduced by this repository:

- **Fig. 8(b)** starts from `vessel_map.sh` output but its yellow annotations were added by hand
  in a photo editor afterwards.
- **Table 4** reports a manual analysis of Fig. 8(b), so nothing here computes it.

`score_all_scenarios.sh --seeds N` additionally produces mean ROC and PR curves with a
variance band across seeds — beyond what the manuscript currently shows, and worth considering
for Fig. 6, since a single-seed curve understates the spread (see
[Training is deterministic](#training-is-deterministic)).

### Domain adaptation: calibrating the simulator

> 📄 **Manuscript: Fig. 5**, and the MAE figures quoted beside it.

The premise of the project is that a model trained purely in simulation transfers to a real
sensor — which only holds if the simulator's markers move like the real ones. This calibrates
the material and contact parameters against the physical sensor by **Bayesian optimisation**:
each iteration proposes a parameter set, replays the four canonical interactions with it, and
scores it by the MAE between simulated and real marker positions at each apex.

```bash
# (inside the container, from the docker/ directory)
./domain_adaptation.sh                          # uses contact.num_opt_steps iterations
DIFFTACTILE_BO_ITERATIONS=20 ./domain_adaptation.sh   # a longer search (~35 s/iteration)
```

**Not differentiable.** An earlier design backpropagated through the Taichi simulation to fit
these parameters; that was abandoned, and all the machinery supporting it has been removed. BO
treats the simulator as a black box, so only forward simulation is needed.

**Every run gets its own timestamped directory** under
`difftactile/output/domain_adaptation/<YYYYmmdd-HHMMSS>/`, so repeated runs accumulate rather
than overwrite — the same convention the training pipeline uses:

| File | Contents |
|---|---|
| `bo_results.json` | best configuration, and every iteration tried |
| `bo_all_params.json` / `bo_all_targets.json` | each parameter set and the MAE it scored |
| `bo_convergence.png` | MAE per iteration, with the running best |
| `best_da_overlay_<name>.png` | **Fig. 5 panels**, from the best configuration |
| `trajectories/iterNNN_<name>.npz` | the collected trajectory: simulated and real markers, MAE, and the parameters that produced it |

Fig. 5 panels are (a) press, (b) twist about *z*, (c) twist about *x*, (d) slide — simulated
markers **red**, real markers **green**.

Measured over 5 iterations: aggregated MAE improved **13.88 px (0.50 mm) → 11.40 px (0.41 mm)**,
with the best set found by the acquisition function. The manuscript reports 14 → 13.5 px over a
longer run. The inter-marker spacing is ~55 px (2 mm), so all of these align to a small fraction
of one grid step.

A diverged parameter set — the FEM solve blows up and markers come back NaN — is scored as the
worst possible value and the search continues, rather than aborting the run.

> **Adopting the result is manual, deliberately.** The best configuration is printed and stored
> in `bo_results.json`, but nothing writes it back into `system-params.json`. Copy it across
> yourself once you are satisfied.

To check a trajectory by eye, set `DIFFTACTILE_SNAPSHOT_DIR` to render the 3D scene periodically
(needs a `DISPLAY`; Taichi GGUI segfaults offscreen in this image):

```bash
DIFFTACTILE_SNAPSHOT_DIR=difftactile/output/da_snapshots ./domain_adaptation.sh
```

### Reproduce the published results

The paper's three models — one per (train → test) configuration over the simulated (**A**),
silicone (**B**) and meat (**C**) datasets — are selected **by name**, with no source editing:

```bash
# (inside the container, from the docker/ directory)

# Evaluate the sim-trained GNN on the real SILICONE phantom -> ROC curve
./run_pipeline.sh A-to-B

# Cross-domain: test the sim-trained checkpoint on real MEAT (no retraining)
./run_pipeline.sh A-to-C

# Train on real MEAT trials, test on silicone
./run_pipeline.sh C-to-B

# ...or all three in sequence
./run_pipeline.sh all-scenarios
```

Each configuration also takes `--train` (reproduce the model from scratch) or `--eval` (load
the published checkpoint); see [Training and evaluating the GNN](#4-training-and-evaluating-the-gnn)
for the full table. Outputs land in `difftactile/output/` (e.g. `roc_curve_A-to-B.pdf`) and `logs/`.

Any run that **trains** writes `*_retrained_<config>` artifacts
(`final_segmentation_model_gnn_meat_retrained_C-to-B.pt`,
`test_loader_gnn_meat_retrained_C-to-B.pickle`, and the `_sim` equivalents) rather than
overwriting the published checkpoints that the evaluation paths read — otherwise running the
configurations in sequence would silently change the reported AUC. The `<config>` tag also
keeps A→B and A→C apart, since they share the same underlying `*_sim` artifact names. Pass
`DIFFTACTILE_OVERWRITE_PUBLISHED=1` if you deliberately want to replace them.

### IoU, AUROC and AP for all six scenarios

> 📄 **Manuscript: Table 3** (the IoU values) and **Fig. 6** (the ROC curves).

The three configurations above, each scored from either the published checkpoint or one you
trained yourself, give six scenarios. `score_all_scenarios.sh` measures all of them in one
pass — per **marker node across video frames**, not from a reprojected phantom map:

```bash
# (inside the container, from the docker/ directory)

# score every scenario whose checkpoint is present -> AUROC_RESULTS.md
./score_all_scenarios.sh

# published checkpoints only (no training needed)
./score_all_scenarios.sh --pretrained

# one configuration
./score_all_scenarios.sh A-to-B
```

Scenarios whose checkpoint is absent are skipped with a note rather than failing the run, so
this is safe on a fresh restore where nothing has been retrained yet. The `retrained` rows need
a `--train` run of the matching configuration first.

> **This is the script that reports the paper's IoU table.** `./score_all_scenarios.sh` prints
> and tabulates **foreground IoU** (vessel present) and **background IoU** (vessel absent) for
> every model, alongside AUROC and AP, into `AUROC_RESULTS.md`. The three `run_pipeline.sh`
> configurations print the same two numbers for their own model. Add `--seeds N` for
> **mean ± std** over N seeds, which is what a table ought to quote.

#### What foreground and background IoU mean

Each is an ordinary per-class intersection-over-union over the **frame-by-frame marker
predictions** — not the reprojected bird's-eye phantom map, which has its own separate IoU from
`vessel_map.sh`:

```
foreground IoU = |predicted vessel AND truly vessel| / |predicted vessel OR truly vessel|
background IoU = the same, for the "vessel absent" class
```

So "foreground" is the **class label**, not a claim about one side: it is the agreement between
prediction and ground truth over the vessel-present class, needing both. A model that finds
every vessel marker but also flags many empty ones scores a low foreground IoU despite perfect
recall, because the union grows.

**Background IoU is always the flattering one** — the negative class is ~89% of nodes on
silicone and ~93% on meat, so even a poor model overlaps with it heavily. Read the foreground
number as the real result. And unlike AUROC and AP, IoU **depends on the decision threshold**
(`DECISION_THRESHOLD`, 0.5), so it carries every caveat about that choice; it is reported
because it is the intuitive quantity, not because it is the most trustworthy one.

#### A→A: the in-domain (simulation → simulation) reference

**A→A is a configuration in its own right**, alongside the three transfers — train on dataset A,
test on a disjoint part of dataset A:

```bash
./run_pipeline.sh A-to-A            # evaluate the published checkpoint (default)
./run_pipeline.sh A-to-A --train    # retrain and evaluate
./score_all_scenarios.sh A-to-A     # into the results table
```

It is the ceiling the transfers are measured against: the gap between A→A and A→B **is the
sim-to-real transfer cost**. On the published checkpoint that is AUROC **0.9369 → 0.7314**.

It scores the *same* published checkpoint as A→B and A→C — the three differ only in which
dataset they are tested on. The simulated test split comes straight out of the test-loader
pickle rather than being re-derived, so it is guaranteed to be the split the checkpoint was
actually held out from.

This is the simulated **test** split, not the validation split. Validation drives early stopping
and checkpoint selection, so a number read off it is optimistic by construction; the test split
is untouched by both.

> A `--train` run of A→B or A→C *also* prints an in-domain reference before its cross-domain
> result, since training already holds out that split. A→A is the standalone way to get the same
> number without retraining.

#### How dataset A is split

**The split is mechanical, not stratified.** `create_splits_single_dataset_scheme()` sorts the
trajectory filenames and cuts at 70% / 85% — 350 train, 75 validation, 75 test of the 500
trajectories. Nothing balances the splits by vein count, depth or any other property.

That turns out not to matter here, because **every trajectory in the shipped dataset contains
exactly one vein** (`vein_polyline` has shape `(frames, 1, 50, 2)` in all 500 files). So a
concern like "the test set contains two vessels but training only ever saw one" cannot arise —
there is no such variation to stratify over. The label balance confirms the split is
unremarkable: the fraction of vein-present frames is 0.5019 / 0.5029 / 0.5017 across
train / val / test, and each trajectory is ~317 frames.

Splitting **by trajectory** (not by clip) is the part that matters, and it is done correctly:
all clips cut from one trajectory land in the same split, so overlapping sliding windows cannot
leak between train and test.

#### Why AUROC *and* average precision

Both are **threshold-free and ranking-based**: they read only the *order* of the predicted
probabilities, never their absolute scale, so no decision threshold is picked anywhere. That is
deliberate — this is a sim-to-real project, and the first thing to shift when a model crosses a
domain gap is the output *scale*, not the ranking. A single-threshold score (IoU, F1) confounds
the two, so it cannot tell "the model doesn't know where the vessels are" from "the model knows,
but its probabilities are miscalibrated for this domain".

Both are reported because each is blind to something the other sees:

| Metric | Ignores | Baseline | Reads as |
|---|---|---|---|
| **AUROC** | the absolute count of false positives, which it normalises by the (huge) negative total | always **0.5** | comparable across papers, but can look reassuring where precision is poor |
| **AP** | true negatives entirely, so the negative majority cannot flatter it | the **positive rate** (~7–11% here) | the honest summary of a needle-in-a-haystack problem |

Because AP's baseline moves with the dataset, the table reports the chance level and the **lift**
(AP / chance) beside it. The two metrics can and do disagree — on the published checkpoints
C-to-B scores *worse* than A-to-B on AUROC (0.679 vs 0.731) but *better* on AP (0.287 vs 0.255).
That disagreement is information, not noise, and is the reason for reporting both.

Outputs, one PDF per scenario:

```
AUROC_RESULTS.md                                        the summary table
difftactile/output/roc_curves/roc_curve_<config>_<weights>.pdf
difftactile/output/pr_curves/pr_curve_<config>_<weights>.pdf
```

Both figure types share their styling — the same threshold colourmap along the curve, the same
marked operating points — so a ROC and a PR panel can sit side by side. The PR figure
additionally draws its **chance baseline** as a dashed line, because a PR curve cannot be read
without it: the same curve is excellent on a 1% positive set and worthless on a 50% one.

The three `run_pipeline.sh` configurations report the same two metrics (and write
`roc_curve_<config>.pdf` / `pr_curve_<config>.pdf` directly under `difftactile/output/`), so you
get them without a separate scoring pass. Use this script when you want all six side by side.

### Inspect predictions frame by frame

> 📄 **Manuscript: Fig. 7.**

An interactive viewer that steps through the test-set frames and shows the confusion overlay
(per marker: green TP, yellow TN, red FP, blue FN — note this is the marker-dot scheme, not the
[vessel map's](#birds-eye-vessel-localisation-map)) alongside the ground truth and soft predictions. The
configuration selects **both** the model weights and the test dataset, so all six scenarios are
reachable by name:

```bash
# (from the docker/ directory on the HOST - this one execs into the container itself)
./view_predictions.sh A-to-B              # central frames (default), published checkpoint
./view_predictions.sh A-to-B --all        # every frame of every window
./view_predictions.sh A-to-C --retrained  # locally trained
./view_predictions.sh A-to-B --all --x11  # force X11 instead of Wayland

# one seed's model out of a sweep (see "Training is deterministic" below)
./view_predictions.sh C-to-B --sweep 20260813-163201 --seed 1
```

`--sweep TS [--seed N]` is the only way to say *which* trained model to view once a sweep has
produced several — `--retrained` alone means "whatever was trained last". `TS` is the sweep's
timestamp (or a full path); an unknown sweep or seed lists what is available rather than
silently falling back to a different model.

> **The viewer shows one model, never an average over seeds — deliberately.** A mean prediction
> over N models is an *ensemble*: a different model, whose AUROC and AP are not the numbers any
> table here reports, so displaying it would put the figures at odds with the results.
> Ensembling is a legitimate research direction, but it would need its own entrypoint and its
> own reported numbers rather than being smuggled in as a display option. To compare seeds, open
> two viewers at different `--seed` values; to quantify the spread, read `AUROC_RESULTS.md`.

Run it from the **host** — like `annotate_data_docker.sh` it `docker exec`s into the running
container for you (start it with `./docker-run.sh` first). It still runs *inside* the
container, because it needs torch and CUDA for inference; there is no bare-metal variant.

#### Which frames: `--central` (default) or `--all`

The model consumes a **`clip_len`-frame sliding window** (`clip_len` is 7) and predicts a label
for *every* frame in it. That is a training decision — supervising all seven frames gives far
more learning signal per window than supervising one — but it is **not** what gets reported.
Only the **central** frame's prediction is scored, because it is the only one with temporal
context on both sides. That masking is real and lives in the code: `dataset.py::get_mask()`
marks exactly frame `clip_len // 2`, and `segmentation_gnn.shared_step()` applies it in the
`val` and `test` stages (but not in `train`), as does `evaluate_and_plot_roc()`.

The viewer offers both views, defaulting to the reported one:

| Flag | Shows | Navigation |
|---|---|---|
| `--central` *(default)* | one prediction per window: its **central frame** — what is reported and scored | `i`/`o` trial, `j`/`k` central frame |
| `--all` | **every** frame of every window, off-centre predictions included — a debugging view | `i`/`o` trial, `j`/`k` clip, `n`/`m` frame |

| Key | `--central` | `--all` |
|---|---|---|
| `i` / `o` | previous / next **trial** | previous / next **trial** |
| `j` / `k` | previous / next **central frame**, within the trial | previous / next **clip**, within the trial |
| `n` / `m` | — | previous / next **frame**, within the clip |
| `q` | quit | quit |

Changing trial (or clip) lands on that unit's first frame, and every move clamps at the ends
rather than wrapping. Nothing advances on its own.

Under `--central` the clips are cut as a **sliding** window (one starting at every frame), so
consecutive windows have consecutive centres and `j`/`k` walks the trial frame by frame. The
one consequence is that a trial's **first and last `clip_len // 2` frames — 3 of each — are
never any window's centre**, so they have no prediction and are skipped. That is inherent to
central-frame reporting, not a limitation of the viewer. Under `--all` the clips are instead
tiled **sequentially** (non-overlapping), so stepping walks each trial once from start to
finish rather than dropping you into the middle of a vein sweep.

The **Metadata** panel reports exactly where you are. Under `--all`: the trial number and what
that trial is (e.g. "1 silicone straw beneath 2 steaks"), the clip number, the frames the clip
covers as a closed interval `[first, last]`, and the frame within the clip. Under `--central`:
the trial, the central-frame number within it, and the video frame that window is centred on.

The five panels (Ground Truth, Hard Prediction, Confusion Matrix, Soft Prediction, Metadata)
are tiled into **one** Qt window rather than five OpenCV ones. That window is a native Wayland
client by default, so it is smooth; `--x11` forces the Xwayland path and is choppy for the same
reason the annotators' is. Five separate windows were not viable under Wayland anyway: a
Wayland client cannot position itself, so `cv2.moveWindow()` silently did nothing and the
panels landed on top of each other.

The **Hard Prediction** and **Confusion Matrix** panels have to turn a probability into a
yes/no, and use `MAP_DECISION_THRESHOLD` (0.58) from `cnn/curve_plots.py` to do it — the same
cut as the vessel map, so the two qualitative views agree. Override it with
`DIFFTACTILE_MAP_THRESHOLD`. The **Soft Prediction** panel beside them shows the underlying
probabilities with no cut at all, and the reported metrics (AUROC, AP) never apply one, so this
is purely a display choice.

### Bird's-eye vessel localisation map

> 📄 **Manuscript: Fig. 8** (a), (c) and (d). Panel **(b)** begins as this script's confusion
> overlay but its yellow shapes were added by hand in a photo editor, and **Table 4** is a manual
> analysis of that edited panel — neither is reproduced here.

Projects the per-marker predictions through the sensor pose onto the phantom surface at
1 mm/pixel and renders the top view against the ground truth:

```bash
# (inside the container, from the docker/ directory)
./vessel_map.sh
```

It writes two confusion maps, each as a raw `.png` and as a `.pdf` carrying a title and a
legend, into `difftactile/output/`:

| Artifact | Compares |
|---|---|
| `confusion_overlay_vein_map.{png,pdf}` | the **prediction** against the video-derived ground truth |
| `ground_truth_sources_overlay.{png,pdf}` | the two **independent ground truths** against each other |

Both use the same colour scheme, defined once in `Visualisation.CONFUSION_COLOURS_RGB`:

| Colour | Meaning |
|---|---|
| 🟩 **green** | both say vessel |
| 🟥 **red** | the **reference** says vessel, the other does not — a **miss** |
| 🟦 **blue** | the **other** says vessel, the reference does not — a **false alarm** |
| ⬛ **black** | neither says vessel |

Red for misses rather than for false alarms is deliberate: in a palpation setting the missed
vessel is the dangerous error, so it takes the warning colour.

The **second** map answers a different question from the first. The vessel positions are known
two independent ways — reprojected from the annotated **video** through the sensor pose (the
same 2D→3D→2D path the prediction takes), and segmented once from an overhead **photo** of the
phantom. That map shows where they disagree, taking the video as the reference and the photo in
the "prediction" role, so the colours carry over unchanged; it also prints the video-vs-photo
IoU. Expect blue wherever the sensor never went — the photo sees the whole phantom, the video
only the swept region — so read it as agreement *within* the swept region rather than as one
source being wrong. It is worth having because it bounds how well any model could score against
either ground truth.

Also writes `segmentation_mask_predicted_aggregated.png` (the predicted mask alone) and
`exp_overlay_downscaled.pdf` (the multi-panel comparison). **Silicone only** — the workspace
bounds and sensor offsets are specific to that rig. Add `--cached` to reuse the probabilities
from a previous run instead of re-running inference.

> **The map is a qualitative figure and applies a decision threshold**, unlike the reported
> AUROC/AP which apply none. It is `MAP_DECISION_THRESHOLD` (0.58) in `cnn/curve_plots.py`, an
> empirical pick rather than a fitted one; set `DIFFTACTILE_MAP_THRESHOLD=0.5` for the
> conventional cut. Because the choice was made by eye on this phantom it should not be assumed
> to transfer, which is why it is confined to figures and touches nothing that is scored.

### Annotate or review the real-world datasets

Manual annotation and annotation review for the two real datasets. In each, one tool does both
jobs: it loads the annotations already on disk, redraws them, and lets you step through frames.

**This is the one entrypoint that runs outside Docker.** These are hand-driven, frame-by-frame
GUI tools, and inside the container every repaint crosses a forwarded X socket, which makes
stepping through frames choppy. Run them natively instead — they need no part of the Docker
stack (no Taichi, no CUDA, no torch), just a small dedicated environment created once:

```bash
micromamba env create -f requirements/annotator-env.yml   # once, ~500 MB, about a minute

# (from the docker/ directory on the HOST - these run on bare metal by design)
./annotate_data_bare_metal.sh --silicone   # click annotator
./annotate_data_bare_metal.sh --meat       # marker-label review
```

The script activates that environment itself. If you would rather use your own interpreter,
point `DIFFTACTILE_ANNOTATOR_PYTHON` at it — it needs numpy, scipy, tqdm, **PySide6** (the
windows) and **av** (video decoding); the script checks for the last two and says so if they
are missing. Note that these two viewers are the **only** part of the project that does not
draw its windows with OpenCV.

**They are Qt 6 applications, so they are native Wayland clients.** The `opencv-python` wheel
ships exactly one Qt platform plugin (`xcb`), so every OpenCV window on a Wayland desktop goes
through Xwayland; the PySide6 wheels bundle the Wayland plugins, so Qt selects `wayland` by
itself and no compatibility layer is involved. Nothing is forced — set `QT_QPA_PLATFORM=xcb`
to fall back to X11, which is what to use inside the container or over X forwarding. Because
Qt needs no X server, `DISPLAY` may be unset entirely: `WAYLAND_DISPLAY` alone is enough.

There is also an **in-container twin**, `docker/annotate_data_docker.sh`, which runs the same
two viewers through the same modules and the same PySide6/PyAV versions inside the Docker
image. It exists so the two can be compared directly — run one, run the other, and any
difference in responsiveness is the container's rather than the code's. Bare metal is still the
normal way to annotate; this is a debugging aid.

```bash
# (from the docker/ directory on the HOST - execs into the container itself)
./annotate_data_docker.sh --meat          # native Wayland window (default)
./annotate_data_docker.sh --meat --x11    # force the X11/Xwayland path instead
```

Run it from the **host** — it `docker exec`s into the running container for you (start it first
with `./docker-run.sh`). The default is a genuine Wayland client: the container gets the
host compositor's socket bind-mounted, which is the entire client/compositor interface, so it
talks to the compositor over the identical IPC path a host-native client uses, with no proxy or
nested compositor in between. `--x11` is the A/B switch for isolating a Wayland-specific bug.
To confirm which you got, run `xlsclients` on the host while a viewer is up — in Wayland mode
it must not appear. A container started before this feature was added has no Wayland socket;
`docker stop vessel-palpation && ./docker/docker-run.sh` picks it up.

> **Known limitation: `--x11` is choppy.** Measured on Ubuntu 24.04 / GNOME Wayland, the
> containerised viewer under Wayland feels **indistinguishable from bare metal** when stepping
> through frames, which is what the socket bind-mount buys. Under `--x11` it is noticeably
> choppy — every repaint takes an extra copy through the Xwayland compatibility layer. This is
> not fixed and is not planned to be: the X11 path exists only as a fallback for X11-only hosts
> and as an A/B switch for isolating Wayland-specific bugs. **If you are annotating rather than
> debugging, use the default Wayland path** (or the bare-metal script). Note the cost is
> Xwayland's, not Docker's — the project's other in-container GUIs are X11-only and pay the same
> price, but they are not hand-driven frame by frame, so it does not matter for them.

Both need a display and set `DIFFTACTILE_INTERACTIVE=1` for you. Keys are printed on start-up
(`m`/`n` video or trial, `k`/`j` frame, `q` quit; silicone additionally: left click to add a
point, `d` to clear the frame, `p` to save).

Annotation points in the silicone tool are real Qt scene objects rather than circles burned
into the image, so **clicking a point selects it and `Delete` removes that one** — alongside
the older `z` (undo last) and `d` (clear frame). The view is scaled to fit the window while the
scene stays in the video's own 1080p pixel grid, so the window is freely resizable and clicks
map back to full-resolution coordinates exactly. (This replaces the old `DIFFTACTILE_VIEW_WIDTH`
downscaling, which existed because OpenCV had to shrink the frame itself before pushing it over
an X socket.)

The meat viewer draws the labels over the **real camera frames**: the bundle ships
`clean/<trial>/frames.mp4`, the 26 decimated frames per trial that preprocessing kept, aligned
1:1 with `marker_labels.npz`. Each trial is decoded and composited once on first visit, so
frame stepping is instant afterwards.

> `--silicone` still needs the dilated videos and annotation pickles, which are **not** in the
> bundle (they are intermediate preprocessing stages — see [`data/MANIFEST.md`](data/MANIFEST.md)).
> Pass `--source DIR` to point at a tree that has them.

### Regenerate the simulated dataset (optional)

The simulated training set ships in the Zenodo bundle, so **this is not required** to
reproduce the results. Run it only if you want to extend the project:

```bash
# (inside the container, from the docker/ directory)

# ~2-3 minutes: a single loop (8 trials), to check the simulator works
./run_pipeline.sh sim-short

# ~2 h 45 m: a full 800-trial collection run
./run_pipeline.sh sim-full
```

> **To regenerate the *published* dataset specifically, set
> `DIFFTACTILE_TRAJECTORIES=3`.** All 500 trajectories in the shipped dataset are
> type 3 ("slide (vein)") — it was collected when the collection loop read
> `range(3, 4)`, which a later commit widened to all four types. A default run
> therefore also produces types 0/1/2, and type 0 yields empty arrays by design
> (it ends in ~36 timesteps, below the `ts > 80` recording threshold).
>
> ```bash
> # (inside the container, from the docker/ directory)
> DIFFTACTILE_TRAJECTORIES=3 ./run_pipeline.sh sim-full
> ```

GUI windows (Taichi GGUI, the cv2 annotation tool, matplotlib) appear on your desktop
automatically — the image ships Vulkan, which GGUI requires — and `DIFFTACTILE_HEADLESS=1`
suppresses them when running over SSH or in CI.

### Rebuilding and restarting the container

If you change the `Dockerfile`, or a `docker-run.sh` change needs picking up (a container keeps
the mounts, environment and ulimits it was **created** with, so a running one never gains them),
stop it and go round again — from the **host**, in `docker/`:

```bash
docker stop vessel-palpation
./docker-build.sh
./docker-run.sh
```

There is no `docker rm` step: `docker-run.sh` starts the container with `--rm`, so **stopping it
also removes it**. That is the container only — the *image* built by `docker-build.sh` persists,
which is why an unchanged `Dockerfile` makes the rebuild a near-instant cache hit and only
`docker-run.sh` really needs re-running. It also means nothing inside the container survives a
stop: keep your work in the repository, which is bind-mounted, not in the container filesystem.

> With the GUI enabled, Taichi may segfault **during interpreter shutdown**, after the
> simulation has already printed `all done` and written its data. This is a teardown-only
> issue in Taichi's GGUI destructor; the output is complete and unaffected. Use
> `DIFFTACTILE_HEADLESS=1` for a clean exit code in scripted runs.

Everything below documents the pipeline in detail, including how to run it outside Docker.

---

## Branches and tags

> ### 👉 `main` is the only branch.
>
> Everything is on `main`: it is the only branch, the only one the documentation describes, and
> the only one the Docker image and Zenodo bundle are tested against. All three of the paper's
> models train and evaluate from it, selected by name (see [Quickstart](#quickstart-docker)).

### The `upstream-difftactile` tag

One tag is published alongside `main`:

| Tag | Commit | What it is |
|---|---|---|
| `upstream-difftactile` | `c9b348e` | The pristine [DiffTactile](https://difftactile.github.io/) state this project forked from, before any masters-project work. Upstream's original README is preserved here. |

It marks the fork point, so you can separate inherited upstream code from the contribution of
this project:

```bash
# What this project changed relative to upstream DiffTactile
git diff upstream-difftactile..main --stat

# Browse the upstream code as it was at the fork
git checkout upstream-difftactile
```

Note that `main` does not descend linearly from this tag — the history was rewritten during
development — so `git diff` is meaningful but `git log upstream-difftactile..main` is not.
The tag is a reference point only; it is not a branch and there is no need to check it out to
run anything in this README.

---

## Setup

### Requirements

- **Python 3.10–3.12.** (Upstream DiffTactile said 3.9.16; that does not apply to this fork's
  torch 2.8 / CUDA 12.6 stack.)
- **An NVIDIA GPU with CUDA is effectively mandatory.** Developed on an RTX 3080 (10 GB);
  `difftactile/main/main.py` requests `device_memory_GB=9` from Taichi. The simulation can be
  switched to CPU (see [Running without a GPU](#running-without-a-gpu)), but the **GNN cannot** —
  `difftactile/cnn/segmentation_gnn.py:570` allocates on a hardcoded `cuda:0` with no fallback, so even
  loading a checkpoint to plot a ROC curve fails on a CPU-only machine.
- **A display is optional, and nothing ever waits for one.** No script blocks on a GUI
  window: figures are written to `difftactile/output/` and the run continues, so the
  simulator, all three GNN scenarios and the preprocessing tools finish unattended over SSH
  or in CI. Set `DIFFTACTILE_INTERACTIVE=1` to get the blocking windows back (you then close
  them by hand), or `DIFFTACTILE_HEADLESS=1` to skip creating windows altogether. See
  [Interactive windows](#interactive-windows).
- Linux (developed on Ubuntu 24.04).

### Install

```bash
# main is the only branch.
git clone https://github.com/piotr-blaszyk/shallow-vessel-palpation-simulator-and-AI.git
cd shallow-vessel-palpation-simulator-and-AI

conda create -n difftactile python=3.10
conda activate difftactile

pip install uv          # the install scripts use uv
pip install -e .
```

Then install the dependencies. There are two sets — the simulator and the ML stack:

```bash
bash requirements/install_dependencies_difftactile.sh   # Taichi sim + ML
bash requirements/install_dependencies_ml.sh            # GNN stack only
```

`install_dependencies_difftactile.sh` is the superset and is enough on its own for most work.

> **Notes on the dependency scripts.** They carry a `#!/bin/zsh` shebang, so invoke them with
> `bash` (as above) or install zsh. They pin CUDA 12.6 wheels and PyTorch Geometric extensions
> built against `torch-2.8.0+cu126` — if you use a different CUDA version, edit the
> `--index-url` and `-f` lines to match. There is **no lockfile**, so exact versions are not
> reproducible; `requirements/requirements_ml.in` records the intended set.
> Two transitively-required packages are not listed explicitly: install them if imports fail:
> ```bash
> uv pip install scipy seaborn
> ```

### Always run from the repository root

Configuration is loaded with a path relative to the working directory
(`difftactile/main/constants.py` reads `difftactile/system_params/system-params.json`).
Running from anywhere else fails immediately. Every command below assumes you are at the repo
root and invokes modules with `python -m`.

---

## Configuration

There are **no command-line flags anywhere in this project.** Three mechanisms control behaviour:

1. **`difftactile/system_params/system-params.json`** — the runtime source of truth for geometry,
   material properties, trajectories, GNN hyperparameters, and all input/output paths.
   Accessed in code as `SYSTEM_PARAMS.gnn.batch_size`, `SYSTEM_PARAMS.files.dataset_root`, …

   The other JSON files in that directory form a small hierarchy:

   | File | Role |
   |---|---|
   | `system-params-distances.json` | **Edit this** for any length. SI metres, unscaled; multiplied by `meta.distance_scaling_factor` into `system-params.json` by `apply_scaling`. |
   | `system-params-youngs-modulus.json` | **Edit this** for stiffnesses; same mechanism. |
   | `system-params.json` | Working config. Partly **regenerated** by `apply_scaling` — hand edits to scaled keys are lost. |
   | `system-params-computed.json` | **Generated** by `pre_main.py` (poses, MPM grid layout). Never hand-edit. |
   | `bo-gp.json` | Best Bayesian-optimisation parameters (inert while `meta.load_params_from_bo = 0`). |
   | `system-params-units.json`, `system-params-literature-values.json` | Documentation only — no code reads them. Useful provenance for the material parameters. |
2. **Module-level constants** — notably `RUN_ON_LAB_MACHINE` at `difftactile/main/main.py:30`,
   which switches Taichi between `ti.cuda` and `ti.cpu`.
3. **Commented-out call lists** — several `main()` functions are a menu of pipeline steps that
   you enable by uncommenting. `difftactile/data_analysis/experiment/preprocess_silicone_data.py` is the clearest
   case; `difftactile/scripts/script_segmentation_gnn.py` toggles between training and evaluation.

To change a parameter, prefer editing the JSON.

---

## Reproducibility — read before running

**The datasets and trained model weights are not in this repository** — they are large binary
artifacts excluded by `.gitignore`. **They are published on Zenodo instead**; see
[the fix](#the-fix-restore-the-published-data-bundle) immediately below the table.

A fresh clone does **not** contain:

| Missing | Expected at | Needed by | Regenerable? |
|---|---|---|---|
| Simulation outputs and intermediates | `difftactile/output/` | most ML and analysis steps | yes, by the sim |
| Canonical 127-marker graph | `difftactile/output/base-graph-connectivity.npz` | **every** dataset/train/eval run | **yes** — `script_base_graph_connectivity`, from an image that *is* in the repo |
| Detected initial marker positions | `difftactile/output/init-marker-positions.npz` | marker tracking, base graph | yes — `script_fisheye_model` |
| Trained GNN weights + test-loader pickle | `saved_models_meat/final_segmentation_model_gnn_meat.pt`, `difftactile/output/test_loader_gnn_meat.pickle` | evaluation / ROC (needs **both**) | only by retraining |
| Silicone dataset | `difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_dense` | silicone training/eval | from raw video via `script_preprocess_silicone_data` |
| Meat experiment trials | `difftactile/manual_or_experimental_data/meat_training_data/clean/` | the meat scenarios (`A-to-C`, `C-to-B`) | from raw video via `script_preprocess_meat_data` |
| Raw experiment videos + robot poses | `.../meat_training_data/raw/<id>.{avi,npz}`, `silicone_training_data/…` | all preprocessing | no — must be supplied |
| Fisheye calibration | `difftactile/output/fisheye_params.npz` | undistortion | yes, with your own checkerboard images |
| Marker-tracker trajectories | `difftactile/output/marker_tracker/domain-adaptation-vascular-markers/traj_{0..3}_out.pkl` | **`script_main` construction** | from raw video via marker tracking |

### The fix: restore the published data bundle

Everything in that table is supplied by the Zenodo archive
([10.5281/zenodo.21900934](https://doi.org/10.5281/zenodo.21900934)), so none of it has to be
regenerated:

```bash
wget https://zenodo.org/records/21900934/files/shallow-vessel-palpation-data.tar.gz
./data/restore_data.sh shallow-vessel-palpation-data.tar.gz   # ~190 MB
./data/restore_data.sh --verify                  # list what is present / missing
```

[`data/MANIFEST.md`](data/MANIFEST.md) documents exactly what the bundle contains and what is
deliberately excluded (raw videos, intermediate preprocessing stages, training logs) — the
exclusions are what take it from 4.5 GB down to ~190 MB without affecting any published result.

Without the bundle, most entrypoints raise `FileNotFoundError`. In particular `script_main`
reads marker-tracker output (`traj_{0..3}_out.pkl`) during construction, so **even the
simulation cannot start from a bare clone**. The parts that work unaided are mesh generation
(`script_generate_*_mesh_gmsh`), `script_apply_scaling` and `script_pre_main`.

Paths are resolved against the repository root by `difftactile/main/paths.py` (override with
`DIFFTACTILE_ROOT`), so scripts run from any working directory and no absolute paths need
editing. Output directories are created on demand.

**Verified:** a clone into an empty directory, plus the bundle, reproduces all three scenarios
in Docker — see [REPRODUCTION_TEST.md](REPRODUCTION_TEST.md) for the transcript and numbers.

Results are **not bit-wise reproducible for the simulation**: `NP_RNG = np.random.default_rng()`
in `difftactile/main/constants.py` is unseeded and drives trajectory and contact-parameter
randomisation, so two runs of `script_main` produce different datasets. Seed `NP_RNG` if you
need repeatable runs. The *evaluation* scenarios are deterministic (AUC agrees to ~15
significant figures across machines).

---

## Running the pipeline

### 1. Simulation — generate synthetic training data

```bash
# scale physical units into simulation units (rewrites system-params.json in place)
python -m difftactile.scripts.script_apply_scaling

# derive poses and MPM grid layout -> writes system-params-computed.json
python -m difftactile.scripts.script_pre_main

# run the Taichi FEM simulation and collect training data
python -m difftactile.scripts.script_main
```

Or all three in order:

```bash
bash difftactile/scripts/run_all.sh   # zsh shebang; invoke with bash or install zsh
```

`run_all_loop.sh` repeats this 100× to accumulate a dataset across randomised runs.

> ⚠️ **`script_apply_scaling` rewrites `system-params.json` in place.** It multiplies the values
> in `system-params-distances.json` (SI metres) and `system-params-youngs-modulus.json` by the
> scale factors in `meta` and merges the result into `system-params.json`, overwriting whatever
> was there. **To change a geometry or material value, edit the `-distances` /
> `-youngs-modulus` file, not `system-params.json`** — otherwise your edit is silently
> discarded on the next run.
>
> ⚠️ **Run the three stages as separate processes** — that is what `run_all.sh` does. A single
> process that imports all three modules up front makes `constants.py` load `system-params.json`
> *before* `apply_scaling` rewrites it, so the simulation runs against pre-scaling constants.
> (An old `script_all.py` had exactly this bug and has been removed.)

Outputs land in `difftactile/output/`: per-trajectory training data at
`difftactile/output/training_data/pickle_<timestamp>/trajectory_XXXX.npz`, containing `markers`,
`markers_mask`, `vein_polyline`, `vein_polyline_mask`, `target_id_array` and `trajectory_type`.

`script_main` opens **two Taichi GGUI windows** unconditionally, so it needs a Vulkan-capable
display; it will not run headless without patching out `visualisation_set_up_gui()`.

**Mesh generation** (optional — regenerates sensor and vein geometry with gmsh; each opens an
interactive Gmsh window that you must close for the script to continue):

```bash
python -m difftactile.scripts.script_generate_vitactip_mesh_gmsh
python -m difftactile.scripts.script_generate_vein_mesh_gmsh
```

**System identification / calibration:**

```bash
python -m difftactile.scripts.script_fisheye_model     # detect markers -> init-marker-positions.{pkl,npz}
python -m difftactile.scripts.script_bo_gp             # Bayesian optimisation of simulation parameters
```

`difftactile/main/cfl_and_contact_params_estimation.py` (CFL timestep + Hertzian
contact-stiffness estimates) is **diagnostics only and destructive if run as an entrypoint**: it
writes contact parameters back to `system-params.json` as scalars, whereas `main.py` expects
3-element lists (one per contact pair), so running it breaks the next `script_main`. Its
`script_*` wrapper has therefore been removed; the module stays because `main.py` imports it.

### 2. Processing real sensor data

Requires the recorded videos.

```bash
python -m difftactile.scripts.script_preprocess_meat_data    # meat experiment: raw/ -> clean/
python -m difftactile.scripts.script_preprocess_silicone_data                 # silicone phantom cleaning pipeline
python -m difftactile.scripts.script_marker_tracker          # marker tracking + tkinter labelling GUI
python -m difftactile.scripts.script_domain_adaptation       # sim-vs-real marker comparison
```

**Meat (`script_preprocess_meat_data`)** — interpolates robot poses onto frames, detects and
Hungarian-reorders markers, then projects the straw geometry through the fisheye camera model to
derive a binary label per marker. Trial geometry is parsed from
[`meat_experiment_spec.md`](difftactile/manual_or_experimental_data/meat_experiment_spec.md),
which catalogues the meat trials (straw depth vs. number of 5 mm steaks above it). Produces,
per trial, `clean/<trial_id>/marker_positions.npz` `(T, 127, 2)` and `marker_labels.npz`
`(T, 127)`.

**Ten trials ship, named descriptively.** The experiment recorded 23 runs, but 13 were repeats
of the same condition at different sensor heights that no split ever referenced; only the 10
the model actually uses are in the bundle. Trial directories are
`<description>-<timestamp>` — e.g. `2-metal-straws-beneath-2-steaks-20260228-235749` — so a
trial's condition is readable from its name. The timestamp remains the trial's identity: the
raw recordings and `meat_experiment_spec.md` are keyed by it, and the pipeline matches
directories on it, so reprocessing from raw rebuilds the same descriptive layout rather than
reverting to bare timestamps. The spec file still lists all 23 as the record of what was
recorded, marking which ten ship.

**It runs from the data bundle.** Each trial ships as `clean/<trial_id>/frames.mp4` (the 26
decimated frames) plus `frames_poses.npz` (their robot poses), so preprocessing starts from
those rather than needing the 1.6 GB raw archive. If `meat_training_data/raw/` is present, any
trial without a shipped video falls back to it — decimating full-rate 1920×1080 AVIs by 15×, as
before. Force the raw path with `DIFFTACTILE_MEAT_FROM_RAW=1`.

> Because `frames.mp4` is H.264, re-running preprocessing **regenerates** the dataset rather
> than reproducing it bit-for-bit: marker positions shift by a median 0.03 px (p99 0.47 px)
> against ~55 px marker spacing, and 16 of ~76000 labels differ (0.02%). The `*.npz` files in
> the bundle stay the authoritative artifacts. The pre-rendered `marker_labels.avi` overlays are
> no longer written by default — the annotation viewer composites the same view live from
> `frames.mp4` and the labels; set `DIFFTACTILE_MEAT_WRITE_OVERLAY=1` if you want the files.

Rebuild the shipped videos from the raw archive (author-side) with:

```bash
python -m difftactile.scripts.script_make_meat_clean_videos
```

**Silicone (`script_preprocess_silicone_data`)** — a chain of directory-to-directory stages: interpolate/trim →
dilate → extract markers → reorder → annotate (a manual cv2 click GUI) → line points → merge into
the simulation `.npz` format → add dense labels, ending at the `_dense` directory that
`exp_data_silicone` points to. Each stage is a method call in `preprocess_silicone_data.main()`. As shipped, the
whole chain is commented out and only `count_annotation_dots()` runs, because the published
`_dense` output is already in the data bundle. **Uncomment the stages you need, in order.**

**Annotation and annotation review.** The one stage of each pipeline that is a manual tool has
its own entrypoint, so it can be reached without editing the commented menu above:

```bash
DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_annotate_silicone         # click annotator
DIFFTACTILE_INTERACTIVE=1 python -m difftactile.scripts.script_browse_meat_annotations   # label review
```

Both load whatever annotations already exist and redraw them, so the same window reviews the
shipped annotations and creates new ones. `docker/annotate_data_bare_metal.sh --silicone|--meat` wraps them,
selects the dedicated bare-metal environment and handles staging the silicone videos, which the
bundle excludes; it is the recommended way to run these two — see
[Annotate or review the real-world datasets](#annotate-or-review-the-real-world-datasets). Note the meat labels are
derived analytically from robot kinematics and straw geometry rather than clicked, so the meat
tool is review-only.

### 3. Preparing datasets

**Run this first** — it builds the canonical 127-marker graph that *every* dataset, training run
and evaluation loads at construction time:

```bash
python -m difftactile.scripts.script_base_graph_connectivity   # -> difftactile/output/base-graph-connectivity.npz
```

Then, for simulated data:

```bash
python -m difftactile.scripts.script_pre_process_sim_data   # Hungarian-reorder sim markers into base-graph order
```

### 4. Training and evaluating the GNN

The paper uses three datasets:

| | Dataset |
|---|---|
| **A** | Simulated, collected in the differentiable tactile simulator. |
| **B** | Real **silicone** phantom — shallow veins ("easy"). |
| **C** | Real **meat** phantom — veins at varying depths ("difficult"). |

and reports three separately trained models, one per (train → test) configuration. Each is
selected **by name**, and each can be either trained from scratch or evaluated from the
published checkpoint — with **no source editing**:

```bash
# Train each of the three models from scratch
python -m difftactile.scripts.script_segmentation_gnn A-to-B --train   # train on sim,  test on silicone
python -m difftactile.scripts.script_segmentation_gnn C-to-B --train   # train on meat, test on silicone
python -m difftactile.scripts.script_segmentation_gnn A-to-C --train   # train on sim,  test on meat

# ...or reproduce the published numbers without retraining
python -m difftactile.scripts.script_segmentation_gnn A-to-B --eval    # evaluate on silicone + ROC
python -m difftactile.scripts.script_segmentation_gnn A-to-C --eval    # cross-domain, no retraining
```

| Config | Train set | Test set | `--train` | `--eval` | Default |
|---|---|---|---|---|---|
| `A-to-B` | simulation | silicone | Trains the large model on sim, tests on silicone. | Loads the published sim-trained checkpoint (`*_sim`), evaluates on silicone, writes the ROC curve (AUC 0.7314). | `--eval` |
| `C-to-B` | meat | silicone | Trains the small model on the real meat trials, tests the best checkpoint on silicone. | Same as `--train` — the published run ends by testing on silicone. | `--train` |
| `A-to-C` | simulation | meat | Trains the large model on sim, tests on meat with every trial in the test split. | Loads the published sim-trained checkpoint and tests it on meat, no retraining. | `--eval` |

Omitting the mode uses the default in the last column (evaluation wherever a published
checkpoint makes it the cheaper path). The configuration can also be given as
`DIFFTACTILE_SCENARIO` and the mode as `DIFFTACTILE_MODE`.

> The **older scenario names still work** as aliases: `sim-to-silicone` → `A-to-B`,
> `sim-to-meat` → `C-to-B`, `silicone-to-meat` → `A-to-C`. Note that `silicone-to-meat` was a
> misnomer: it loads `final_segmentation_model_gnn_sim.pt`, which is the **simulation**-trained
> checkpoint, so the configuration it actually runs is sim → meat (A→C), as the paper describes.

Two architectures exist, and a checkpoint only loads into the one it was trained with —
`GNN(arch="compact")` is the small model (`latent_dim` 64) and `GNN(arch="large")` the large one
(`latent_dim` 256), whose sizes come from the `*_large` keys of the `gnn` config block. The
dispatcher picks the right one per configuration: the two sim-trained models (A→B, A→C) use
the large architecture, the meat-trained model (C→B) the small one.

Training writes `*_retrained_<config>` artifacts rather than overwriting the published
checkpoints — see the note in
[Reproduce the published results](#reproduce-the-published-results). The configuration name is
part of the suffix because A→B and A→C share `train_on_sim()` and therefore the same `*_sim`
artifact names; tagging them keeps a later run from silently replacing an earlier one's
checkpoint. Re-running the same configuration overwrites in place.

#### Training is deterministic

Every training run is seeded, so re-running the same configuration on the same machine
reproduces it exactly — verified bit-identical, down to the last decimal of the IoU. The seed
defaults to **42** and is printed at the top of each run:

```bash
DIFFTACTILE_SEED=7 ./run_pipeline.sh C-to-B    # a different, equally reproducible run
```

`difftactile/main/seeding.py` covers all four sources of randomness — torch and numpy, the
project's shared `NP_RNG` (augmentation, shuffles, per-epoch subset choice), the DataLoader
worker processes, and the CUDA kernels themselves. That last one is what actually closed the
gap: the GNN's `GINEConv` / `global_add_pool` scatter with atomics, and atomic float addition is
not associative, so identical inputs still produced different gradients until deterministic
algorithms were requested. Expect a modest slowdown as a result.

Reproducibility is *same machine, same code, same seed*. Different GPU models, CUDA versions or
worker counts change floating-point reduction order, so cross-machine agreement is not promised.

> ⚠️ **The seed matters a lot here — do not report a single run.** Measured on C→B, seven seeds
> gave **AUROC 0.604–0.779** and **AP 0.239–0.342**. That spread is wider than any difference
> the three configurations claim between one another, because the meat training set is small
> (139 clips) and which subset a run happens to favour dominates the outcome.
>
> So: report a mean and spread over several seeds rather than one number, compare
> configurations as distributions rather than single runs, and **never pick the seed that scores
> best** — that is fitting to the test set as surely as tuning a threshold on it would be.

`--seeds N` does the sweep and the statistics for you:

```bash
# (inside the container, from the docker/ directory)
./score_all_scenarios.sh --seeds 5           # all three configurations
./score_all_scenarios.sh --seeds 5 C-to-B    # just one
```

It trains each configuration from scratch once per seed (0…N−1) and appends a **Seed sweep**
section to `AUROC_RESULTS.md` with mean ± std and range of AUROC, AP **and both IoU values**,
plus every per-seed value so an outlier is visible rather than averaged away. The existing
scenario table is kept — the section is replaced rather than duplicated on a re-run.

Each seed runs in a **fresh subprocess**, so "seed *k*" means the same thing inside a sweep as
it does standalone; run in one process, the second run would inherit CUDA context and RNG state
from the first. Measured on three seeds:

| Config | AUROC mean ± std | AP mean ± std | IoU foreground | IoU background |
|---|---|---|---|---|
| A-to-B | 0.7740 ± 0.0088 | 0.3243 ± 0.0048 | 0.2023 ± 0.0314 | 0.8636 ± 0.0193 |
| C-to-B | 0.6767 ± 0.0208 | 0.2788 ± 0.0106 | 0.1536 ± 0.0133 | 0.5397 ± 0.1184 |
| A-to-C | 0.8238 ± 0.0008 | 0.2203 ± 0.0004 | 0.1922 ± 0.0032 | 0.7765 ± 0.0188 |

Two things worth noticing, both arguments for quoting ± rather than a single run:

- **The variance is very unevenly spread.** A→C is almost seed-independent on AUROC (±0.0008)
  because it trains on the large simulated set, while C→B swings 25× more (±0.0208) on its 139
  meat clips.
- **IoU is markedly less stable than the ranking metrics**, which is expected — it depends on
  the decision threshold, so it absorbs calibration wobble that AUROC and AP ignore by
  construction. C→B's *background* IoU is the extreme case at ±0.1184, an eighth of the metric's
  whole range. A single-seed IoU there is close to meaningless.

**Every seed's weights are kept.** Each sweep creates a new timestamped directory, so sweeping
repeatedly accumulates rather than overwriting:

```
saved_models_sweeps/20260813-163201/
    sweep.json                  all metrics, machine-readable, with each checkpoint's path
    C-to-B_seed00/
        final_segmentation_model_gnn_meat_retrained_C-to-B.pt
        test_loader_gnn_meat_retrained_C-to-B.pickle
    C-to-B_seed01/
        ...
```

The checkpoint and its test-loader pickle travel together deliberately: the pickle holds the
normalisation statistics that checkpoint was trained with, and pairing a checkpoint with the
wrong statistics evaluates it on mis-normalised inputs — a silent wrong answer rather than an
error. (That exact mistake once understated the cross-domain result six-fold.) Nothing prunes
these directories; delete them when you are done. The published checkpoints and the ordinary
`*_retrained_<config>` artifacts are untouched by a sweep.

> Sweeping **trains** once per seed, so it is far slower than plain scoring.

The legacy `python -m difftactile.scripts.script_gnn` entrypoint (large model) still exists.

> ⚠️ **A CUDA GPU is required to even construct the model**, not just to train quickly:
> `difftactile/cnn/segmentation_gnn.py` allocates accumulators with a hardcoded `device='cuda:0'`
> and no CPU fallback, so evaluation fails on a CPU-only machine too.

Hyperparameters come from the `gnn` block of `system-params.json`. Outputs:

| Artifact | Path |
|---|---|
| Per-epoch metrics (CSVLogger) | `logs/my_experiment/run_<timestamp>/metrics.csv` |
| Best checkpoint (monitors `val_iou/1`) | `lightning_logs/.../best-model-sim.ckpt` (A→B, A→C), `best-model-meat.ckpt` (C→B) |
| Published weights read by `--eval` | `saved_models_sim/final_segmentation_model_gnn_sim.pt` (A→B, A→C), `saved_models_meat/final_segmentation_model_gnn_meat.pt` (C→B) |
| Pickled test set + normalisation stats | `difftactile/output/test_loader_gnn_sim.pickle` (A→B, A→C), `..._meat.pickle` (C→B) |
| Weights written by `--train` | the same paths with a `_retrained_<config>` suffix |
| ROC curve (`A-to-B --eval`) | `difftactile/output/roc_curve_A-to-B.pdf` |

`A-to-B --eval` loads the **simulation**-trained checkpoint
(`saved_models_sim/final_segmentation_model_gnn_sim.pt`) and needs **both** it and the
matching `difftactile/output/test_loader_gnn_sim.pickle` (it recovers the normalisation
statistics from the latter), plus the silicone dataset at
`SYSTEM_PARAMS.files.exp_data_silicone` — all three ship in the Zenodo bundle.

> **Corrected in this revision.** `evaluate_and_plot_roc()` previously loaded
> `final_segmentation_model_gnn_meat.pt` — the small, *meat*-trained model — with a
> default-architecture `GNN()`. That made `A-to-B --eval` compute **C→B**, duplicating the
> C-to-B configuration, so the AUC it reported was the meat-trained model on silicone rather
> than the sim-to-real result. Loading the sim checkpoint changes the reported figure from
> **0.6786 to 0.7314**. The `--train` path always built A→B correctly; only this evaluate
> shortcut was wrong.

The ROC PDF is always written to `difftactile/output/roc_curve_A-to-B.pdf`, and the script
exits on its own rather than waiting for you to close a plot window — open the PDF to inspect
the curve. See [Interactive windows](#interactive-windows) if you want the window back.

### 5. Visualisation and results

```bash
python -m difftactile.scripts.script_visualise        # predictions overlaid on sensor frames
python -m difftactile.scripts.script_visualise_mesh   # simulation mesh
python -m difftactile.scripts.script_predict_exp      # run a trained model on experimental data
```

`script_visualise` accepts a configuration name (`A-to-B`, `C-to-B`, `A-to-C`) plus
`--pretrained` / `--retrained` to pick the weights and test set together; `docker/view_predictions.sh`
is the wrapper around it. `script_predict_exp` builds the bird's-eye vessel map, wrapped by
`docker/vessel_map.sh`.

**Domain-adaptation overlay figures.** The sim-vs-real marker alignment images for the four
canonical interactions (press, twist-z, twist-x, slide) are drawn by
`Contact.generate_validation_img()` in [`difftactile/main/main.py`](difftactile/main/main.py),
called from `compute_da_loss()` in the same file — the simulator writes them inline while it
computes the alignment MAE, rather than in a separate plotting stage. They land in
`difftactile/output/da_overlay_{press,twist_z,twist_x,slide}.png` and are produced during a
collection run with `meta.load_params_from_bo == 1`. The real marker positions they are compared
against come from `extract_reorder_save_markers()` in
[`difftactile/data_analysis/experiment/domain_adaptation.py`](difftactile/data_analysis/experiment/domain_adaptation.py)
(`script_domain_adaptation`).

### Interactive windows

**No script waits for user input.** Every figure, mask and 3-D view is written to disk (mostly
under `difftactile/output/`), and the script then carries on and exits. This is what makes the
pipeline safe to run unattended — in Docker, over SSH, or in CI — where a window nobody can
close would hang the run forever. To look at a result, open the saved `.pdf` / `.png`.

Two environment variables change this:

| Variable | Effect |
|---|---|
| `DIFFTACTILE_INTERACTIVE=1` | Restore the original blocking behaviour: `plt.show()` waits, frame browsers step on your key presses (`j`/`k`/`q`), the Gmsh FLTK viewer opens, and the tkinter marker-labelling GUI runs. Requires a real display. |
| `DIFFTACTILE_HEADLESS=1` | Stronger: do not create windows at all. Implied automatically when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set. |
| `DIFFTACTILE_MAX_FRAMES=N` | How many frames a viewer loop steps through before returning when non-interactive. |
| `QT_QPA_PLATFORM=xcb` | Force the Qt annotation viewers onto X11 instead of letting Qt pick Wayland. Use inside the container or over X forwarding. |
| `DIFFTACTILE_ANNOTATOR_PYTHON` | Interpreter `docker/annotate_data_bare_metal.sh` should use, instead of the `vessel-palpation-annotator` micromamba env. |

Two tools exist *only* to collect manual input — the Qt click-annotator in
`preprocess_silicone_data.py::annotate()` and the tkinter labeller in
`marker_tracker.py::VideoPlayer.run()`. Without `DIFFTACTILE_INTERACTIVE=1` they print a note
and return immediately, leaving any annotations already on disk untouched.

The policy lives in `difftactile/main/paths.py`'s neighbour, `difftactile/main/display.py`;
route new GUI calls through its `wait_key()`, `imshow()`, `finish_plot()` and `prompt()`
helpers rather than calling OpenCV or pyplot directly.

### Running without a GPU

Set `RUN_ON_LAB_MACHINE = False` at `difftactile/main/main.py:30` to switch Taichi to the CPU
backend for the **simulation**. Expect a large slowdown.

The **GNN has no working CPU path**: several call sites do check `torch.cuda.is_available()`, but
`difftactile/cnn/segmentation_gnn.py:570` allocates its metric accumulators on a hardcoded `cuda:0`, so
constructing the model at all requires CUDA. Patch that line if you need CPU inference.

---

## Repository layout

```
difftactile/
├── main/                 Taichi FEM simulation core
│   ├── main.py           contact simulation + training-data collection (the big one)
│   ├── pre_main.py       trajectory/geometry precomputation
│   ├── apply_scaling.py  physical units -> simulation units
│   └── generate_*_gmsh.py  mesh generation
├── sensor_model/         ViTacTip FEM model + fisheye camera projection
├── object_model/         phantom, vein, mesh loading
├── cnn/                  GNN and CNN models, datasets, training, visualisation
│   ├── segmentation_gnn.py  the three paper configurations
│   ├── gnn.py            large (simulation-trained) model
│   └── dataset.py        dataset construction and splits
├── data_analysis/
│   ├── experiment/       real sensor data: tracking, calibration, annotation, ROC
│   ├── sim/              simulated data postprocessing
│   ├── training/ testing/  metrics and figures
├── scripts/              entrypoint wrappers — run these
├── system_params/        JSON configuration
├── meshes/               STL geometry
└── manual_or_experimental_data/   reference photos, calibration images, annotations, specs
```

Every runnable entrypoint is a thin wrapper in `difftactile/scripts/`; the logic lives in the
corresponding module. To add one, follow the existing 3-line pattern.

---

## Limitations and known issues

This is a research snapshot rather than a packaged tool, and it is published as-is. The first
two entries below are deliberate modelling decisions, and understanding them is essential to
interpreting the simulator's output correctly. The remainder are known rough edges and are
*not* worth reporting as bugs:

- **Contact compliance is deliberately asymmetric across the three contact pairs.** The
  sensor↔phantom pair is tuned to transfer very little deformation to either body, so the bulk
  of the phantom surface registers only weakly on the sensor membrane. Visible sensor
  deformation is instead driven almost entirely by the sensor↔vein pair. The simulator is
  therefore best understood as a *targeted model of the subsurface feature's mechanical
  signature* rather than a general-purpose soft-body contact solver: the quantity it is built
  to reproduce is the marker displacement field induced by a stiff inclusion beneath a
  compliant surface, not the absolute contact mechanics of the surface itself.

  This is a strong simplification of the underlying physics, and it is adopted because it
  reproduces the target signal well. The resulting marker deformations match those measured on
  the real ViTacTip across all four domain-adaptation trajectories (press, press-and-slide,
  press-and-twist-x, press-and-twist-z) and throughout training-set collection — which is the
  property the downstream GNN actually consumes. Since the network is trained purely in
  simulation and evaluated on real sensor video, the fidelity that matters is fidelity of the
  marker field, and the sim-to-real transfer results reported for the A→B and A→C
  configurations bear this out. Treat absolute contact forces and phantom-surface deformation
  magnitudes as uncalibrated; treat the marker displacements as the validated output.

- **The MPM phantom is kinematically fixed.** Every phantom material point is pinned rather
  than advected: in `Phantom.g2p()` (`difftactile/object_model/phantom.py`) each particle whose
  `is_fixed` flag is set has its velocity and affine velocity field zeroed and its position
  copied unchanged to the next substep. Despite the name, `phantom.fix_bottom_points` does not
  restrict this to the bottom layer — `is_fixed` is assigned `np.ones_like(...)`, so the flag
  is set for *all* particles and the phantom acts as a rigid, immovable body. (The commented-out
  `z_coords <= z_threshold` line and the now-unused `phantom.fixed_points_z_ratio` parameter
  are remnants of the earlier bottom-only scheme.) Note this pins the **particles**; the
  background Eulerian grid is rebuilt each substep as usual.

  This is intentional. Allowing the phantom to deform freely produced two failure modes that
  are avoided entirely by pinning it:

  1. **Collapse of the MPM body**, encountered when the grid node spacing was set too large —
     the deformation field is band-limited by the cell size, so an under-resolved grid cannot
     sustain the phantom's shape.
  2. **High-frequency jitter**, in which the phantom vibrated persistently and small clusters
     of particles were ejected far from the body.

  Because the informative signal is the sensor's response to the subsurface feature, freezing
  the phantom removes both instabilities without affecting the quantity being learned.

- **Configuration is partly "edit the source".** Enabling a pipeline stage, switching train vs.
  evaluate, or choosing the Taichi backend all mean editing Python, not passing a flag.
- **Absolute paths** to the original machine remain in a handful of files (listed above).
- **Non-interactive by default.** Nothing blocks waiting for a window to be closed; inspect
  the saved figures in `difftactile/output/` instead. `DIFFTACTILE_INTERACTIVE=1` restores the
  blocking windows — see [Interactive windows](#interactive-windows).
- **The annotation viewers need their own environment.** They are the only Qt (PySide6) windows
  in the project — everything else uses OpenCV — so they run from the dedicated
  `vessel-palpation-annotator` env on bare metal, not inside the container, which ships neither
  PySide6 nor PyAV. Being native Wayland clients is the point: it removed the stale-frame
  double-present workaround the OpenCV viewers needed, so one keypress moves exactly one frame.
- **`script_main` can segfault at exit when the Taichi GGUI window is open** (exit code 139,
  "Segmentation fault (core dumped)"). This happens *after* `main()` has finished and printed
  `all done`, during CUDA/GGUI teardown, so **the collected trajectories are complete and
  valid** — the crash cannot corrupt them. It only occurs when a display is available:
  `docker-run.sh` passes `DISPLAY` through, and `run_pipeline.sh` forces headless mode only
  when `DISPLAY` is unset. Run with `DIFFTACTILE_HEADLESS=1` to avoid it:

  ```bash
  docker exec -e DIFFTACTILE_HEADLESS=1 vessel-palpation ./docker/run_pipeline.sh sim-short
  ```

  Headless is also markedly faster (~108 s vs ~149 s for `sim-short`), so prefer it unless you
  actually want to watch the simulation.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `FileNotFoundError: difftactile/system_params/system-params.json` | Not running from the repository root. |
| `FileNotFoundError` on a `.npz` / `.pkl` / `.pt` | Missing dataset or checkpoint — see [Reproducibility](#reproducibility--read-before-running). |
| CUDA out of memory in Taichi | Lower `device_memory_GB` in `difftactile/main/main.py:2611`. |
| `ModuleNotFoundError: scipy` / `seaborn` | `uv pip install scipy seaborn`. |
| PyG extension import errors | The `pyg_lib` / `torch_scatter` wheels must match your Torch+CUDA build; edit the `-f` URL in the install scripts. |
| A pipeline step does nothing | Its call is commented out in the module's `main()` — uncomment it. |
| Taichi GGUI / Vulkan error, or hang on a headless machine | `script_main` always opens two GGUI windows and the gmsh scripts open FLTK windows. A display with a working Vulkan driver is required. |
| Edits to `system-params.json` keep reverting | `script_apply_scaling` regenerates them — edit `system-params-distances.json` instead. |
| `script_main` breaks after running the CFL script | It wrote scalar contact params where lists are expected; restore them in `system-params.json`. |

---

## Domain-adaptation reference photographs

The four domain-adaptation interactions are calibrated against **single photographs** of the
real ViTacTip sensor pressed into the silicone phantom — one per interaction, not videos. Each
captures the sensor at that interaction's **apex**, which is why `domain_adaptation.sh` scores
by apex MAE rather than over a whole trajectory: one frame per interaction is all the ground
truth there is.

All five live in `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/` and
are wired up in the `files` block of `difftactile/system_params/system-params.json`.

| Config key | Path (relative to repository root) |
|---|---|
| `da_press` | `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/press_press_depth=4_angle=10_slide_length=50_timestamp=2025-08-30-21-23-31.jpg` |
| `da_twist_z` | `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/twist_z_press_depth=4_angle=90_slide_length=50_timestamp=2025-08-30-21-26-57.jpg` |
| `da_twist_x` | `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/twist_x_press_depth=2_angle=20_slide_length=50_timestamp=2025-08-30-21-34-40.jpg` |
| `da_slide` | `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/slide_press_depth=3_angle=10_slide_length=50_timestamp=2025-08-30-21-49-26.png` |
| `flat_sensor_default_state` | `difftactile/manual_or_experimental_data/domain_adaptation_flat_sensor/press_press_depth=0_angle=10_slide_length=50_timestamp=2025-08-30-21-23-00.jpg` |

The filename encodes the capture settings — `press_depth`, `angle` and `slide_length` — and the
timestamp. The last row is the **undeformed** reference (press depth 0), not an interaction.
Note `da_slide` is the only `.png`; the rest are `.jpg`.

`extract_real_marker_positions()` turns these into `difftactile/output/da_<name>.npz`, which
holds the marker positions the MAE is actually computed against; `domain_adaptation_main()`
calls it automatically and skips any that already exist. The red/green alignment overlays a run
writes (`da_overlay_<name>.png`, manuscript Fig. 5) compare the simulated markers against
exactly these photographs.

**These are not the GNN training data.** The silicone dataset under
`difftactile/manual_or_experimental_data/silicone_training_data/` is a separate experiment — a
raster scan of the phantom, stored as `.avi` videos plus `.npz` metadata — and is unrelated to
these four calibration images.

---

## Citation

This work builds on DiffTactile. If you use this code, please cite the original simulator:

```bibtex
@inproceedings{si2024difftactile,
  title     = {{DIFFTACTILE}: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation},
  author    = {Zilin Si and Gu Zhang and Qingwei Ben and Branden Romero and Zhou Xian and Chao Liu and Chuang Gan},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=eJHnSg783t}
}
```

## License

MIT — see [LICENSE](LICENSE). Inherited from upstream DiffTactile.

## Contact

Piotr Blaszyk — for questions about this fork please open an issue on this repository. The
experimental datasets and trained weights are not in the repository itself; they are published
on Zenodo at [10.5281/zenodo.21900934](https://doi.org/10.5281/zenodo.21900934).
