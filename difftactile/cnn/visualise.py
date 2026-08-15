import os
import pickle
import sys
import time

import cv2
import matplotlib
from difftactile.main.display import (
    destroy_windows, imshow, is_headless, iteration_limit, prompt, wait_key,
)
# Non-interactive backend before pyplot is imported, so plt.figure() does not
# try to open a Tk window on a display-less machine.
if is_headless():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from difftactile.cnn.curve_plots import DECISION_THRESHOLD, MAP_DECISION_THRESHOLD
from difftactile.cnn.dataset import *
from difftactile.cnn.gnn import *
from difftactile.main.paths import repo_path


def _segmentation_gnn(arch):
    """Build the arch-aware GNN from `cnn/segmentation_gnn.py`.

    Imported inside the function rather than at module scope on purpose.
    `cnn/gnn.py` and `cnn/segmentation_gnn.py` each define a class called `GNN`,
    and several modules reach this file through `from ... import *` chains
    (predict_exp.py among them), which resolve `GNN` to whichever name this
    module happens to export. A module-level import here would silently rebind
    their `GNN` to the other class and break checkpoint loading, so the
    arch-aware class is kept out of this module's namespace entirely.
    """
    from difftactile.cnn.segmentation_gnn import GNN as SegmentationGNN
    return SegmentationGNN(arch=arch)


def has_flat_stats_dict(all_stats):
    """True when a stats mapping is a single flat stats dict.

    Meat-scheme loaders store statistics flat; simulation loaders key them by
    curriculum difficulty. Detected by looking for a float difficulty key, since
    what is unpickled here is the `dataset_stats` value rather than the whole
    test-loader dict that `has_flat_stats()` inspects.
    """
    return not any(isinstance(k, float) for k in all_stats)


# The six canonical scenarios the prediction viewer can be pointed at:
# each of the three (train -> test) configurations, loaded either from the
# published checkpoint or from one retrained locally with `--train`.
#
#   arch          : architecture the checkpoint was trained with
#   ckpt_key      : SYSTEM_PARAMS.files key of the published checkpoint
#   stats_key     : test-loader pickle holding the normalisation statistics
#   test_dataset  : which dataset the predictions are shown on
VIEWER_SCENARIOS = {
    "A-to-A": {
        "description": "train on simulation (A), view predictions on held-out simulation (A)",
        "arch": "large",
        "ckpt_key": "final_segmentation_model_gnn_sim",
        "stats_key": "test_loader_gnn_sim",
        "test_dataset": "sim",
    },
    "A-to-B": {
        "description": "train on simulation (A), view predictions on silicone (B)",
        "arch": "large",
        "ckpt_key": "final_segmentation_model_gnn_sim",
        "stats_key": "test_loader_gnn_sim",
        "test_dataset": "silicone",
    },
    "C-to-B": {
        "description": "train on meat (C), view predictions on silicone (B)",
        "arch": "compact",
        "ckpt_key": "final_segmentation_model_gnn_meat",
        "stats_key": "test_loader_gnn_meat",
        "test_dataset": "silicone",
    },
    "A-to-C": {
        "description": "train on simulation (A), view predictions on meat (C)",
        "arch": "large",
        "ckpt_key": "final_segmentation_model_gnn_sim",
        "stats_key": "test_loader_gnn_sim",
        "test_dataset": "meat",
    },
}


def _retrained_variant(rel, config):
    """Path of the `*_retrained_<config>` artifact matching a published one.

    Mirrors `segmentation_gnn._retrained_path()`, which is where a `--train` run
    writes so that the published checkpoints survive.
    """
    base, ext = os.path.splitext(rel)
    return f"{base}_retrained_{config}{ext}"


def _sweep_artifacts(sweep_dir, config, seed):
    """(checkpoint, stats pickle) for one seed of a sweep.

    Both come from the same `<config>_seed<NN>/` directory, which is what keeps a
    checkpoint with the normalisation statistics it was trained with.

    `sweep_dir` may be a full path or just the timestamp, in which case it is
    resolved under `saved_models_sweeps/`. Raises with the available seeds listed
    when the requested one is not there - a missing model should say so, not fall
    back to the published checkpoint and quietly show a different model.
    """
    candidate = sweep_dir
    if not os.path.isdir(candidate):
        candidate = repo_path(f"saved_models_sweeps/{sweep_dir}")
    if not os.path.isdir(candidate):
        raise SystemExit(
            f"No such sweep directory: {sweep_dir}\n"
            f"Looked in {candidate}. Available sweeps:\n  "
            + "\n  ".join(_available_sweeps() or ["(none - run --seeds N first)"])
        )

    seed_dir = os.path.join(candidate, f"{config}_seed{int(seed):02d}")
    if not os.path.isdir(seed_dir):
        available = sorted(
            d for d in os.listdir(candidate)
            if os.path.isdir(os.path.join(candidate, d))
        )
        raise SystemExit(
            f"No model for {config} seed {seed} in {candidate}.\n"
            f"Available: {', '.join(available) or '(none)'}"
        )

    ckpt = [f for f in sorted(os.listdir(seed_dir)) if f.endswith(".pt")]
    stats = [f for f in sorted(os.listdir(seed_dir)) if f.endswith(".pickle")]
    if not ckpt or not stats:
        raise SystemExit(
            f"{seed_dir} is missing a checkpoint (.pt) or its stats pickle "
            f"(.pickle); found: {sorted(os.listdir(seed_dir))}"
        )
    return os.path.join(seed_dir, ckpt[0]), os.path.join(seed_dir, stats[0])


def _available_sweeps():
    """Timestamps of the sweeps present on disk, newest last."""
    root = repo_path("saved_models_sweeps")
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )


def _clip_trials(dataset, clip_ids):
    """`{clip_ix: (trial_id, video_frame_indices)}` for the viewer's clips.

    The viewer renders a flat list of clips; this recovers which TRIAL (a real
    recording or a simulated trajectory) each came from and which video frames
    it covers, so the navigators can offer a trial level on every dataset:

      * meat (`dataset.meat_data`): trial directory basename and the clip's
        frame indices, as `populate_clips_meat()` stored them;
      * simulated / silicone (`dataset.data_points`, the single_dataset scheme):
        the `.npz` file each sliding window was cut from, and
        `start, start + dilation, ...` for its frames.

    Anything else falls back to one synthetic trial per clip, which keeps the
    arithmetic valid even if the trial line is then meaningless.
    """
    meat_clips = getattr(dataset, "meat_data", None) or []
    data_points = getattr(dataset, "data_points", None) or []
    clip_len = getattr(dataset, "clip_len", SYSTEM_PARAMS.gnn.clip_len)
    out = {}
    for clip_ix in clip_ids:
        if clip_ix < len(meat_clips):
            out[clip_ix] = (
                os.path.basename(meat_clips[clip_ix][0]),
                list(meat_clips[clip_ix][2]),
            )
        elif clip_ix < len(data_points) and len(data_points[clip_ix]) == 3:
            file_path, start_ix, dilation = data_points[clip_ix]
            trial = os.path.splitext(os.path.basename(file_path))[0]
            frames = list(range(int(start_ix),
                                int(start_ix) + clip_len * int(dilation),
                                int(dilation)))
            out[clip_ix] = (trial, frames)
        else:
            out[clip_ix] = (f"clip-{clip_ix}", None)
    return out


class _CentralFrameNavigator:
    """Two-level cursor over one central-frame prediction per sliding window.

    The `--central` counterpart to `_MeatNavigator`. The model consumes a
    `clip_len`-frame window and predicts a label for *every* frame in it, but
    only the central frame's prediction is ever reported: it is the one with
    temporal context on both sides, and it is the one the metrics use
    (`dataset.py::get_mask()` marks exactly `clip_len // 2`, and
    `segmentation_gnn.shared_step()` applies that mask in the val/test stages).
    This mode shows that frame and nothing else, so what you step through is
    what the numbers are computed from.

    With the sliding-window tiling (a clip starting at every frame) the central
    frames of consecutive clips are themselves consecutive video frames, so one
    axis disappears entirely and the navigation collapses to two levels:

        i / o   previous / next trial
        j / k   previous / next central frame, within the trial

    The cost is the trial's first and last `clip_len // 2` frames, which are
    never any window's centre and so have no prediction to show. That is
    inherent to reporting the central frame, not a limitation of the viewer.

    `frame_index` is the parallel list of `(clip_ix, frame_in_clip)` built while
    the panels were rendered; only the entries whose `frame_in_clip` is the
    centre are kept. `dataset` is used to name the trials and to recover which
    video frame each window is centred on.
    """

    def __init__(self, frame_index, dataset):
        self.clip_len = max((f for _, f in frame_index), default=0) + 1
        self.centre = self.clip_len // 2

        # One entry per clip: the position in the flat frame list of that clip's
        # central frame. Everything else is dropped, which is the whole point.
        self.positions = []
        clip_ids = []
        for pos, (clip_ix, frame_in_clip) in enumerate(frame_index):
            if frame_in_clip == self.centre:
                self.positions.append(pos)
                clip_ids.append(clip_ix)

        # Trial each retained entry belongs to (meat trial, or the simulated /
        # silicone trajectory file), and the video frame its window is centred
        # on. See `_clip_trials()` for what counts as a trial on each dataset.
        trials_of = _clip_trials(dataset, clip_ids)
        self.entry_trial = []
        self.entry_video_frame = []
        for clip_ix in clip_ids:
            trial, frames = trials_of[clip_ix]
            self.entry_trial.append(trial)
            self.entry_video_frame.append(
                None if frames is None else frames[self.centre]
            )

        # Trials in first-appearance order, so numbering follows playback order.
        self.trials = []
        for t in self.entry_trial:
            if t not in self.trials:
                self.trials.append(t)
        # Retained-entry indices belonging to each trial, in order.
        self.entries_of = {
            t: [i for i, et in enumerate(self.entry_trial) if et == t]
            for t in self.trials
        }

        self.trial_ix = 0
        self.entry_in_trial = 0

    # --- position ---------------------------------------------------------

    @property
    def _entry_ix(self):
        """Index into the retained (central-frame-only) entry list."""
        return self.entries_of[self.trials[self.trial_ix]][self.entry_in_trial]

    @property
    def position(self):
        """Index into the flat frame list of the frame currently shown."""
        return self.positions[self._entry_ix]

    # --- movement ---------------------------------------------------------

    def handle_key(self, key):
        """Apply one keypress. Returns True if the position actually changed."""
        before = (self.trial_ix, self.entry_in_trial)
        if key == ord('i'):
            self._go_trial(self.trial_ix - 1)
        elif key == ord('o'):
            self._go_trial(self.trial_ix + 1)
        elif key == ord('j'):
            self._go_entry(self.entry_in_trial - 1)
        elif key == ord('k'):
            self._go_entry(self.entry_in_trial + 1)
        else:
            return False
        return (self.trial_ix, self.entry_in_trial) != before

    def _go_trial(self, ix):
        self.trial_ix = max(0, min(ix, len(self.trials) - 1))
        # Land at the start of the trial, as the three-level navigator does.
        self.entry_in_trial = 0

    def _go_entry(self, ix):
        n = len(self.entries_of[self.trials[self.trial_ix]])
        self.entry_in_trial = max(0, min(ix, n - 1))

    def walk_keys(self):
        """Key presses that visit every central frame of every trial, in order.

        Used by record mode (qt_viewer.run_browser `auto_keys`): `k` through
        each trial, `o` to the next. Starts from the initial position (trial 1,
        frame 1), which is where a fresh viewer opens.
        """
        keys = []
        for i, trial in enumerate(self.trials):
            keys += ["k"] * (len(self.entries_of[trial]) - 1)
            if i < len(self.trials) - 1:
                keys.append("o")
        return keys

    # --- reporting --------------------------------------------------------

    def metadata_lines(self):
        """The lines shown in the Metadata panel."""
        trial = self.trials[self.trial_ix]
        n_entries = len(self.entries_of[trial])
        video_frame = self.entry_video_frame[self._entry_ix]
        centred_on = "n/a" if video_frame is None else str(video_frame)
        return [
            f"trial: {self.trial_ix + 1}/{len(self.trials)}",
            meat_trial_description(trial),
            f"central frame: {self.entry_in_trial + 1}/{n_entries}",
            f"video frame: {centred_on}",
            f"(centre of a {self.clip_len}-frame window)",
        ]


class _MeatNavigator:
    """Three-level cursor over the viewer's flat list of rendered frames.

    The viewer renders one flat sequence of frames, but they are really nested:
    a trial holds several clips, and a clip holds `clip_len` frames. This maps
    one onto the other so each key pair moves exactly one level, with no key
    doing two jobs:

        i / o   previous / next trial
        j / k   previous / next clip, within the current trial
        n / m   previous / next frame, within the current clip

    Movement is clamped rather than wrapping, and changing trial or clip lands
    on that unit's first frame - so `o` always means "start of the next trial",
    never "wherever I happened to be in the last one".

    `frame_index` is the parallel list of `(clip_ix, frame_in_clip)` built while
    the panels were rendered; `dataset` is the meat dataset, used only to name
    the trials. On the simulated and silicone datasets there is no trial
    structure, so every clip is reported as its own trial - the arithmetic still
    works, the trial line is just not meaningful there.
    """

    def __init__(self, frame_index, dataset):
        self.frame_index = frame_index
        self.clip_len = max((f for _, f in frame_index), default=0) + 1

        # Trial id per clip index, in clip order (meat trial, or the simulated /
        # silicone trajectory file - see `_clip_trials()`).
        clip_ids = sorted({c for c, _ in frame_index})
        trials_of = _clip_trials(dataset, clip_ids)
        self.clip_trial = [trials_of[clip_ix][0] for clip_ix in clip_ids]

        # Trials in first-appearance order, so the numbering follows playback
        # order rather than alphabetical order.
        self.trials = []
        for t in self.clip_trial:
            if t not in self.trials:
                self.trials.append(t)
        # Clip indices belonging to each trial, in order.
        self.clips_of = {
            t: [i for i, ct in enumerate(self.clip_trial) if ct == t]
            for t in self.trials
        }
        # Frame range each clip covers in the source video, for the metadata
        # panel. Reported as a closed interval [first, last].
        self.clip_frames = {}
        for clip_ix in clip_ids:
            frames = trials_of[clip_ix][1]
            if frames is not None:
                self.clip_frames[clip_ix] = (frames[0], frames[-1])

        self.trial_ix = 0
        self.clip_in_trial = 0
        self.frame_in_clip = 0

    # --- position ---------------------------------------------------------

    @property
    def clip_ix(self):
        """Index into the flat clip list of the clip currently shown."""
        trial = self.trials[self.trial_ix]
        return self.clips_of[trial][self.clip_in_trial]

    @property
    def position(self):
        """Index into the flat frame list of the frame currently shown."""
        target = (self.clip_ix, self.frame_in_clip)
        try:
            return self.frame_index.index(target)
        except ValueError:
            return 0

    # --- movement ---------------------------------------------------------

    def handle_key(self, key):
        """Apply one keypress. Returns True if the position actually changed."""
        before = (self.trial_ix, self.clip_in_trial, self.frame_in_clip)
        if key == ord('i'):
            self._go_trial(self.trial_ix - 1)
        elif key == ord('o'):
            self._go_trial(self.trial_ix + 1)
        elif key == ord('j'):
            self._go_clip(self.clip_in_trial - 1)
        elif key == ord('k'):
            self._go_clip(self.clip_in_trial + 1)
        elif key == ord('n'):
            self.frame_in_clip = max(0, self.frame_in_clip - 1)
        elif key == ord('m'):
            self.frame_in_clip = min(self.clip_len - 1, self.frame_in_clip + 1)
        else:
            return False
        return (self.trial_ix, self.clip_in_trial, self.frame_in_clip) != before

    def _go_trial(self, ix):
        self.trial_ix = max(0, min(ix, len(self.trials) - 1))
        # Land at the start of the trial, not at whatever offset we held before.
        self.clip_in_trial = 0
        self.frame_in_clip = 0

    def _go_clip(self, ix):
        n_clips = len(self.clips_of[self.trials[self.trial_ix]])
        self.clip_in_trial = max(0, min(ix, n_clips - 1))
        self.frame_in_clip = 0

    def walk_keys(self):
        """Key presses that visit every frame of every clip of every trial.

        Used by record mode (qt_viewer.run_browser `auto_keys`): `m` through
        each clip's frames, `k` to the next clip, `o` to the next trial.
        """
        keys = []
        for i, trial in enumerate(self.trials):
            n_clips = len(self.clips_of[trial])
            for c in range(n_clips):
                keys += ["m"] * (self.clip_len - 1)
                if c < n_clips - 1:
                    keys.append("k")
            if i < len(self.trials) - 1:
                keys.append("o")
        return keys

    # --- reporting --------------------------------------------------------

    def metadata_lines(self):
        """The five lines shown in the Metadata panel."""
        trial = self.trials[self.trial_ix]
        n_clips = len(self.clips_of[trial])
        first_last = self.clip_frames.get(self.clip_ix)
        if first_last is not None:
            # Closed interval on both ends: these are the first and last frames
            # the clip actually covers, not a half-open range.
            frames_covered = f"[{first_last[0]}, {first_last[1]}]"
        else:
            frames_covered = "n/a"
        return [
            f"trial: {self.trial_ix + 1}/{len(self.trials)}",
            meat_trial_description(trial),
            f"clip: {self.clip_in_trial + 1}/{n_clips}",
            f"frames covered by clip: {frames_covered}",
            f"frame in clip: {self.frame_in_clip + 1}/{self.clip_len}",
        ]


class Visualisation:
    def __init__(self, scenario=None, weights="pretrained", frames="central",
                 sweep_dir=None, sweep_seed=0, trials=None):
        """Interactive viewer for per-frame predictions.

        With `scenario=None` the historical behaviour is kept: the checkpoint and
        test-loader come from `meta.cnn_gnn` and the dataset from the hardcoded
        `if True:` block in visualise_gnn().

        Passing one of VIEWER_SCENARIOS selects the model weights AND the test
        dataset together, so all six canonical scenarios are reachable by name
        instead of by editing source. `weights` is "best" (default: the
        best-of-N seed instance of the published sweep, see
        cnn/model_selection.py), "pretrained" (the checkpoint at the published
        path), "retrained" (what a local `--train` run wrote) or "legacy" (the
        pre-2026-08-15 checkpoint; A-to-B / C-to-B only).

        `frames` picks which of the model's outputs are shown:

          "central"  (default) only each window's central frame, navigated
                     trial/frame. Shows what is actually reported and scored.
          "all"      every frame of every window, navigated trial/clip/frame.
                     Shows what the model emits, including the off-centre
                     predictions that training learns from but reporting
                     ignores - a debugging view.

        Central is the default because it is the reported view: if the two ever
        disagree about how good the model looks, that is the one to see first.

        `sweep_dir` selects one seed's model out of a seed sweep instead
        (`saved_models_sweeps/<timestamp>/`, with `sweep_seed` choosing which),
        which is the only way to say *which* trained model to view once a sweep
        has produced several. It overrides `weights`.

        Deliberately shows ONE model, not an average over the sweep. A mean
        prediction is a different model - an ensemble - whose metrics are not the
        ones any table here reports, so displaying it would make the figures
        disagree with the numbers. Ensembling is a research direction, not a
        display option; it would need its own entrypoint and its own row.
        """
        if frames not in ("all", "central"):
            raise ValueError(f"unknown frames mode {frames!r}; expected 'all' or 'central'")
        # Optional restriction of the test set to some trials - see
        # `_select_trials()`. None shows everything.
        self.trials = trials
        self.scenario = scenario
        self.weights = weights
        # Overwritten below once the source is known; set here so the legacy
        # `scenario=None` path, which returns early, still has a label.
        self.weights_label = weights
        self.frames = frames
        self.scenario_cfg = None

        if scenario is None:
            if SYSTEM_PARAMS.meta.cnn_gnn == 0:
                self.model_path = SYSTEM_PARAMS.files.final_segmentation_model
                self.test_loader = SYSTEM_PARAMS.files.test_loader
            elif SYSTEM_PARAMS.meta.cnn_gnn == 1:
                self.model_path = SYSTEM_PARAMS.files.final_segmentation_model_gnn
                self.test_loader = SYSTEM_PARAMS.files.test_loader_gnn_sim
            return

        if scenario not in VIEWER_SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; expected one of {list(VIEWER_SCENARIOS)}"
            )
        cfg = VIEWER_SCENARIOS[scenario]
        self.scenario_cfg = cfg
        ckpt_rel = getattr(SYSTEM_PARAMS.files, cfg["ckpt_key"])
        stats_rel = getattr(SYSTEM_PARAMS.files, cfg["stats_key"])

        if sweep_dir is not None:
            # One seed's model out of a sweep. The checkpoint and the stats
            # pickle are taken from the SAME directory, never mixed with the
            # defaults: the pickle carries the normalisation statistics that
            # checkpoint was trained with, and pairing a checkpoint with the
            # wrong statistics evaluates it on mis-normalised inputs - a silent
            # wrong answer, not an error.
            self.model_path, self.test_loader = _sweep_artifacts(
                sweep_dir, scenario, sweep_seed
            )
            weights_label = f"sweep {os.path.basename(sweep_dir)} seed {sweep_seed}"
        elif weights in ("best", "legacy"):
            # "best" (the default): the best-of-N seed instance of this
            # configuration in the published sweep - the project-wide
            # convention wherever a single model is shown (see
            # cnn/model_selection.py). "legacy": the pre-2026-08-15 checkpoint,
            # only for A-to-B / C-to-B and only with a 7-frame window.
            from difftactile.cnn.model_selection import resolve_model
            spec = resolve_model(scenario, weights)
            self.model_path, self.test_loader = spec["checkpoint"], spec["stats"]
            weights_label = spec["description"]
        else:
            if weights == "retrained":
                ckpt_rel = _retrained_variant(ckpt_rel, scenario)
                stats_rel = _retrained_variant(stats_rel, scenario)
            self.model_path = repo_path(ckpt_rel)
            self.test_loader = repo_path(stats_rel)
            weights_label = weights

        self.weights_label = weights_label
        print(f"=== {scenario} [{weights_label}]: {cfg['description']} ===")
        print(f"checkpoint:  {self.model_path}")
        print(f"stats:       {self.test_loader}")

    def _build_scenario_dataset(self, all_stats):
        """Dataset for the selected scenario, normalised as the checkpoint expects."""
        cfg = self.scenario_cfg
        if has_flat_stats_dict(all_stats):
            stats, difficulty = all_stats, None
        else:
            difficulty = 1.0 if 1.0 in all_stats else next(iter(all_stats))
            stats = all_stats[difficulty]

        if cfg["test_dataset"] == "sim":
            # A-to-A views the held-out SIMULATED split, taken from the pickle
            # rather than re-derived so it is the split the checkpoint was
            # actually held out from.
            with open(self.test_loader, "rb") as f:
                test_dataset = pickle.load(f)["dataset"]
            if difficulty is not None:
                test_dataset.set_difficulty_level(difficulty)
        elif cfg["test_dataset"] == "silicone":
            full_dataset = MyDataset(
                scheme="single_dataset",
                sim_exp="exp",
                data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
                apply_augmentations=False,
                name="silicone",
            )
            _, _, test_dataset = full_dataset.create_splits(
                train_size=0.0, val_size=0.0, test_size=1.0
            )
            if difficulty is not None:
                test_dataset.set_difficulty_level(difficulty)
        else:
            # Which tiling to cut the trials into depends on the frames mode.
            #
            # "all": sequential (non-overlapping) clips, so stepping through the
            #   viewer walks each trial once from start to finish. The sliding
            #   window would drop the viewer into the middle of a vein sweep,
            #   which looks like the vein has already passed the sensor centre.
            # "central": sliding clips (one starting at every frame), because
            #   only their centres are shown - and the centres of consecutive
            #   sliding windows are themselves consecutive video frames. That is
            #   what makes j/k a real per-frame axis. Sequential clips would
            #   instead give one prediction every clip_len frames, with the rest
            #   of the trial simply missing.
            full_dataset = MyDataset(
                scheme="meat",
                sim_exp="apple",
                data_dir="banana",
                apply_augmentations=False,
                meat_sequential_clips=(self.frames == "all"),
                name="meat",
            )
            _, _, test_dataset = full_dataset.create_splits(all_to_test=True)
        test_dataset.set_stats(stats)
        test_dataset.eval()
        self._select_trials(test_dataset)
        return test_dataset

    def _select_trials(self, dataset):
        """Order `dataset`'s clips by trial and time, and restrict to `self.trials`.

        `self.trials` is a comma-separated list. Each item is either the token
        `first-vessel-present` - the first trial (in dataset order) with at
        least one vessel-present frame - or a substring of a trial id (a meat
        trial directory such as `2-metal-straws-beneath-2-steaks-20260228-235749`
        or a simulated/silicone file stem such as `trajectory_0426`). Trials
        keep their dataset order. Raises if nothing matches, since a viewer
        that silently shows a different trial than asked for is worse than one
        that stops.

        Exists for the README recordings, which show one vessel-present
        simulated trajectory rather than all 75 held-out ones (~23k sliding
        windows, minutes of rendering, and hours of video at half a second per
        frame). Perfectly usable by hand too.
        """
        wanted = [t.strip() for t in str(self.trials or "").split(",") if t.strip()]
        is_meat = getattr(dataset, "scheme", None) == "meat"
        # The meat split datasets carry their clips in `meat_data` only.
        clips = list(dataset.meat_data if is_meat else dataset.data_points)
        trials_of = _clip_trials(dataset, range(len(clips)))
        # Trial ids in first-appearance order, and the clips of each, in the
        # order of the frames they cover. The pickled simulated test set keeps
        # the shuffled clip order it was trained with, which as a viewing order
        # would jump back and forth in time; sorting here changes only what
        # the viewer shows next, never what the model is given.
        trial_ids = []
        clips_of = {}
        for clip_ix, (trial, frames) in trials_of.items():
            if trial not in clips_of:
                trial_ids.append(trial)
                clips_of[trial] = []
            clips_of[trial].append((frames[0] if frames else clip_ix, clips[clip_ix]))
        for trial in trial_ids:
            clips_of[trial] = [c for _, c in sorted(clips_of[trial], key=lambda t: t[0])]
        if not wanted:
            keep = trial_ids
        else:
            keep = self._match_trials(wanted, trial_ids, clips_of)
        selected = [c for t in trial_ids if t in keep for c in clips_of[t]]
        if is_meat:
            dataset.meat_data = selected
        else:
            dataset.data_points = selected
        if wanted:
            print(f"Trials restricted to {len(keep)}/{len(trial_ids)}: {', '.join(keep)}")

    @staticmethod
    def _match_trials(wanted, trial_ids, clips_of):
        """The trial ids selected by the `--trials` items, in dataset order."""

        def has_vessel(trial):
            """True if any frame of the trial is labelled vessel-present."""
            first = clips_of[trial][0]
            path = first[0]
            if os.path.isdir(path):  # meat trial directory
                labels = np.load(os.path.join(path, "marker_labels.npz"))["marker_labels"]
                return bool(np.any(labels))
            with np.load(path) as d:  # simulated / silicone .npz
                return bool(np.any(d["vein_classification"]))

        keep = []
        for item in wanted:
            if item == "first-vessel-present":
                match = next((t for t in trial_ids if has_vessel(t)), None)
                matches = [match] if match is not None else []
            else:
                matches = [t for t in trial_ids if item in t]
            if not matches:
                raise SystemExit(
                    f"--trials {item!r} matches no trial. Available "
                    f"({len(trial_ids)}): {', '.join(trial_ids[:20])}"
                    + (" ..." if len(trial_ids) > 20 else "")
                )
            keep += [t for t in matches if t not in keep]
        return keep

    @staticmethod
    def calculate_iou(ground_truth, prediction):
        intersection = np.logical_and(ground_truth, prediction)
        union = np.logical_or(ground_truth, prediction)
        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        return iou_score

    # The project-wide confusion colour scheme, in RGB, as floats in [0, 1].
    #
    # Read it as "what is there, and did we find it":
    #   green  agreement on a positive - both say vessel
    #   red    a MISS - ground truth says vessel, the prediction does not
    #   blue   a FALSE ALARM - the prediction says vessel, ground truth does not
    #   black  agreement on a negative - neither says vessel
    #
    # Red for misses and blue for false alarms is the deliberate part: a missed
    # vessel is the dangerous error in a palpation setting, so it takes the
    # warning colour. Keep the four in step wherever confusion is drawn.
    CONFUSION_COLOURS_RGB = {
        "tp": (0.0, 1.0, 0.0),   # green
        "fn": (1.0, 0.0, 0.0),   # red
        "fp": (0.0, 0.0, 1.0),   # blue
        "tn": (0.0, 0.0, 0.0),   # black
    }

    @staticmethod
    def create_confusion_matrix_overlay(ground_truth, prediction):
        """Per-pixel confusion overlay of `prediction` against `ground_truth`.

        Returns float RGB in [0, 1] - matplotlib's order, so `plt.imshow()` takes
        it directly. Anything writing it with `cv2.imwrite()` must convert first;
        `confusion_overlay_bgr()` below does that and is what the PNG writers use.

        Both inputs are treated as binary: non-zero is positive. The two need not
        be a model prediction and a label - `vessel_map.sh`'s second artifact
        passes the video-derived and photo-derived ground truths, taking the
        video as the reference and the photo as the thing being judged.
        """
        colours = Visualisation.CONFUSION_COLOURS_RGB
        gt = np.asarray(ground_truth) != 0
        pred = np.asarray(prediction) != 0

        overlay = np.zeros((*gt.shape, 3), dtype=float)
        overlay[~gt & ~pred] = colours["tn"]   # black: agreed negative
        overlay[gt & pred] = colours["tp"]     # green: agreed positive
        overlay[gt & ~pred] = colours["fn"]    # red:   missed
        overlay[~gt & pred] = colours["fp"]    # blue:  false alarm
        return overlay

    @staticmethod
    def confusion_overlay_bgr(ground_truth, prediction):
        """`create_confusion_matrix_overlay` as a uint8 BGR image for cv2.imwrite.

        The channel flip is the whole point: the helper above returns RGB, and
        handing RGB straight to `cv2.imwrite` silently swaps red and blue - which
        on this colour scheme turns every miss into a false alarm and vice versa.
        """
        rgb = Visualisation.create_confusion_matrix_overlay(ground_truth, prediction)
        return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    @staticmethod
    def confusion_legend_handles(plt, positive="vessel", reference="ground truth",
                                 candidate="prediction"):
        """Legend patches for the confusion colour scheme, in a fixed order.

        Parameterised because the scheme is used for two different comparisons:
        prediction-vs-ground-truth, and photo-derived-vs-video-derived ground
        truth. The colours and their meanings do not change, only the words for
        who is being compared with whom.
        """
        colours = Visualisation.CONFUSION_COLOURS_RGB
        labels = [
            ("tp", f"both say {positive}"),
            ("fn", f"{reference} says {positive}, {candidate} does not"),
            ("fp", f"{candidate} says {positive}, {reference} does not"),
            ("tn", f"neither says {positive}"),
        ]
        return [
            plt.Rectangle((0, 0), 1, 1, fc=colours[key], ec="0.5", lw=0.5, label=text)
            for key, text in labels
        ]


    def visualize_experiment(
            self,
            mode,
            frame_num=None
        ):
        if mode == 'curved': 
            npz_path = SYSTEM_PARAMS.files.exp_video_npz
            video_path = SYSTEM_PARAMS.files.vein_slide_across_extracted_markers
        elif mode == 'straight':
            npz_path = SYSTEM_PARAMS.files.experiment_straight_markers_npz
            video_path = SYSTEM_PARAMS.files.experiment_straight_processed_video

        self.exp_data = np.load(npz_path)
        
        # Initialize model
        model = SegmentationModel()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Initialize video capture
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return

        # Get parameters needed for clip extraction
        w = SYSTEM_PARAMS.fisheye_model.crop_width
        h = SYSTEM_PARAMS.fisheye_model.crop_height
        k = SYSTEM_PARAMS.fisheye_model.down_scaling_factor
        clip_len = SYSTEM_PARAMS.gnn.clip_len
        dilation = 8
        dilated_clip_len = clip_len * dilation
        n = self.exp_data['markers'].shape[0]
        m = 0

        while m <= n - dilated_clip_len:
            if frame_num is not None:
                start_ix = frame_num
            else:
                # start_ix = NP_RNG.integers(0, n - dilated_clip_len)
                start_ix = m
            
            # Get and process clip
            clip = MyDataset.get_clip(h, w, k, self.exp_data, clip_len, dilation, start_ix=start_ix)
            
            with torch.no_grad():
                clip_input = clip.to(device)
                logits = model(clip_input)
                probs = torch.sigmoid(logits)
                pred = (probs > DECISION_THRESHOLD).float()
                pred = pred.cpu()
            
            # Convert tensors to numpy arrays
            image_seq = clip.numpy().squeeze()  # Shape: (T, H, W)
            pred_seq = pred.numpy().squeeze()  # Shape: (T, H, W)

            current_frame = 0
            total_frames = image_seq.shape[0]

            # Interactively this steps through frames until 'q'. Non-interactively
            # nobody can press a key, so play a bounded number of frames and move
            # on (override with DIFFTACTILE_MAX_FRAMES).
            frame_limit = iteration_limit("DIFFTACTILE_MAX_FRAMES", total_frames)
            shown = 0

            while frame_limit is None or shown < frame_limit:
                shown += 1
                # Prepare the current frame
                current_image = image_seq[current_frame]
                current_pred = pred_seq[current_frame]
                
                # Normalize images for display
                current_image = (current_image * 255).astype(np.uint8)
                current_pred = (current_pred * 255).astype(np.uint8)

                # Scale up images by 4x using NEAREST neighbor interpolation
                scale_factor = 2
                current_image = cv2.resize(current_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                current_pred = cv2.resize(current_pred, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

                # Add frame counter text
                frame_text = f"Frame: {current_frame + 1}/{total_frames} | Start Index: {start_ix}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5  # Reduced from 1.0
                font_thickness = 1  # Reduced from 2
                text_color = (255, 255, 255)  # White text
                
                # Get text size to position it at the bottom
                text_size = cv2.getTextSize(frame_text, font, font_scale, font_thickness)[0]
                text_x = 5  # Reduced from 10
                text_y = current_image.shape[0] - 10  # Reduced from 20
                
                # Add black background for text visibility
                padding = 3  # Reduced from 5
                cv2.rectangle(current_image, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                cv2.rectangle(current_pred, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                
                # Add text to both images
                cv2.putText(current_image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                cv2.putText(current_pred, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

                # Create display windows
                imshow(cv2, 'Input Image', current_image)
                imshow(cv2, 'Predicted Image', current_pred)

                # Read and display video frame
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_ix + current_frame * dilation)
                ret, video_frame = video_cap.read()
                if ret:
                    # Crop video frame using fisheye model parameters
                    start_x = SYSTEM_PARAMS.fisheye_model.crop_x
                    start_y = SYSTEM_PARAMS.fisheye_model.crop_y
                    crop_width = SYSTEM_PARAMS.fisheye_model.crop_width
                    crop_height = SYSTEM_PARAMS.fisheye_model.crop_height
                    
                    # Crop the frame using the fisheye model parameters
                    scale_factor = 1/2
                    video_frame = video_frame[start_y:start_y+crop_height, start_x:start_x+crop_width]
                    video_frame = cv2.resize(video_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                    
                    # Add the same frame counter text to video frame
                    cv2.rectangle(video_frame, 
                                (text_x - padding, text_y - text_size[1] - padding),
                                (text_x + text_size[0] + padding, text_y + padding),
                                (0, 0, 0), -1)
                    cv2.putText(video_frame, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                    
                    imshow(cv2, 'Video Frame', video_frame)

                # Position windows side by side
                window_width = current_image.shape[1]
                cv2.moveWindow('Input Image', 0, 0)
                cv2.moveWindow('Predicted Image', window_width + 25, 0)
                cv2.moveWindow('Video Frame', (window_width + 25) * 2, 0)

                # Handle keyboard input
                key = wait_key(cv2, 0) & 0xFF

                if key == ord('q'):  # Quit visualization
                    destroy_windows(cv2)
                    video_cap.release()
                    return
                elif key == ord('j'):  # Previous frame
                    current_frame = (current_frame - 1) % total_frames
                elif key == ord('c'):  # Close current sequence and load next
                    destroy_windows(cv2)
                    break
                else:
                    # 'k' advances interactively; with no key press this also
                    # drives the bounded non-interactive loop forward.
                    current_frame = (current_frame + 1) % total_frames
            m += dilated_clip_len
        
        # Clean up
        video_cap.release()

    def visualise(self, mode):
        """
        Unified visualization method that can show either dataset samples or model predictions
        Args:
            mode: Either 'dataset' or 'predictions'
        """
        BATCH_SIZE = 1
        NUM_WORKERS = 1
        if mode == 'predictions':
            with open(self.test_loader, 'rb') as f:
                test_data = pickle.load(f)
            data_loader = DataLoader(
                test_data['dataset'],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS
            )
            
            # Initialize model
            model = SegmentationModel()
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        else:  # dataset mode
            full_dataset = MyDataset(
                data_dir=SYSTEM_PARAMS.files.dataset_root
            )
            train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
                full_dataset, train_size=1.0, val_size=0.0, test_size=0.0
            )
            data_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
            )

        data_iter = iter(data_loader)
        i = 0
        
        while True:  # Main loop for continuous data loading
            try:
                image, label = next(data_iter)
            except StopIteration:
                print("End of dataset reached. Restarting...")
                data_iter = iter(data_loader)
                continue

            if label.sum() == 0:
                continue
            # Handle predictions if in prediction mode
            if mode == 'predictions':
                with torch.no_grad():
                    image_input = image.to(device)
                    logits = model(image_input)
                    probs = torch.sigmoid(logits)
                    pred = (probs > DECISION_THRESHOLD).float()
                    pred = pred.cpu()

            # Convert tensors to numpy arrays
            image_seq = image.numpy().squeeze()  # Shape: (T, H, W)
            label_seq = label.numpy().squeeze()  # Shape: (T, H, W)
            if mode == 'predictions':
                pred_seq = pred.numpy().squeeze()  # Shape: (T, H, W)

            current_frame = 0
            total_frames = image_seq.shape[0]

            # Bounded when non-interactive; see the note on the browser above.
            frame_limit = iteration_limit("DIFFTACTILE_MAX_FRAMES", total_frames)
            shown = 0

            while frame_limit is None or shown < frame_limit:
                shown += 1
                # Prepare the current frame
                current_image = image_seq[current_frame]
                current_label = label_seq[current_frame]
                if mode == 'predictions':
                    current_pred = pred_seq[current_frame]
                    current_overlay = Visualisation.create_confusion_matrix_overlay(current_label, current_pred)
                    iou_score = Visualisation.calculate_iou(current_label, current_pred)
                
                # Normalize images for display
                current_image = (current_image * 255).astype(np.uint8)
                if mode == 'dataset':
                    current_right = (current_label * 255).astype(np.uint8)
                    
                    # Create binary versions (0 or 255)
                    binary_left = np.where(current_image > 0, 255, 0).astype(np.uint8)
                    binary_right = np.where(current_right > 0, 255, 0).astype(np.uint8)
                    
                    # Create RGB overlay
                    overlay_image = np.zeros((current_image.shape[0], current_image.shape[1], 3), dtype=np.uint8)
                    overlay_image[..., 0] = binary_left  # Red channel for markers
                    overlay_image[..., 1] = binary_right  # Green channel for ground truth
                else:  # predictions mode
                    # Convert overlay from float [0,1] to uint8 [0,255]
                    current_right = (current_overlay * 255).astype(np.uint8)

                # Scale up images by 4x using NEAREST neighbor interpolation
                scale_factor = 2
                current_image = cv2.resize(current_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                current_right = cv2.resize(current_right, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                if mode == 'dataset':
                    overlay_image = cv2.resize(overlay_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

                # Add frame counter text and other information
                frame_text = f"Frame: {current_frame + 1}/{total_frames}"
                if mode == 'predictions':
                    frame_text += f" | IoU: {iou_score:.3f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_color = (255, 255, 255)  # White text
                
                # Get text size to position it at the bottom
                text_size = cv2.getTextSize(frame_text, font, font_scale, font_thickness)[0]
                text_x = 10
                text_y = current_image.shape[0] - 20  # 20 pixels from bottom
                
                # Add black background for text visibility
                padding = 5
                cv2.rectangle(current_image, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                cv2.rectangle(current_right, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (0, 0, 0), -1)
                if mode == 'dataset':
                    cv2.rectangle(overlay_image, 
                                (text_x - padding, text_y - text_size[1] - padding),
                                (text_x + text_size[0] + padding, text_y + padding),
                                (0, 0, 0), -1)
                
                # Add text to images
                cv2.putText(current_image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                cv2.putText(current_right, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
                if mode == 'dataset':
                    cv2.putText(overlay_image, frame_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

                # Create display windows
                imshow(cv2, f'Input Image {i}', current_image)
                right_window_title = 'Ground Truth Label' if mode == 'dataset' else 'Prediction Overlay'
                imshow(cv2, f'{right_window_title} {i}', current_right)
                if mode == 'dataset':
                    imshow(cv2, f'Overlay {i}', overlay_image)

                # Get screen dimensions using cv2
                window_width = current_image.shape[0]
                
                # Position windows - left window at (0,0), middle at (window_width + 25, 0), right at (2 * window_width + 50, 0)
                cv2.moveWindow(f'Input Image {i}', 0, 0)
                cv2.moveWindow(f'{right_window_title} {i}', window_width + 25, 0)
                if mode == 'dataset':
                    cv2.moveWindow(f'Overlay {i}', 2 * window_width + 50, 0)

                # Handle keyboard input
                key = wait_key(cv2, 0) & 0xFF

                if key == ord('q'):  # Quit visualization
                    destroy_windows(cv2)
                    return
                elif key == ord('j'):  # Previous frame
                    current_frame = (current_frame - 1) % total_frames
                elif key == ord('c'):  # Close current sequence and load next
                    i += 1
                    destroy_windows(cv2)
                    break
                else:
                    # 'k' advances interactively; with no key press (the
                    # non-interactive case) this also steps the bounded loop on.
                    current_frame = (current_frame + 1) % total_frames
    
    def visualise_gnn(self, mode, data_source):
        """
        Visualize GNN predictions and ground truth segmentation masks.
        Shows images per frame:
        For mode='predictions':
            1. Ground truth labels (red = 0, green = 1) - only shown for central frame
            2. Hard predicted labels (red = 0, green = 1) - only shown for central frame
            3. Soft predicted labels (color intensity shows confidence) - only shown for central frame
            4. Original labels image - shown for all frames
            5. Graph connectivity visualization - shown for all frames
        For mode='dataset':
            1. Ground truth labels (red = 0, green = 1) - only shown for central frame
            2. Original labels image - shown for all frames
            3. Graph connectivity visualization - shown for all frames
        Args:
            mode: Either 'dataset' or 'predictions'
        """
        BATCH_SIZE = 1
        NUM_WORKERS = 1
        LABELS_DOWNSIZE = 4
        MARKER_SIZE = 10
        base_graph_data = np.load(SYSTEM_PARAMS.files.base_graph_connectivity)
        adjacency_matrix = base_graph_data['adjacency_matrix']

        if mode == 'predictions':
            # Initialize model. A named scenario dictates the architecture, since
            # a checkpoint only loads into the one it was trained with.
            if self.scenario_cfg is not None:
                model = _segmentation_gnn(self.scenario_cfg["arch"])
            else:
                model = GNN()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model.eval()
            model = model.to(device)

        # Load test data
        with open(self.test_loader, 'rb') as f:
            test_data = pickle.load(f)
        all_stats = test_data['dataset_stats']

        if data_source == 'pickled_test_dataset':
            dataset = test_data['dataset']
            dataset.eval()
            data_loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS
            )
        elif data_source == 'fresh_dataset':  # dataset mode
            # A named scenario selects the dataset (and its normalisation) to
            # match the checkpoint; otherwise fall through to the historical
            # hardcoded if True/if False toggle below.
            if self.scenario_cfg is not None:
                test_dataset = self._build_scenario_dataset(all_stats)
                # Kept so _MeatNavigator can recover the trial each clip came
                # from, and the frame range it covers.
                self.viewer_dataset = test_dataset
                data_loader = DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )
            elif True:
                full_dataset = MyDataset(
                    scheme="single_dataset",
                    sim_exp="exp",
                    data_dir=SYSTEM_PARAMS.files.exp_data_silicone,
                    apply_augmentations=False,
                )
            if False:
                full_dataset = MyDataset(
                    scheme="single_dataset",
                    sim_exp="sim",
                    data_dir=SYSTEM_PARAMS.files.sim_data,
                    apply_augmentations=True,
                )
            # Legacy path only: the scenario branch above has already built and
            # normalised its dataset, and would be overwritten by this block
            # (which also assumes a difficulty-keyed stats dict that the meat
            # loader does not have).
            if self.scenario_cfg is None:
                _, _, test_dataset = full_dataset.create_splits(
                    train_size=0.0,
                    val_size=0.0,
                    test_size=1.0
                )
                target_difficulty = 1.0
                stats = all_stats[target_difficulty]
                test_dataset.set_stats(stats)
                test_dataset.set_difficulty_level(target_difficulty)
                test_dataset.eval()
                data_loader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
                )
        elif data_source == 'exp_npz':
            exp_test_dataset_grid_search = MyDataset(
                mode='exp',
                exp_markers_npz=SYSTEM_PARAMS.files.experiment_og_markers_reordered_npz,
                exp_ground_truth_labels_npz=SYSTEM_PARAMS.files.experiment_og_ground_truth_labels_npz,
                exp_dilation=2,
                scheme='new',
            )
            exp_test_dataset_straight_line_slide = MyDataset(
                mode='exp',
                exp_markers_npz=SYSTEM_PARAMS.files.experiment_straight_markers_reordered_npz,
                exp_ground_truth_labels_npz=SYSTEM_PARAMS.files.experiment_straight_ground_truth_labels_npz,
                exp_dilation=2,
                scheme='new',
            )
            dataset = exp_test_dataset_grid_search
            target_difficulty = 1.0
            stats = all_stats[target_difficulty]
            dataset.set_stats(stats)
            dataset.set_difficulty_level(target_difficulty)
            data_loader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
            )

        data_iter = iter(data_loader)
        data_list = list(data_iter)
        gt = []
        sp = []
        hp = []
        co = []
        meta = []
        # Parallel to the five stacks above: (clip index, frame index in clip)
        # for every entry, which is what the trial/clip/frame navigation and the
        # metadata panel are built from.
        frame_index = []

        for sequence_idx in range(len(data_list)):  # Main loop for continuous data loading
            try:
                # batch, labels_images, poses, metadata, frame_ix = next(data_iter)
                batch, labels_images, poses, metadata, frame_ix = data_list[sequence_idx]
                poses = poses.numpy()[0]
                metadata = metadata.numpy()[0]
                frame_ix = frame_ix.item()
                # if not (
                #     metadata[1] == 0 
                #     and metadata[0] == 0 
                #     and frame_ix == 0
                # ):
                #     continue
            except StopIteration:
                print("End of dataset reached. Restarting...")
                data_iter = iter(data_loader)
                continue
            
            ground_truth_labels_present = labels_images.numel() != 0
            
            num_frames = SYSTEM_PARAMS.gnn.clip_len
            if ground_truth_labels_present:
                labels_images = labels_images.numpy()[0, ...]
                labels_h = labels_images.shape[1] // LABELS_DOWNSIZE
                labels_w = labels_images.shape[2] // LABELS_DOWNSIZE
            else:
                labels_h = 270
                labels_w = 480
                labels_images = np.zeros((num_frames, labels_h * LABELS_DOWNSIZE, labels_w * LABELS_DOWNSIZE), dtype=np.uint8)

            # Get number of frames from the mask
            num_nodes_per_frame = SYSTEM_PARAMS.vitactip.num_markers
            central_frame = num_frames // 2

            # Pre-compute image dimensions
            h, w = 400, 400
            MARKER_SIZE = 6
            MARKER_RADIUS = MARKER_SIZE // 2

            # Initialize image stacks for each color channel with white background
            metadata_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
            ground_truth_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
            prediction_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
            soft_prediction_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)  # New stack for soft predictions
            labels_stack = np.zeros((num_frames, labels_h, labels_w, 3), dtype=np.uint8)
            graph_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)  # New stack for graph visualization
            confusion_matrix_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)  # New stack for confusion matrix visualization
            stats_stack = np.zeros((num_frames, 200, 400, 3), dtype=np.uint8) + 255  # White background for stats display

            if mode == 'predictions':
                # Get predictions
                with torch.no_grad():
                    batch = batch.to(device)
                    x, x_mask, edge_index, edge_index_regular_nodes, edge_attr = model.my_prepare_data(batch, batch.num_graphs)
                    out = model(x, edge_index, edge_attr, batch.batch)
                    out = out.squeeze(-1)  # Remove the channel dimension
                    out = out[x_mask]
                    # mask = data.mask
                    # out = out[mask]
                    probs = torch.sigmoid(out)
                    # Display only: this cut drives the Hard Prediction and
                    # Confusion Matrix panels. It matches the vessel map's
                    # threshold so the two qualitative views agree with each
                    # other; the Soft Prediction panel beside them shows the
                    # underlying probabilities with no cut at all.
                    pred = (probs > MAP_DECISION_THRESHOLD).float()

                    assert batch.num_graphs == 1
                    
                    # Compute IoU scores per frame
                    num_nodes_per_frame = SYSTEM_PARAMS.vitactip.num_markers
                    clip_stats = []
                    for frame_idx in range(num_frames):
                        start_idx = frame_idx * num_nodes_per_frame
                        end_idx = (frame_idx + 1) * num_nodes_per_frame
                        frame_pred = pred[start_idx:end_idx]
                        frame_truth = batch.y[start_idx:end_idx]
                        frame_metrics = GNN.iou_score(frame_pred, frame_truth)
                        fg_iou = frame_metrics[1]
                        bg_iou = frame_metrics[0]
                        
                        # Compute confusion matrix
                        frame_pred = pred[start_idx:end_idx].cpu().numpy()
                        frame_truth = batch.y[start_idx:end_idx].cpu().numpy()
                        tp = np.sum((frame_pred == 1) & (frame_truth == 1))
                        tn = np.sum((frame_pred == 0) & (frame_truth == 0))
                        fp = np.sum((frame_pred == 1) & (frame_truth == 0))
                        fn = np.sum((frame_pred == 0) & (frame_truth == 1))
                        clip_stats.append({
                            'fg_iou': fg_iou,
                            'bg_iou': bg_iou,
                            'tp': tp,
                            'tn': tn,
                            'fp': fp,
                            'fn': fn,
                        })
                    
                    probs = probs.cpu().numpy().astype(np.float32)
                    pred = pred.cpu().numpy().astype(int)

            # if data.y.cpu().numpy().sum() == 0 or pred.sum() == 0:
            #     continue

            # Pre-compute all frames
            for frame_idx in range(num_frames):
                # Get marker positions for current frame
                start_idx = frame_idx * num_nodes_per_frame
                end_idx = (frame_idx + 1) * num_nodes_per_frame
                frame_points = batch.pos[start_idx:end_idx].cpu().numpy()[:, :2]
                
                # Transform from (-1,1) to (0,200) range
                points = (frame_points + 3) / 6 * w  # Now in range (0,200)
                points = points.astype(np.float32)  # Keep as float for draw_point
                
                # Create graph connectivity visualization
                # _, points, adjacency_matrix = Adjacency.get_graph_connectivity(points)
                graph_img = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Draw edges from adjacency matrix in green
                for edge in adjacency_matrix:
                    start_idx, end_idx = edge
                    start_point = tuple(map(int, points[start_idx]))
                    end_point = tuple(map(int, points[end_idx]))
                    cv2.line(graph_img, start_point, end_point, color=(0, 255, 0), thickness=1)
                
                # Draw points in red
                for point in points:
                    x, y = map(int, point)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(graph_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                
                graph_stack[frame_idx] = graph_img

                # Get predictions for current frame
                start_idx = frame_idx * num_nodes_per_frame
                end_idx = (frame_idx + 1) * num_nodes_per_frame
                ground_truth = batch.y[start_idx:end_idx].cpu().numpy()
                
                # Draw markers on ground truth image
                for point_idx, point in enumerate(points):
                    if 0 <= point[0] < w and 0 <= point[1] < h:
                        center = (int(point[0]), int(point[1]))
                        if ground_truth[point_idx] == 1:
                            # Magenta (BGR = (255, 0, 255)) for positive class
                            cv2.circle(ground_truth_stack[frame_idx], center, MARKER_RADIUS, (255, 0, 255), -1, cv2.LINE_AA)
                        else:
                            # Cyan (BGR = (255, 255, 0)) for negative class
                            cv2.circle(ground_truth_stack[frame_idx], center, MARKER_RADIUS, (255, 255, 0), -1, cv2.LINE_AA)
                
                if mode == 'predictions':
                    frame_pred = pred[start_idx:end_idx]
                    frame_probs = probs[start_idx:end_idx]
                    
                    # Draw markers on prediction image (hard predictions)
                    for point_idx, point in enumerate(points):
                        if 0 <= point[0] < w and 0 <= point[1] < h:
                            center = (int(point[0]), int(point[1]))
                            if frame_pred[point_idx] == 1:
                                # Magenta (BGR = (255, 0, 255)) for positive class
                                cv2.circle(prediction_stack[frame_idx], center, MARKER_RADIUS, (255, 0, 255), -1, cv2.LINE_AA)
                            else:
                                # Cyan (BGR = (255, 255, 0)) for negative class
                                cv2.circle(prediction_stack[frame_idx], center, MARKER_RADIUS, (255, 255, 0), -1, cv2.LINE_AA)
                    
                    # Draw markers on soft prediction image
                    for point_idx, point in enumerate(points):
                        if 0 <= point[0] < w and 0 <= point[1] < h:
                            center = (int(point[0]), int(point[1]))
                            prob = frame_probs[point_idx]
                            intensity = int(255 * prob)  # Scale to [0,255]
                            # Use white color with varying intensity for all points
                            cv2.circle(soft_prediction_stack[frame_idx], center, MARKER_RADIUS, (intensity, intensity, intensity), -1, cv2.LINE_AA)
                    
                    # Draw confusion matrix visualization
                    for point_idx, point in enumerate(points):
                        if 0 <= point[0] < w and 0 <= point[1] < h:
                            center = (int(point[0]), int(point[1]))
                            pred_val = frame_pred[point_idx]
                            true_val = ground_truth[point_idx]
                            
                            # The project-wide confusion scheme
                            # (CONFUSION_COLOURS_RGB, shared with the vessel
                            # map): green = both say vessel, RED = a MISS
                            # (truth says vessel, prediction does not), BLUE =
                            # a FALSE ALARM. True negatives are dim grey rather
                            # than the map's black, since a black dot on this
                            # black panel would be invisible. BGR for cv2.
                            if pred_val == 1 and true_val == 1:  # TP
                                color = (0, 255, 0)
                            elif pred_val == 0 and true_val == 0:  # TN
                                color = (90, 90, 90)
                            elif pred_val == 1 and true_val == 0:  # FP (false alarm)
                                color = (255, 0, 0)
                            else:  # FN (miss)
                                color = (0, 0, 255)
                            
                            cv2.circle(confusion_matrix_stack[frame_idx], center, MARKER_RADIUS, color, -1, cv2.LINE_AA)
                    
                    # Draw statistics for current frame
                    stats_img = stats_stack[frame_idx]
                    frame_stats = clip_stats[frame_idx]
                    
                    # Define text positions and font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    line_spacing = 30
                    x_pos = 20
                    y_pos = 40
                    
                    # Draw statistics text
                    cv2.putText(stats_img, f"Foreground IoU: {frame_stats['fg_iou']:.3f}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"Background IoU: {frame_stats['bg_iou']:.3f}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"True Positives: {frame_stats['tp']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"True Negatives: {frame_stats['tn']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"False Positives: {frame_stats['fp']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)
                    y_pos += line_spacing
                    
                    cv2.putText(stats_img, f"False Negatives: {frame_stats['fn']}", (x_pos, y_pos), 
                              font, font_scale, (0, 0, 0), thickness)

                # Get and process labels image for current frame
                if ground_truth_labels_present:
                    labels_image = labels_images[frame_idx]
                    # Convert labels_image to BGR for visualization
                    labels_display = np.zeros((labels_image.shape[0], labels_image.shape[1], 3), dtype=np.uint8)
                    # Convert torch tensor to numpy and scale back to [0, 255]
                    labels_np = (labels_image * 255).astype(np.uint8)
                    labels_display[..., 0] = labels_np  # Set blue channel
                    labels_display[..., 1] = labels_np  # Set green channel
                    labels_display[..., 2] = labels_np  # Set red channel

                    # Downscale by factor of 4 using INTER_AREA interpolation
                    labels_stack[frame_idx] = cv2.resize(labels_display, (labels_w, labels_h), interpolation=cv2.INTER_AREA)
                else:
                    # Keep the black image for labels_stack when no ground truth is present
                    pass

            # Keep EVERY frame of the clip, not just its centre.
            #
            # The old code kept only `clip_len // 2` and threw the rest away,
            # which is why the viewer could only step clip-to-clip. Retaining
            # all of them is what makes "frame within the clip" (the n/m keys) a
            # real axis. The panels are already rendered for every frame just
            # above, so this costs nothing but the references.
            #
            # The metadata panel is left BLANK here and drawn at display time
            # instead: its text depends on the frame you are looking at, which
            # is not known until you navigate there.
            for frame_in_clip in range(num_frames):
                gt.append(ground_truth_stack[frame_in_clip])
                sp.append(soft_prediction_stack[frame_in_clip])
                hp.append(prediction_stack[frame_in_clip])
                co.append(confusion_matrix_stack[frame_in_clip])
                meta.append(metadata_stack[frame_in_clip])
                # (clip index, frame index within that clip) for each entry, so
                # the navigation and the metadata panel can recover both.
                frame_index.append((sequence_idx, frame_in_clip))

        # NOTE: the old `filter_left()` decimation is gone. It kept 10 of every
        # 20 clips on the assumption of the simulated dataset's "5 trajectories
        # x 2 directions x 10 frames" layout, which does not describe dataset C
        # at all - on meat it silently hid half the clips of every trial. The
        # viewer now shows every clip the loader produced.

        # --- display: one Qt window, stepped by hand ---------------------------
        #
        # This used to be five OpenCV windows auto-playing on timed wait_key()
        # delays, which had two problems. The delays meant the frame you wanted
        # to look at had already gone by the time you registered it, and the
        # five windows were placed with cv2.moveWindow(), which does nothing on
        # Wayland - a Wayland client cannot position itself, by design. So the
        # panels landed wherever the compositor put them, overlapping.
        #
        # Both are fixed by compositing the panels into a single image and
        # handing that to the shared Qt browser: one window the compositor is
        # free to place, laid out deterministically by us, advancing only when
        # the user presses a key. Same FrameBrowser the annotation viewers use,
        # so this is a native Wayland client for the same reason they are.
        if not meta:
            print("Nothing to display: the loader produced no clips.")
            return

        # Navigation levels depend on the frames mode: three for "all" (trial,
        # clip within trial, frame within clip) and two for "central" (trial,
        # central frame within trial), with one key pair per level and no key
        # doing two jobs. The navigator owns all the index arithmetic; the
        # callbacks below only translate keys into its moves.
        viewer_dataset = getattr(self, "viewer_dataset", None)
        if self.frames == "central":
            nav = _CentralFrameNavigator(frame_index, viewer_dataset)
            keys_help = "i/o trial   j/k central frame   q quit"
        else:
            nav = _MeatNavigator(frame_index, viewer_dataset)
            keys_help = "i/o trial   j/k clip   n/m frame   q quit"
        panels_for = self._prediction_panels(mode, gt, hp, co, sp, meta)

        def render():
            panels = panels_for(nav.position)
            # The Metadata panel is drawn here rather than baked in during
            # collection, because its text depends on where you have navigated
            # to. `panels` is rebuilt each call, so this never accumulates.
            return self._compose_panels(
                [(c, self._metadata_panel(img, nav.metadata_lines())
                     if c == "Metadata" else img)
                 for c, img in panels]
            )

        def status():
            # Deliberately no frame counter here: all the positional detail now
            # lives in the Metadata panel, and duplicating it made the two
            # disagree about what "frame" meant (clip-index vs frame-in-clip).
            return [
                f"[{self.scenario} {self.weights_label} {self.frames}]",
                keys_help,
            ]

        def on_key(key):
            if key == ord('q'):
                return "quit"
            moved = nav.handle_key(key)
            return "redraw" if moved else None

        # Imported lazily, exactly as the annotation viewers do it, for two
        # separate reasons - do NOT hoist this to module scope:
        #   1. It keeps this module importable by the non-GUI analysis paths on
        #      an interpreter that has no PySide6.
        #   2. On Python 3.10, importing PySide6 BEFORE torch breaks torch:
        #      PySide6 ships a typing_extensions that shadows the one torch
        #      needs, and `import torch._dynamo` then dies with
        #      "TypeError: Plain typing.Self is not valid as type argument".
        #      Importing here means torch is always in first, which is fine.
        from difftactile.main.qt_viewer import run_browser

        run_browser(
            title=f"Predictions - {self.scenario} ({self.weights_label}, {self.frames} frames)",
            render=render,
            on_key=on_key,
            status=status,
            # Record mode only (DIFFTACTILE_RECORD_MP4): the key script that
            # walks every trial and frame. Ignored when driven by hand.
            auto_keys=nav.walk_keys(),
        )

    @staticmethod
    def _metadata_panel(base, lines):
        """Draw the metadata lines onto a copy of the (blank) metadata frame.

        A copy, because the underlying array is one slice of a per-clip stack
        that gets revisited every time you navigate back to this frame - drawing
        in place would stack text on text until it was unreadable.
        """
        img = base.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 40
        for line in lines:
            cv2.putText(img, line, (12, y), font, 0.62, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y += 34
        return img

    @staticmethod
    def _prediction_panels(mode, gt, hp, co, sp, meta):
        """Return `idx -> [(caption, image), ...]`, the panels for one frame.

        In `predictions` mode all five are shown; otherwise only the ground
        truth, which is what the old five-window code did via its `if mode ==
        'predictions'` guards.
        """
        def panels(idx):
            out = [("Ground Truth", gt[idx])]
            if mode == 'predictions':
                out += [
                    ("Hard Prediction", hp[idx]),
                    ("Confusion Matrix", co[idx]),
                    ("Soft Prediction", sp[idx]),
                    ("Metadata", meta[idx]),
                ]
            return out
        return panels

    @staticmethod
    def _compose_panels(panels, columns=3, pad=12, caption_h=34):
        """Tile captioned panels into one BGR image on a neutral background.

        Replaces the cv2.moveWindow() grid, which silently did nothing under
        Wayland. Doing the layout ourselves means the arrangement is identical
        on every backend and the whole thing is one window to manage.

        Panels are padded (never scaled) into a uniform cell, so pixels stay
        1:1 with the arrays and nothing is resampled on the way to the screen.
        """
        cells = []
        for caption, img in panels:
            if img is None:
                continue
            if img.ndim == 2:                      # grayscale -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cells.append((caption, img))
        if not cells:
            return np.zeros((120, 480, 3), dtype=np.uint8)

        cell_w = max(img.shape[1] for _, img in cells)
        cell_h = max(img.shape[0] for _, img in cells)
        rows = (len(cells) + columns - 1) // columns

        sheet_w = columns * cell_w + (columns + 1) * pad
        sheet_h = rows * (cell_h + caption_h) + (rows + 1) * pad
        # Mid-grey: keeps both the black metadata panel and the bright overlays
        # legible against it, in either desktop theme.
        sheet = np.full((sheet_h, sheet_w, 3), 40, dtype=np.uint8)

        for i, (caption, img) in enumerate(cells):
            r, c = divmod(i, columns)
            x0 = pad + c * (cell_w + pad)
            y0 = pad + r * (cell_h + caption_h + pad)
            cv2.putText(sheet, caption, (x0, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1,
                        cv2.LINE_AA)
            # Top-left within the cell rather than centred: the panels are all
            # the same size in practice, and this keeps them aligned when one
            # of them is not.
            h_i, w_i = img.shape[:2]
            y1 = y0 + caption_h
            sheet[y1:y1 + h_i, x0:x0 + w_i] = img
        return sheet

    def test_data_loader(self):
        BATCH_SIZE = 16
        NUM_WORKERS = 16
        full_dataset = MyDataset(
            data_dir=SYSTEM_PARAMS.files.dataset_root
        )
        train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
            full_dataset, train_size=1.0, val_size=0.0, test_size=0.0
        )
        data_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )

        data_iter = iter(data_loader)
        i = 0
        pbar = tqdm()
        while True:
            try:
                image, label = next(data_iter)
            except StopIteration:
                print("End of dataset reached. Restarting...")
                break
            i += 1
            pbar.update(1)
            pbar.set_description(f"Processed {i} batches")
        pbar.close()
    
    def graph(self):
        full_dataset = MyDataset(
            data_dir=SYSTEM_PARAMS.files.dataset_root
        )
        train_dataset, val_dataset, test_dataset = MyDataset.create_splits(
            full_dataset, train_size=0.1, val_size=0.0, test_size=0.0
        )
        num_clips = len(train_dataset)
        visualise = False
        
        # Lists to store features from all frames
        all_node_features = []
        all_edge_features = []
        
        for i in range(num_clips):
            points = train_dataset.get_markers(i)
            adjacency = Visualisation.compute_knn_adjacency(points)
            
            # Compute node and edge features
            node_features, edge_features = Visualisation.compute_graph_features(points, adjacency)
            
            # Store features
            all_node_features.append(node_features)
            all_edge_features.append(edge_features)
            
            if visualise:
                should_break = Visualisation.visualise_adjacency_graph(points, adjacency)
                if should_break:
                    break
        
        plt.close('all')  # Close any remaining figures
        return all_node_features, all_edge_features

    @staticmethod
    def visualise_adjacency_graph(points, adjacency):
        # Create a new figure for each frame
        plt.figure(figsize=(10, 10))
        
        # Plot edges first (connections between points)
        for node_idx in range(len(points)):
            # Get the neighbors for this node
            neighbors = adjacency[node_idx]
            # Draw lines from this node to all its neighbors
            for neighbor_idx in neighbors:
                plt.plot([points[node_idx, 0], points[neighbor_idx, 0]],
                        [points[node_idx, 1], points[neighbor_idx, 1]],
                        'gray', alpha=0.5, linewidth=1)
        
        # Plot nodes (points)
        plt.scatter(points[:, 0], points[:, 1], c='red', s=50)
        
        plt.title(f'Frame {i+1}: K-Nearest Neighbors Graph (k=6)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # Make the plot aspect ratio equal
        plt.axis('equal')
        
        # Display the plot
        plt.draw()
        plt.pause(0.1)  # Add a small pause to allow for visualization

        # Wait for key press to continue. prompt() returns "" immediately unless
        # DIFFTACTILE_INTERACTIVE=1, so an unattended run is never stuck here.
        key = prompt("Press Enter to continue to next frame, or 'q' to quit: ")
        if key.lower() == 'q':
            return True
        
        plt.close()  # Close the current figure before showing the next one
        return False


def main():
    """Open the interactive per-frame prediction viewer.

    Pass one of the three configurations to select the checkpoint and the test
    dataset together:

        python -m difftactile.scripts.script_visualise A-to-B
        python -m difftactile.scripts.script_visualise A-to-C --all --retrained

    --central  (default) only each window's central frame (trial / frame) -
               the prediction the reported metrics are computed from.
    --all      every frame of every sliding window (trial / clip / frame),
               off-centre predictions included. A debugging view.

    The default is the reported view on purpose: if the two ever disagree about
    how good the model looks, that is the one that should be seen first.

    --trials SPEC  restrict the test set to some trials: a comma-separated list
               of trial-id substrings, or `first-vessel-present`
               (see `Visualisation._select_trials()`).

    The configuration may also come from DIFFTACTILE_SCENARIO, the weight source
    from DIFFTACTILE_WEIGHTS (pretrained | retrained), the frames mode from
    DIFFTACTILE_FRAMES (all | central) and the trial selection from
    DIFFTACTILE_VIEW_TRIALS.
    """
    argv = sys.argv[1:]
    # --sweep, --seed and --trials take a value, so they are pulled out before
    # the rest is split into flags and positionals.
    sweep_dir = os.environ.get("DIFFTACTILE_SWEEP_DIR")
    sweep_seed = int(os.environ.get("DIFFTACTILE_SWEEP_SEED", 0))
    trials = os.environ.get("DIFFTACTILE_VIEW_TRIALS")
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--sweep" and i + 1 < len(argv):
            sweep_dir = argv[i + 1]; i += 2
        elif argv[i].startswith("--sweep="):
            sweep_dir = argv[i].split("=", 1)[1]; i += 1
        elif argv[i] == "--trials" and i + 1 < len(argv):
            trials = argv[i + 1]; i += 2
        elif argv[i].startswith("--trials="):
            trials = argv[i].split("=", 1)[1]; i += 1
        elif argv[i] == "--seed" and i + 1 < len(argv):
            sweep_seed = int(argv[i + 1]); i += 2
        elif argv[i].startswith("--seed="):
            sweep_seed = int(argv[i].split("=", 1)[1]); i += 1
        else:
            rest.append(argv[i]); i += 1
    argv = rest

    flags = [a for a in argv if a.startswith("--")]
    positional = [a for a in argv if not a.startswith("--")]

    scenario = positional[0] if positional else os.environ.get("DIFFTACTILE_SCENARIO")
    if "--retrained" in flags:
        weights = "retrained"
    elif "--pretrained" in flags:
        weights = "pretrained"
    elif "--legacy" in flags:
        weights = "legacy"
    elif "--best" in flags:
        weights = "best"
    else:
        # Default: the best-of-N seed instance from the published sweep.
        weights = os.environ.get("DIFFTACTILE_WEIGHTS", "best")

    # Asking for both is contradictory, so reject it rather than letting one
    # silently win.
    if "--all" in flags and "--central" in flags:
        raise SystemExit(
            "ERROR: --all and --central are mutually exclusive; pass exactly one."
        )
    if "--all" in flags:
        frames = "all"
    elif "--central" in flags:
        frames = "central"
    else:
        # Defaults to the frames the paper reports, so the unqualified view is
        # the honest one; --all is the opt-in debugging view.
        frames = os.environ.get("DIFFTACTILE_FRAMES", "central")

    v = Visualisation(scenario=scenario, weights=weights, frames=frames,
                      sweep_dir=sweep_dir, sweep_seed=sweep_seed, trials=trials)
    v.visualise_gnn(
        mode='predictions',
        data_source='fresh_dataset'
    )


if __name__ == "__main__":
    main()
