"""Pick, per Meat trial, the Sim->Meat video frame whose confusion mix is most typical.

The poster's workflow diagram shows ONE per-frame confusion overlay of the
Sim->Meat model. To avoid cherry-picking, the frame is chosen by a fixed rule:

  * the reference is the model's pooled confusion over its whole Meat test set
    (Table 4 of the manuscript, threshold 0.5): TP, FP, FN, TN as percentages
    of all scored markers (they sum to 100 %);
  * for every central frame of every one of the 10 Meat trials the same four
    percentages are computed over that frame's 127 markers;
  * the frame's loss is the L2 distance between its percentage vector and the
    reference vector, and the frame with the smallest loss in each trial is
    kept (10 candidates, one per trial; the user picks one by eye).

Inputs: results/frame_space_predictions_A-to-C.npz (written by
scripts/frame_space_predictions.py; verified to reproduce Table 4).

Outputs, in poster-workflow-diagram-per-frame-predictions-sim-to-meat/<timestamp>/:
  trial_NN_<name>_frame_FF_confusion.png            overlay on black (as the video)
  trial_NN_<name>_frame_FF_confusion_on_frame.png   overlay drawn over the real tactile image
  best_frames.md                                    the loss of every chosen frame + the rule
Colours follow the project's confusion scheme: green = TP, red = FN (missed
vessel), blue = FP (false alarm), grey = TN.
"""

import datetime
import os
import subprocess

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ANALYSIS, REPO, RESULTS  # noqa: E402

MEAT = os.path.join(REPO, "difftactile/manual_or_experimental_data/meat_training_data/clean")
PRED = os.path.join(RESULTS, "frame_space_predictions_A-to-C.npz")
OUT_ROOT = os.path.join(ANALYSIS, "overlays", "sim_to_meat_frame_choice")
THRESHOLD = 0.5                      # the table's threshold (the video used 0.58)
COLOURS = {"tp": (0, 255, 0), "fn": (255, 0, 0), "fp": (0, 0, 255), "tn": (90, 90, 90)}  # RGB


def confusion(pred, lab):
    return np.array([(pred & lab).sum(), (pred & ~lab).sum(), (~pred & lab).sum(), (~pred & ~lab).sum()], float)


def draw_overlay(canvas, pos, pred, lab, radius):
    for (x, y), p, l in zip(pos, pred, lab):
        key = "tp" if (p and l) else "fn" if l else "fp" if p else "tn"
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, COLOURS[key], -1, cv2.LINE_AA)
    return canvas


def real_frame(trial, frame):
    """Frame `frame` of the trial's frames.mp4 (1:1 with the marker arrays)."""
    cap = cv2.VideoCapture(os.path.join(MEAT, trial, "frames.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame} of {trial}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    d = np.load(PRED, allow_pickle=True)
    probs, labels, pos, trials, frames = d["probs"], d["labels"].astype(bool), d["pos"], d["trial"], d["frame"]
    pred = probs >= THRESHOLD
    ref_counts = confusion(pred, labels)
    ref = 100.0 * ref_counts / ref_counts.sum()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(OUT_ROOT, stamp)
    os.makedirs(out)
    rows = []
    for k, trial in enumerate(sorted(set(trials))):
        ix = np.where(trials == trial)[0]
        losses = []
        for i in ix:
            c = confusion(pred[i], labels[i])
            pct = 100.0 * c / c.sum()
            losses.append(float(np.linalg.norm(pct - ref)))
        best = ix[int(np.argmin(losses))]
        loss = min(losses)
        c = confusion(pred[best], labels[best]).astype(int)
        frame = int(frames[best])
        stem = f"trial_{k + 1:02d}_{trial}_frame_{frame:02d}"
        # Overlay on black, cropped to the marker cloud (like the viewer's panel).
        p = pos[best]
        x0, y0 = p.min(axis=0) - 60
        x1, y1 = p.max(axis=0) + 60
        black = np.zeros((int(y1 - y0), int(x1 - x0), 3), np.uint8)
        draw_overlay(black, p - [x0, y0], pred[best], labels[best], radius=14)
        Image.fromarray(black).save(os.path.join(out, stem + "_confusion.png"))
        # The same overlay on the real tactile image of that frame.
        img = real_frame(trial, frame).copy()
        draw_overlay(img, p, pred[best], labels[best], radius=12)
        img = img[int(max(y0, 0)):int(y1), int(max(x0, 0)):int(x1)]
        Image.fromarray(img).save(os.path.join(out, stem + "_confusion_on_frame.png"))
        rows.append((k + 1, trial, frame, len(ix), c, loss))
        print(f"{stem}: TP {c[0]} FP {c[1]} FN {c[2]} TN {c[3]}  loss {loss:.2f}")

    lines = [
        "# Representative Sim→Meat video frames (one per Meat trial)", "",
        f"Model: {str(d['model'])}. Predictions thresholded at {THRESHOLD} (the threshold of the",
        "manuscript's Table 4; the project-page video draws its confusion panel at 0.58).", "",
        "Reference = pooled confusion of the whole Meat test set (Table 4, video-frame space):",
        f"TP {int(ref_counts[0])}, FP {int(ref_counts[1])}, FN {int(ref_counts[2])}, TN {int(ref_counts[3])} "
        f"= **{ref[0]:.2f} / {ref[1]:.2f} / {ref[2]:.2f} / {ref[3]:.2f} %**.", "",
        "For every central frame of every trial the same four percentages are computed over its",
        "127 markers; loss = L2 distance (in percentage points) between the frame's vector and the",
        "reference; the lowest-loss frame of each trial is listed. `frame` is the video frame index",
        "within the trial's 26 frames (frames.mp4 / marker_*.npz row).", "",
        "| # | trial | frame | central frames | TP | FP | FN | TN | TP/FP/FN/TN % | loss |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, trial, frame, n, c, loss in rows:
        pct = 100.0 * c / c.sum()
        lines.append(f"| {k} | {trial} | {frame} | {n} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | "
                     f"{pct[0]:.1f} / {pct[1]:.1f} / {pct[2]:.1f} / {pct[3]:.1f} | **{loss:.2f}** |")
    lines += ["", "Files: `<trial>_frame_FF_confusion.png` (black background, as in the project-page video) and",
              "`..._confusion_on_frame.png` (same markers drawn over the real tactile image).",
              "Colours: green = TP, red = FN (missed vessel), blue = FP (false alarm), grey = TN.", ""]
    with open(os.path.join(out, "best_frames.md"), "w") as f:
        f.write("\n".join(lines))
    print("written", out)


if __name__ == "__main__":
    main()
