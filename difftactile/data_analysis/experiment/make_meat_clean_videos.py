"""Build the per-trial meat videos that ship in the data bundle.

The raw meat recordings are 1920x1080 MJPEG at 30 fps, ~429 frames and ~70 MB
per trial (1.6 GB for all 23) - far too large to publish. Preprocessing keeps
only every 15th frame *that has a valid interpolated robot pose*, which is 26
frames per trial, and everything downstream (marker positions, labels, the
annotation viewer) is indexed against exactly those frames.

This script writes that 26-frame subset as a compressed H.264 video into
`meat_training_data/clean/<trial_id>/frames.mp4`, alongside the
`marker_positions.npz` / `marker_labels.npz` already there. At CRF 26 a trial is
~2.5 MB, so all 23 add ~57 MB to the bundle instead of 1.6 GB, while staying
frame-for-frame aligned with the labels.

Frame selection deliberately reuses `MeatPreprocessData._decimate_video_and_poses()`
rather than reimplementing the stride, because the pose-validity mask - not just
the stride - decides which frames survive. Reimplementing it would risk an
off-by-one against the labels.

    python -m difftactile.scripts.script_make_meat_clean_videos

Requires the raw recordings in `meat_training_data/raw/`; they are the author-side
input to the bundle, not part of it.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np

from difftactile.data_analysis.experiment.preprocess_meat_data import MeatPreprocessData

# Constant quality rather than a bitrate: these are 26 near-still frames, so CRF
# keeps the marker dots and the straw ridge crisp while collapsing the large
# black border around the sensor.
H264_CRF = 26


def _encode(src, dst):
    """Re-encode `src` to H.264 at CRF, preserving every frame."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "libx264",
            "-crf", str(H264_CRF),
            "-pix_fmt", "yuv420p",
            str(dst),
        ],
        check=True,
    )


def main():
    processor = MeatPreprocessData()
    raw_dir = processor.input_dir
    if not raw_dir.is_dir():
        raise SystemExit(
            f"No raw meat recordings at {raw_dir}.\n"
            "These are the author-side input to the bundle and are not shipped in it."
        )

    avi_paths = sorted(raw_dir.glob("*.avi"))
    if not avi_paths:
        raise SystemExit(f"No .avi files in {raw_dir}.")

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH; it is needed to compress the frames.")

    total_bytes = 0
    for avi_path in avi_paths:
        trial_id = avi_path.stem
        pose_path = raw_dir / f"{trial_id}.npz"
        if not pose_path.exists():
            print(f"[{trial_id}] no pose .npz alongside the video; skipping.")
            continue

        # The raw recordings are named by bare timestamp but the clean/
        # directories carry a descriptive prefix, so resolve rather than
        # concatenate. `_trial_out_dir()` matches on the timestamp the two share.
        out_dir = processor._trial_out_dir(trial_id)
        if not out_dir.is_dir():
            print(f"[{trial_id}] no clean/ directory; skipping (run preprocessing first).")
            continue

        # Same frame selection the labels were built from.
        decimated_video, decimated_poses = processor._decimate_video_and_poses(
            avi_path, pose_path, trial_id
        )

        # Cross-check against the labels, so a mismatch is caught here rather
        # than showing up as a silent off-by-one in the viewer.
        labels_path = out_dir / "marker_labels.npz"
        if labels_path.exists():
            with np.load(labels_path) as d:
                n_labels = d["marker_labels"].shape[0]
            if n_labels != len(decimated_poses):
                print(
                    f"[{trial_id}] WARNING: {len(decimated_poses)} decimated frames "
                    f"but {n_labels} label frames; not writing a video for this trial."
                )
                continue

        out_path = out_dir / "frames.mp4"
        _encode(decimated_video, out_path)
        size = out_path.stat().st_size

        # Ship the poses for exactly these frames too. Without them, re-running
        # preprocessing would still need the 1.6 GB raw archive purely to
        # recover the robot poses, since they live in the raw .npz at full frame
        # rate and are interpolated during decimation.
        np.savez(out_dir / "frames_poses.npz", poses=decimated_poses)
        size += (out_dir / "frames_poses.npz").stat().st_size

        total_bytes += size
        print(f"[{trial_id}] {len(decimated_poses)} frames -> {out_path.name} ({size/1e6:.2f} MB)")

    print(f"\nTotal: {total_bytes/1e6:.1f} MB across {len(avi_paths)} trials.")
