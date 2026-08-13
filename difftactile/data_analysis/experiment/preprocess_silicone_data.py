import copy
import os
import glob
import math
import numpy as np
import cv2
import pickle
from scipy.interpolate import interp1d
import tqdm
import time
import shutil
from scipy.spatial.distance import pdist, squareform

# NOTE: `predict_exp` is deliberately NOT imported here. It pulls in torch,
# torch-geometric and the rest of the GNN stack, none of which the interactive
# annotator (`annotate()`, at the bottom of this file) touches - it only decodes
# video and draws circles. Keeping the import inside the two methods that
# actually call PredictExp lets the annotator run in the small
# `vessel-palpation-annotator` env (requirements/annotator-env.yml), which has
# no torch installed. `math` used to arrive via that star-import; it is now
# imported directly above.
from difftactile.main.constants import *
from difftactile.main.display import (
    destroy_windows, fit_to_view, imshow, is_interactive, iteration_limit,
    run_frame_browser, wait_key,
)


class SiliconePreprocessData:
    def __init__(self):
        self.root = "difftactile/manual_or_experimental_data/silicone_training_data"
        self.dir = "20250901-131547"
        self.input_dir = os.path.join(self.root, self.dir)
        self.output_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        os.makedirs(self.output_dir, exist_ok=True)
        self.og_theta = SYSTEM_PARAMS.geometry.camera_rotation_angle
        self.cx = SYSTEM_PARAMS.fisheye_model.circle_centre_x
        self.cy = SYSTEM_PARAMS.fisheye_model.circle_centre_y
        self.r = SYSTEM_PARAMS.fisheye_model.circle_radius
        th = np.zeros(shape=(10,), dtype=float)
        th[0] = 0
        th[1] = 0
        th[2] = 0
        th[3] = 0
        th[4] = -5
        th[5] = -5
        th[6] = -5
        th[7] = -5
        th[8] = 0
        th[9] = 0
        self.thetas = th

    def _get_file_pairs(self, dir):
        avi_files = sorted(glob.glob(os.path.join(dir, "*.avi")))
        npz_files = sorted(glob.glob(os.path.join(dir, "*.npz")))
        return avi_files, npz_files

    def _get_file_pairs_2(self, dir):
        avi_files = sorted(glob.glob(os.path.join(dir, "*.avi")))
        npz_files_poses = sorted(glob.glob(os.path.join(dir, "*_poses.npz")))
        npz_files_markers = sorted(glob.glob(os.path.join(dir, "*_markers.npz")))
        return avi_files, npz_files_poses, npz_files_markers

    def interpolate_metadata_and_trim_videos(self):
        avi_files, npz_files = self._get_file_pairs(self.input_dir)
        for avi_path, npz_path in zip(avi_files, npz_files):
            npz_data = np.load(npz_path)
            output = npz_data["output"]
            metadata = npz_data["metadata"]
            vy = metadata[3]
            frame_numbers = output[:, 0].astype(int)
            poses = output[:, 1:]
            unique_frames, unique_idx = np.unique(frame_numbers, return_index=True)
            poses_unique = poses[unique_idx]
            interp_funcs = [
                interp1d(
                    unique_frames, poses_unique[:, d], kind="linear", bounds_error=False
                )
                for d in range(poses_unique.shape[1])
            ]
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            all_frames = np.arange(n_frames)
            interp_poses = np.stack([f(all_frames) for f in interp_funcs], axis=1)
            valid_mask = (all_frames >= unique_frames.min()) & (
                all_frames <= unique_frames.max()
            )
            interp_poses = interp_poses[valid_mask]
            valid_frames = all_frames[valid_mask]
            y_range_low, y_range_high = vy - 100 + 20, vy - 20
            y = interp_poses[:, 1]
            y_mask = (y >= y_range_low) & (y <= y_range_high)
            interp_poses = interp_poses[y_mask]
            valid_frames = valid_frames[y_mask]
            output_npz = os.path.join(self.output_dir, os.path.basename(npz_path))
            np.savez(
                output_npz,
                output=interp_poses,
                metadata=metadata,
            )
            output_avi = os.path.join(self.output_dir, os.path.basename(avi_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_avi, fourcc, fps, (w, h))
            frame_idx = 0
            kept_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in valid_frames:
                    writer.write(frame)
                    kept_idx += 1
                frame_idx += 1
            cap.release()
            writer.release()

    def apply_dilation(self, k):
        input_dir = os.path.join(self.root, f"{self.dir}_interpolated_trimmed")
        output_dir = os.path.join(self.root, f"{self.dir}_dilated")
        os.makedirs(output_dir, exist_ok=True)
        avi_files, npz_files = self._get_file_pairs(input_dir)
        for avi_path, npz_path in zip(avi_files, npz_files):
            npz_data = np.load(npz_path)
            output = npz_data["output"]
            metadata = npz_data["metadata"]
            dilated_output = output[::k]
            output_npz = os.path.join(output_dir, os.path.basename(npz_path))
            np.savez(
                output_npz,
                output=dilated_output,
                metadata=metadata,
            )
            cap = cv2.VideoCapture(avi_path)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS) / k
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_avi = os.path.join(output_dir, os.path.basename(avi_path))
            writer = cv2.VideoWriter(output_avi, fourcc, fps, (w, h))
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % k == 0:
                    writer.write(frame)
                frame_idx += 1
            cap.release()
            writer.release()

    def visualise_videos(self):
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        if not avi_files:
            print("No videos found.")
            return
        video_idx = 0
        frame_idx = 0
        # Interactively this browses frames until 'q'. Non-interactively there is
        # no key press to end it, so step through a bounded number of frames and
        # return (override with DIFFTACTILE_MAX_FRAMES).
        limit = iteration_limit("DIFFTACTILE_MAX_FRAMES", 20)
        shown = 0
        while limit is None or shown < limit:
            shown += 1
            avi_path = avi_files[video_idx]
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = max(0, min(frame_idx, n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {avi_path}")
                break
            display = frame.copy()
            text = f"Video {video_idx + 1}/{len(avi_files)} | Frame {frame_idx + 1}/{n_frames}"
            cv2.putText(
                display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            imshow(cv2, "Video Viewer", display)
            key = wait_key(cv2, 0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                video_idx = (video_idx + 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("n"):
                video_idx = (video_idx - 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("j"):
                frame_idx -= 1
            else:
                # 'k' advances interactively; with no key press this also steps
                # the bounded non-interactive loop forward.
                frame_idx += 1
        destroy_windows(cv2)

    def extract_markers(self):
        # Lazy: see the note by the imports at the top of this file.
        from difftactile.data_analysis.experiment.predict_exp import PredictExp

        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_markers")
        os.makedirs(output_dir, exist_ok=True)
        avi_files, npz_files = self._get_file_pairs(input_dir)
        for avi_path in avi_files:
            base_name = os.path.splitext(os.path.basename(avi_path))[0]
            video_in = avi_path
            video_out = os.path.join(output_dir, f"{base_name}_markers.avi")
            npz_out = os.path.join(output_dir, f"{base_name}_markers.npz")
            PredictExp.compute_npz_helper3(
                video_in=video_in,
                video_out=video_out,
                npz_out=npz_out,
            )
            for npz_path in npz_files:
                base_name = os.path.splitext(os.path.basename(npz_path))[0]
                dst_path = os.path.join(output_dir, f"{base_name}_poses.npz")
                shutil.copy(npz_path, dst_path)

    def reorder_interpolate_markers(self):
        # Lazy: see the note by the imports at the top of this file.
        from difftactile.data_analysis.experiment.predict_exp import PredictExp

        input_dir = os.path.join(self.root, f"{self.dir}_markers")
        output_dir = os.path.join(
            self.root, f"{self.dir}_reordered_interpolated_markers"
        )
        os.makedirs(output_dir, exist_ok=True)
        avi_files, npz_files_poses, npz_files_markers = self._get_file_pairs_2(
            input_dir
        )
        for avi_path, npz_in in zip(avi_files, npz_files_markers):
            base_name = os.path.splitext(os.path.basename(avi_path))[0]
            video_in = avi_path
            video_out = os.path.join(output_dir, f"{base_name}.avi")
            npz_out = os.path.join(output_dir, f"{base_name}.npz")
            PredictExp.compute_npz_helper2(
                video_in=video_in,
                video_out=video_out,
                npz_in=npz_in,
                npz_out=npz_out,
            )
            for npz_path in npz_files_poses:
                base_name = os.path.splitext(os.path.basename(npz_path))[0]
                dst_path = os.path.join(output_dir, f"{base_name}.npz")
                shutil.copy(npz_path, dst_path)

    def annotate(self):
        """Hand-annotate up to four points per frame by clicking on them.

        This is a manual-input tool: every annotation comes from a mouse click,
        so there is nothing for it to do unattended. Without
        DIFFTACTILE_INTERACTIVE=1 it returns immediately, leaving any existing
        `.pkl` annotations on disk untouched for the later stages to consume.

        Opening the tool on already-annotated videos redraws what is on disk, so
        the same window reviews existing annotations and creates new ones.

        Two safeguards, because it is easy to modify a published annotation set
        by accident while only meaning to look at it:

        * **Editing is off until you press `g`.** Clicks are ignored while the
          guard is on, which is the default. The setting is per session, not per
          frame or per video, so it stays as you left it while browsing.
        * **`z` undoes the last point added in THIS session**, across videos.
          Points loaded from disk are never on the undo stack, so pressing `z`
          more times than you made annotations cannot eat into previous work.

        Nothing reaches disk until `p` (save) or `q` (save and quit). `x` quits
        **without** saving, discarding everything changed since the last save
        (confirmed by a second `x`, so a mistyped key cannot throw work away).
        Killing the process has the same effect as `x`.
        """
        if not is_interactive():
            print(
                "Skipping annotate(): it only produces annotations from mouse "
                "clicks. Set DIFFTACTILE_INTERACTIVE=1 to open the annotator. "
                "Existing annotations on disk are left as they are."
            )
            return
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_annotations")
        os.makedirs(output_dir, exist_ok=True)
        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        if not avi_files:
            print("No videos found.")
            return
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        annotations = {}
        for avi_path in avi_files:
            base = os.path.splitext(os.path.basename(avi_path))[0]
            pkl_path = os.path.join(output_dir, f"{base}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    annotations[avi_path] = pickle.load(f)
            else:
                cap = cv2.VideoCapture(avi_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                annotations[avi_path] = [[] for _ in range(n_frames)]
        video_idx = 0
        # Frame index lives in a dict so the render/key callbacks below can
        # mutate it without `nonlocal` gymnastics; `dirty` is set by the mouse
        # callback to request a repaint after a click.
        # `scale` is the factor the last rendered frame was shrunk by for
        # display, so the mouse callback can convert clicks back to full
        # resolution.
        state = {"frame": 0, "dirty": False, "scale": 1.0}

        # Editing guard, toggled with `g` and held for the whole session. Off by
        # default so that merely opening the tool to look at existing
        # annotations cannot modify them with a stray click.
        edit_enabled = [False]

        # Points added during THIS session, most recent last, as
        # (avi_path, frame_idx). `z` pops from here, so undo can walk back across
        # videos and frames in the order the points were actually placed, and
        # stops dead once the session's own additions are exhausted - annotations
        # loaded from disk are never pushed here and so can never be undone.
        session_additions = []

        # Snapshot of what is on disk, so `x` can report whether there is
        # anything to discard. Updated on every save, because after `p` the
        # on-disk state IS the current state and only later edits are unsaved.
        saved_state = copy.deepcopy(annotations)

        def save_all():
            """Write every video's annotations and re-baseline `saved_state`."""
            for path in avi_files:
                base = os.path.splitext(os.path.basename(path))[0]
                pkl_path = os.path.join(output_dir, f"{base}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(annotations[path], f)
            saved_state.clear()
            saved_state.update(copy.deepcopy(annotations))

        def unsaved_changes():
            """True when the in-memory annotations differ from what was saved."""
            return annotations != saved_state

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if not edit_enabled[0]:
                    print(
                        "Editing is locked - press 'g' to enable adding points "
                        "(the setting lasts for the whole session)."
                    )
                    return
                avi_path = avi_files[video_idx]
                frame_idx = state["frame"]
                frame_annots = annotations[avi_path][frame_idx]
                if len(frame_annots) < 4:
                    # The window shows a downscaled copy of the frame (see
                    # fit_to_view), so a click arrives in display pixels. Map it
                    # back to full-resolution pixels before storing, since every
                    # downstream stage expects annotations in the video's own
                    # coordinates.
                    scale = state["scale"] or 1.0
                    annotations[avi_path][frame_idx].append(
                        [int(round(x / scale)), int(round(y / scale))]
                    )
                    session_additions.append((avi_path, frame_idx))
                    # Ask the browser loop to repaint so the new dot appears
                    # without needing a keypress.
                    state["dirty"] = True
                    print(f"Added point {len(frame_annots) + 1}/{4}")
                else:
                    print("Maximum 4 points allowed.")

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", mouse_callback)

        # Decoded frames per video, cached: the loop used to re-open the .avi and
        # seek on every keypress, which is the bulk of the per-step cost.
        frame_cache = {}

        def video_frames(path):
            if path not in frame_cache:
                cap = cv2.VideoCapture(path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                frame_cache[path] = frames
            return frame_cache[path]

        def render():
            """Frame with annotations and status lines, or None to end."""
            avi_path = avi_files[video_idx]
            frames = video_frames(avi_path)
            if not frames:
                print(f"Failed to read frames from {avi_path}")
                return None
            n_frames = len(frames)
            state["frame"] = max(0, min(state["frame"], n_frames - 1))
            # Downscale FIRST, then draw: the overlays are then rendered into a
            # ~4x smaller buffer, and the status text keeps a constant on-screen
            # size instead of shrinking with the frame. The stored annotations
            # stay in full-resolution coordinates, so they are scaled on the way
            # in here and clicks are scaled back out in the mouse callback.
            display, scale = fit_to_view(cv2, frames[state["frame"]])
            display = display.copy()
            state["scale"] = scale
            frame_annots = annotations[avi_path][state["frame"]]
            for idx, pt in enumerate(frame_annots):
                color = colors[idx % len(colors)]
                cv2.circle(
                    display,
                    (int(round(pt[0] * scale)), int(round(pt[1] * scale))),
                    max(3, int(round(30 * scale))),
                    color,
                    -1,
                )
            text = (
                f"Video {video_idx + 1}/{len(avi_files)} | "
                f"Frame {state['frame'] + 1}/{n_frames} | "
                f"Annotations {len(frame_annots)}/4"
            )
            cv2.putText(
                display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            # Second line: whether a click will do anything, and how much of this
            # session's work `z` can still take back. Red when locked, green when
            # editing, so the state is obvious at a glance.
            if edit_enabled[0]:
                status = f"EDIT ON (g to lock) | undo available: {len(session_additions)}"
                status_colour = (0, 255, 0)
            else:
                status = "EDIT LOCKED - press g to enable clicking"
                status_colour = (0, 0, 255)
            cv2.putText(
                display, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                status_colour, 2
            )
            # Third line: whether anything would be lost by quitting with `x`.
            dirty = unsaved_changes()
            exit_hint = (
                "UNSAVED changes | p save, q save+quit, x quit WITHOUT saving"
                if dirty
                else "no unsaved changes | q or x to quit"
            )
            cv2.putText(
                display, exit_hint, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 200, 255) if dirty else (200, 200, 200), 2
            )
            return display

        def on_key(key):
            """Map a keypress to a state change; see the docstring for bindings."""
            nonlocal video_idx
            avi_path = avi_files[video_idx]
            n_frames = len(video_frames(avi_path))
            if key == ord("q"):
                save_all()
                return "quit"
            if key == ord("x"):
                # Quit WITHOUT saving. Nothing has reached disk unless `p` was
                # pressed, so this drops the in-memory state - which is why the
                # prompt talks about changes since the last save, not launch.
                if not unsaved_changes():
                    print("No unsaved changes; exiting.")
                    return "quit"
                print(
                    f"Discard {len(session_additions)} unsaved point(s) added this "
                    "session and any frame clears, and quit? Press x again to "
                    "confirm, or any other key to stay."
                )
                # Confirm on a second press, so a mistyped key cannot throw away
                # work the same way an accidental `q` used to save it.
                if (wait_key(cv2, 3000) & 0xFF) == ord("x"):
                    print("Exited without saving. Files on disk are unchanged.")
                    return "quit"
                print("Cancelled; still editing.")
                return None
            if key == ord("m"):
                video_idx = (video_idx + 1) % len(avi_files)
                state["frame"] = 0
                return "redraw"
            if key == ord("n"):
                video_idx = (video_idx - 1) % len(avi_files)
                state["frame"] = 0
                return "redraw"
            if key == ord("k"):
                new_frame = min(state["frame"] + 1, n_frames - 1)
                if new_frame == state["frame"]:
                    return None
                state["frame"] = new_frame
                return "redraw"
            if key == ord("j"):
                new_frame = max(state["frame"] - 1, 0)
                if new_frame == state["frame"]:
                    return None
                state["frame"] = new_frame
                return "redraw"
            if key == ord("g"):
                edit_enabled[0] = not edit_enabled[0]
                print(
                    "Editing ENABLED - clicks now add points."
                    if edit_enabled[0]
                    else "Editing LOCKED - clicks are ignored."
                )
                return "redraw"
            if key == ord("z"):
                if not session_additions:
                    print(
                        "Nothing to undo: no points were added in this session. "
                        "(Annotations loaded from disk are never undone.)"
                    )
                    return None
                undo_path, undo_frame = session_additions.pop()
                frame_points = annotations[undo_path][undo_frame]
                if frame_points:
                    frame_points.pop()
                # Jump to wherever that point was, so the undo is visible rather
                # than silently happening on an off-screen frame.
                video_idx = avi_files.index(undo_path)
                state["frame"] = undo_frame
                print(
                    f"Undid one point ({len(session_additions)} left from this session)."
                )
                return "redraw"
            if key == ord("d"):
                # Clearing a frame drops points this session added there too, so
                # they must leave the undo stack - otherwise a later `z` would
                # pop a point that no longer exists and remove someone else's.
                removed = len(annotations[avi_path][state["frame"]])
                annotations[avi_path][state["frame"]] = []
                session_additions[:] = [
                    entry for entry in session_additions
                    if entry != (avi_path, state["frame"])
                ]
                print(f"Deleted {removed} annotation(s) for this frame.")
                return "redraw"
            if key == ord("p"):
                save_all()
                print("Annotations saved.")
                return "redraw"
            return None

        def needs_repaint():
            """True when a mouse click changed the annotations since last draw."""
            if state["dirty"]:
                state["dirty"] = False
                return True
            return False

        run_frame_browser(cv2, "Annotator", render, on_key, needs_repaint)
        destroy_windows(cv2)
        for _ in range(10):
            cv2.waitKey(1)
            time.sleep(0.1)

    def annotations_to_line_points(self):
        cx = self.cx
        cy = self.cy
        r = self.r
        ann_dir = os.path.join(self.root, f"{self.dir}_annotations")
        out_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        os.makedirs(out_dir, exist_ok=True)
        pkl_files = sorted(glob.glob(os.path.join(ann_dir, "*.pkl")))
        if not pkl_files:
            print("No annotation files found.")
            return
        C = np.array([float(cx), float(cy)], dtype=np.float64)
        R2 = float(r) ** 2
        max_lines = 4
        num_pts = 50
        for i in range(len(pkl_files)):
            th = np.deg2rad(self.thetas[i])
            u = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
            pkl_path = pkl_files[i]
            base = os.path.splitext(os.path.basename(pkl_path))[0]
            out_path = os.path.join(out_dir, f"{base}.npz")
            with open(pkl_path, "rb") as f:
                ann = pickle.load(f)
            num_frames = len(ann)
            line_points = np.zeros(
                (num_frames, max_lines, num_pts, 2), dtype=np.float32
            )
            annotation_points = np.zeros(
                (num_frames, max_lines, 2), dtype=np.float32
            )
            mask = np.zeros((num_frames, max_lines), dtype=bool)
            for fi, frame_pts in enumerate(ann):
                for li, pt in enumerate(frame_pts[:max_lines]):
                    P0 = np.array(pt, dtype=np.float64)
                    t0 = -np.dot(P0 - C, u)
                    F = P0 + t0 * u
                    d2 = np.sum((F - C) ** 2)
                    if d2 > R2:
                        continue
                    L = float(np.sqrt(max(R2 - d2, 0.0)))
                    A = F - L * u
                    B = F + L * u
                    xs = np.linspace(A[0], B[0], num_pts, dtype=np.float32)
                    ys = np.linspace(A[1], B[1], num_pts, dtype=np.float32)
                    line_points[fi, li, :, 0] = xs
                    line_points[fi, li, :, 1] = ys
                    mask[fi, li] = True
                    annotation_points[fi, li] = P0
            np.savez(
                out_path,
                line_points=line_points,
                mask=mask,
                annotation_points=annotation_points,
            )

    def count_annotation_dots(self):
        ann_dir = os.path.join(self.root, f"{self.dir}_annotations")
        pkl_files = sorted(glob.glob(os.path.join(ann_dir, "*.pkl")))
        if not pkl_files:
            print(f"No .pkl files found in {ann_dir}")
            return

        total_dots = 0
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                ann = pickle.load(f)

            file_dots = 0
            for frame_pts in ann:
                if frame_pts is None:
                    continue
                file_dots += len(frame_pts)

            total_dots += file_dots
            print(f"{os.path.basename(pkl_path)}: {file_dots} annotation dots")

        print(f"Total annotation dots across all .pkl files: {total_dots}")

    def visualise_line_points(self):
        cx = self.cx
        cy = self.cy
        r = self.r
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        npz_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not avi_files or not npz_files:
            print("No videos or line points found.")
            return
        avi_map = {os.path.splitext(os.path.basename(f))[0]: f for f in avi_files}
        npz_map = {os.path.splitext(os.path.basename(f))[0]: f for f in npz_files}
        common_keys = sorted(set(avi_map.keys()) & set(npz_map.keys()))
        if not common_keys:
            print("No matching video/npz pairs found.")
            return
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        video_idx = 0
        frame_idx = 0
        # Bounded when non-interactive; see the note in visualise_videos().
        limit = iteration_limit("DIFFTACTILE_MAX_FRAMES", 20)
        shown = 0
        while limit is None or shown < limit:
            shown += 1
            key_base = common_keys[video_idx]
            avi_path = avi_map[key_base]
            npz_path = npz_map[key_base]
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = max(0, min(frame_idx, n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {avi_path}")
                break
            data = np.load(npz_path)
            line_points = data["line_points"]
            mask = data["mask"]
            annotation_points = data['annotation_points']
            display = frame.copy()
            if frame_idx < line_points.shape[0]:
                for line_idx in range(4):
                    if mask[frame_idx, line_idx]:
                        pts = line_points[frame_idx, line_idx]
                        for x, y in pts.astype(int):
                            cv2.circle(display, (x, y), 10, colors[line_idx], -1)
                        p0 = annotation_points[frame_idx, line_idx]
                        x0, y0 = p0
                        cv2.circle(display, (int(x0), int(y0)), 30, colors[line_idx], -1)
            # cv2.circle(display, (cx, cy), radius=r, color=(0, 255, 255), thickness=3)
            text = f"Video {video_idx + 1}/{len(common_keys)} | Frame {frame_idx + 1}/{n_frames}"
            cv2.putText(
                display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            imshow(cv2, "Line Points Viewer", display)
            key = wait_key(cv2, 0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                video_idx = (video_idx + 1) % len(common_keys)
                frame_idx = 0
            elif key == ord("n"):
                video_idx = (video_idx - 1) % len(common_keys)
                frame_idx = 0
            elif key == ord("j"):
                frame_idx -= 1
            else:
                # 'k' advances interactively and drives the bounded loop here.
                frame_idx += 1
        destroy_windows(cv2)

    def merge_npz_to_sim_format(self):
        annotations_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        markers_dir = os.path.join(
            self.root, f"{self.dir}_reordered_interpolated_markers"
        )
        output_dir = os.path.join(self.root, f"{self.dir}_sim_format")
        os.makedirs(output_dir, exist_ok=True)
        ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.npz")))
        if not ann_files:
            print("No annotation npz files found.")
            return
        for ann_path in ann_files:
            base = os.path.splitext(os.path.basename(ann_path))[0]
            marker_path = os.path.join(markers_dir, f"{base}_markers.npz")
            if not os.path.exists(marker_path):
                print(f"Skipping {base}: marker file not found at {marker_path}")
                continue
            line_point_data = np.load(ann_path)
            marker_data = np.load(marker_path)
            markers = marker_data["markers"]
            markers_mask = marker_data["markers_mask"]
            vein_polyline = line_point_data["line_points"]
            vein_polyline_mask = line_point_data["mask"]
            a, b, c, d = vein_polyline.shape
            vein_polyline_mask = np.broadcast_to(
                vein_polyline_mask[..., None], (a, b, c)
            )
            out_path = os.path.join(output_dir, f"{base}.npz")
            np.savez(
                out_path,
                markers=markers,
                markers_mask=markers_mask,
                vein_polyline=vein_polyline,
                vein_polyline_mask=vein_polyline_mask,
            )


    def merge_npz_sim_format_poses(self):
        sim_dir = os.path.join(self.root, f"{self.dir}_sim_format")
        dilated_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_sim_format_poses")
        os.makedirs(output_dir, exist_ok=True)

        sim_files = sorted(glob.glob(os.path.join(sim_dir, "*.npz")))
        if not sim_files:
            print("No sim_format npz files found.")
            return

        for sim_path in sim_files:
            base = os.path.basename(sim_path)
            dilated_path = os.path.join(dilated_dir, base)
            if not os.path.exists(dilated_path):
                print(f"Skipping {base}: matching dilated npz not found.")
                continue

            sim_format_data = np.load(sim_path)
            dilated_data = np.load(dilated_path)

            markers = sim_format_data["markers"]
            markers_mask = sim_format_data["markers_mask"]
            vein_polyline = sim_format_data["vein_polyline"]
            vein_polyline_mask = sim_format_data["vein_polyline_mask"]
            poses = dilated_data["output"]
            metadata = dilated_data["metadata"]

            out_path = os.path.join(output_dir, base)
            np.savez(
                out_path,
                markers=markers,
                markers_mask=markers_mask,
                vein_polyline=vein_polyline,
                vein_polyline_mask=vein_polyline_mask,
                poses=poses,
                metadata=metadata,
            )

    def add_dense_line_data(self):
        input_dir = os.path.join(self.root, f"{self.dir}_sim_format_poses")
        output_dir = os.path.join(self.root, f"{self.dir}_dense")
        # input_dir = 'difftactile/output/training_data/pickle_20250901_220921_reordered'
        # output_dir = 'difftactile/output/training_data/pickle_20250901_220921_reordered_dense'
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
        if not files:
            print("No sim_format_poses npz files found.")
            return

        for path in files:
            base = os.path.basename(path)
            data = np.load(path)

            # Extract existing arrays
            markers = data["markers"]
            markers_mask = data["markers_mask"]
            vein_polyline = data["vein_polyline"]          # (frames, veins, pts, 2)
            vein_polyline_mask = data["vein_polyline_mask"]  # (frames, veins, pts)
            poses = data["poses"]
            metadata = data["metadata"]

            num_frames, max_num_veins, num_points, _ = vein_polyline.shape

            vein_classification = np.zeros((num_frames, max_num_veins), dtype=np.int32)
            vein_regression = np.zeros((num_frames, max_num_veins, 3), dtype=np.float32)

            # You should already know cx, cy
            cx, cy = self.cx, self.cy  # Make sure these exist in your class

            for t in range(num_frames):
                for i in range(max_num_veins):
                    vein = vein_polyline[t, i][vein_polyline_mask[t, i]]

                    if len(vein) < 2:
                        # Not enough points → no vein
                        vein_classification[t, i] = 0
                        vein_regression[t, i] = [0, 0, 0]
                        continue

                    # Compute pairwise distances using pdist
                    dists = pdist(vein)  # condensed form
                    dist_matrix = squareform(dists)
                    a, b = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
                    p1, p2 = vein[a], vein[b]

                    # Construct vector (dx, dy)
                    dx, dy = p2 - p1
                    theta = math.atan2(dy, dx)

                    # cos(2θ), sin(2θ)
                    cos2 = math.cos(2 * theta)
                    sin2 = math.sin(2 * theta)

                    # Line equation: y = m(x - x0) + y0
                    if dx != 0:
                        m = dy / dx
                        y_intercept = m * (cx - p1[0]) + p1[1]
                    else:
                        y_intercept = cy  # vertical line, intersect at cy

                    vein_classification[t, i] = 1
                    vein_regression[t, i] = [cos2, sin2, y_intercept]

            # Save with extra fields
            out_path = os.path.join(output_dir, base)
            np.savez(
                out_path,
                markers=markers,
                markers_mask=markers_mask,
                vein_polyline=vein_polyline,
                vein_polyline_mask=vein_polyline_mask,
                poses=poses,
                metadata=metadata,
                vein_classification=vein_classification,
                vein_regression=vein_regression,
            )

    @staticmethod
    def line_dense_to_sparse(vein_classification, vein_regression, cx, cy, r, num_points=50):
        """
        Convert dense vein representation (classification + regression) 
        into sparse polyline format (line points + mask).
        
        Args:
            vein_classification (np.ndarray): shape (num_frames, 4), classification labels (0/1).
            vein_regression (np.ndarray): shape (num_frames, 4, 3), regression values [cos(2θ), sin(2θ), y_intercept].
            cx, cy (float): circle center used to define chords.
            r (float): circle radius.
            num_points (int): number of points along each line polyline.

        Returns:
            vein_polyline (np.ndarray): shape (num_frames, 4, num_points, 2).
            vein_polyline_mask (np.ndarray): shape (num_frames, 4, num_points).
        """
        num_frames, num_veins = vein_classification.shape
        vein_polyline = np.zeros((num_frames, num_veins, num_points, 2), dtype=np.float32)
        vein_polyline_mask = np.zeros((num_frames, num_veins, num_points), dtype=bool)

        for t in range(num_frames):
            for i in range(num_veins):
                if vein_classification[t, i] == 0:
                    continue  # vein absent

                cos2, sin2, y_intercept = vein_regression[t, i]

                # Reconstruct angle theta from cos(2θ), sin(2θ)
                theta = 0.5 * math.atan2(sin2, cos2)

                # Direction vector of the line
                dx, dy = math.cos(theta), math.sin(theta)

                # Point on the line: intersection with vertical line x = cx
                x0, y0 = cx, y_intercept

                # Solve intersection with circle (x-cx)^2 + (y-cy)^2 = r^2
                # Parametric line: (x, y) = (x0, y0) + t * (dx, dy)
                A = dx**2 + dy**2
                B = 2 * (dx * (x0 - cx) + dy * (y0 - cy))
                C = (x0 - cx)**2 + (y0 - cy)**2 - r**2

                disc = B**2 - 4 * A * C
                if disc < 0:
                    continue  # no intersection
                sqrt_disc = math.sqrt(disc)
                t1 = (-B - sqrt_disc) / (2 * A)
                t2 = (-B + sqrt_disc) / (2 * A)

                p1 = np.array([x0 + t1 * dx, y0 + t1 * dy])
                p2 = np.array([x0 + t2 * dx, y0 + t2 * dy])

                # Generate evenly spaced points between p1 and p2
                line_points = np.linspace(p1, p2, num_points)

                vein_polyline[t, i, :, :] = line_points
                vein_polyline_mask[t, i, :] = True

        return vein_polyline, vein_polyline_mask

    def validate_line_dense_to_sparse(self, cx, cy, r, num_points=50):
        """
        Validate dense → sparse reconstruction of vein data.

        Loads all npz files from {self.dir}_dense, reconstructs vein_polyline and
        vein_polyline_mask from vein_classification + vein_regression, and saves
        updated npz files to {self.dir}_sparse_dense_sparse_validation.

        Args:
            cx, cy (float): Circle center used to define chords.
            r (float): Circle radius.
            num_points (int): Number of evenly spaced points along each vein chord.
        """
        input_dir = os.path.join(self.root, f"{self.dir}_dense")
        output_dir = os.path.join(self.root, f"{self.dir}_sparse_dense_sparse_validation")
        os.makedirs(output_dir, exist_ok=True)

        npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
        if not npz_files:
            print(f"No .npz files found in {input_dir}")
            return

        for npz_path in npz_files:
            base_name = os.path.basename(npz_path)
            out_path = os.path.join(output_dir, base_name)

            data = np.load(npz_path, allow_pickle=True)

            markers = data["markers"]
            markers_mask = data["markers_mask"]
            poses = data["poses"]
            metadata = data["metadata"]
            vein_classification = data["vein_classification"]
            vein_regression = data["vein_regression"]

            # Reconstruct polyline + mask from dense fields
            vein_polyline, vein_polyline_mask = SiliconePreprocessData.line_dense_to_sparse(
                vein_classification, vein_regression, cx, cy, r, num_points
            )

            # Save with same structure but updated polyline/mask
            np.savez(
                out_path,
                markers=markers,
                markers_mask=markers_mask,
                vein_polyline=vein_polyline,
                vein_polyline_mask=vein_polyline_mask,
                poses=poses,
                metadata=metadata,
                vein_classification=vein_classification,
                vein_regression=vein_regression,
            )


def main():
    e = SiliconePreprocessData()
    # e.interpolate_metadata_and_trim_videos()
    # e.apply_dilation(16)
    # e.visualise_videos()
    # e.extract_markers()
    # e.reorder_interpolate_markers()
    # e.annotate()
    # e.annotations_to_line_points()
    # e.visualise_line_points()
    # e.merge_npz_to_sim_format()
    # e.merge_npz_sim_format_poses()
    # e.add_dense_line_data()
    # e.validate_line_dense_to_sparse(
    #     cx=e.cx,
    #     cy=e.cy,
    #     r=e.r,
    # )

    e.count_annotation_dots()


def annotate():
    """Open the silicone click-annotator (create AND review annotations).

    Exposed as its own entrypoint because `main()` above is a commented-out menu
    of the full preprocessing pipeline; this reaches the one stage that is a
    manual tool without editing source. `annotate()` loads any existing `.pkl`
    for each video and redraws it, so the same window both reviews the shipped
    annotations and adds to them.
    """
    SiliconePreprocessData().annotate()


if __name__ == "__main__":
    main()
