import os
import glob
import cv2
import time
import pickle
import numpy as np

class Endgame:
    def __init__(self):
        self.root = 'difftactile/manual_or_experimental_data/endgame'
        self.dir = '20250901-131547'

    def annotate(self):
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_annotations")
        os.makedirs(output_dir, exist_ok=True)

        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        if not avi_files:
            print("No videos found.")
            return

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]  # BGR: red, green, blue, magenta

        # Load or initialize annotations
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

        # State variables
        video_idx = 0
        frame_idx = 0

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                avi_path = avi_files[video_idx]
                frame_annots = annotations[avi_path][frame_idx]
                if len(frame_annots) < 4:
                    annotations[avi_path][frame_idx].append([x, y])
                    print(f"Added point {len(frame_annots)}/{4}")
                else:
                    print("Maximum 4 points allowed.")

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", mouse_callback)

        while True:
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

            # Draw annotations for current frame
            frame_annots = annotations[avi_path][frame_idx]
            for idx, pt in enumerate(frame_annots):
                color = colors[idx % len(colors)]
                cv2.circle(display, (int(pt[0]), int(pt[1])), 6, color, -1)

            # Overlay text
            text = f"Video {video_idx+1}/{len(avi_files)} | Frame {frame_idx+1}/{n_frames} | Annotations {len(frame_annots)}/4"
            cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Annotator", display)
            key = cv2.waitKey(20) & 0xFF  # refresh faster to show new points immediately

            if key == ord("q"):
                # Save all annotations before exit
                for avi_path in avi_files:
                    base = os.path.splitext(os.path.basename(avi_path))[0]
                    pkl_path = os.path.join(output_dir, f"{base}.pkl")
                    with open(pkl_path, "wb") as f:
                        pickle.dump(annotations[avi_path], f)
                break
            elif key == ord("m"):
                video_idx = (video_idx + 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("n"):
                video_idx = (video_idx - 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("k"):
                frame_idx += 1
            elif key == ord("j"):
                frame_idx -= 1
            elif key == ord("d"):
                annotations[avi_path][frame_idx] = []
                print("Deleted annotations for this frame.")
            elif key == ord("p"):
                for avi_path in avi_files:
                    base = os.path.splitext(os.path.basename(avi_path))[0]
                    pkl_path = os.path.join(output_dir, f"{base}.pkl")
                    with open(pkl_path, "wb") as f:
                        pickle.dump(annotations[avi_path], f)
                print("Annotations saved.")

        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
            time.sleep(0.1)

    def annotations_to_line_points(self, cx=0.0, cy=0.0, r=100.0, theta=0.0):
        """
        Convert per-frame point annotations into evenly spaced points along chords
        of a circle with center (cx, cy), radius r, and chord orientation theta (deg).

        For each frame: up to 4 annotated points -> up to 4 chords -> 50 points each.
        Saves (num_frames, 4, 50, 2) array + (num_frames, 4) boolean mask to NPZ.
        """
        ann_dir = os.path.join(self.root, f"{self.dir}_annotations")
        out_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        os.makedirs(out_dir, exist_ok=True)

        pkl_files = sorted(glob.glob(os.path.join(ann_dir, "*.pkl")))
        if not pkl_files:
            print("No annotation files found.")
            return

        # Unit direction vector for the chord orientation
        th = np.deg2rad(theta)
        u = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
        C = np.array([float(cx), float(cy)], dtype=np.float64)
        R2 = float(r) ** 2

        max_lines = 4
        num_pts = 50

        for pkl_path in pkl_files:
            base = os.path.splitext(os.path.basename(pkl_path))[0]
            out_path = os.path.join(out_dir, f"{base}.npz")

            with open(pkl_path, "rb") as f:
                ann = pickle.load(f)  # list over frames; each element: list of points [[x,y], ...]

            num_frames = len(ann)
            line_points = np.zeros((num_frames, max_lines, num_pts, 2), dtype=np.float32)
            mask = np.zeros((num_frames, max_lines), dtype=bool)

            for fi, frame_pts in enumerate(ann):
                for li, pt in enumerate(frame_pts[:max_lines]):
                    P0 = np.array(pt, dtype=np.float64)

                    # Closest point on the chord line to the circle center
                    # Line parameterization: L(t) = P0 + t * u
                    # t0 chosen so that L(t0) is orthogonal projection of C onto the line
                    t0 = -np.dot(P0 - C, u)
                    F = P0 + t0 * u  # foot of perpendicular from C to the line

                    d2 = np.sum((F - C) ** 2)
                    if d2 > R2:
                        # No intersection: skip this annotation
                        # (Optionally clamp due to numerical noise)
                        # print(f"Warning: frame {fi} ann {li}: line misses circle (d^2={d2:.2f} > R^2={R2:.2f})")
                        continue

                    # Half-chord length along the line
                    L = float(np.sqrt(max(R2 - d2, 0.0)))

                    A = F - L * u  # endpoint 1 on circle
                    B = F + L * u  # endpoint 2 on circle

                    xs = np.linspace(A[0], B[0], num_pts, dtype=np.float32)
                    ys = np.linspace(A[1], B[1], num_pts, dtype=np.float32)
                    line_points[fi, li, :, 0] = xs
                    line_points[fi, li, :, 1] = ys
                    mask[fi, li] = True

            np.savez(
                out_path,
                line_points=line_points,
                mask=mask,
                cx=float(cx),
                cy=float(cy),
                r=float(r),
                theta=float(theta),
            )
            print(f"Saved {out_path}")
