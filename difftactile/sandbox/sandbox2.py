import os
import glob
import pickle
import numpy as np

class Endgame:
    def __init__(self):
        self.root = "difftactile/manual_or_experimental_data/endgame"
        self.dir = "20250901-131547"

    def annotations_to_line_points(self, cx=0, cy=0, r=100, theta=0):
        input_dir = os.path.join(self.root, f"{self.dir}_annotations")
        output_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        os.makedirs(output_dir, exist_ok=True)

        pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
        if not pkl_files:
            print("No annotation files found.")
            return

        # Convert theta to radians
        theta_rad = np.deg2rad(theta)
        dx, dy = np.cos(theta_rad), np.sin(theta_rad)

        for pkl_path in pkl_files:
            base = os.path.splitext(os.path.basename(pkl_path))[0]
            npz_path = os.path.join(output_dir, f"{base}.npz")

            with open(pkl_path, "rb") as f:
                annotations = pickle.load(f)  # list of frames, each frame = list of points

            num_frames = len(annotations)
            max_lines = 4
            num_line_points = 50

            # Allocate padded arrays
            line_points = np.zeros((num_frames, max_lines, num_line_points, 2), dtype=np.float32)
            mask = np.zeros((num_frames, max_lines), dtype=bool)

            for frame_idx, frame_annots in enumerate(annotations):
                for line_idx, pt in enumerate(frame_annots[:max_lines]):
                    px, py = pt

                    # Compute circle-line intersection
                    # Line through circle center at (cx,cy) with direction (dx,dy)
                    a = dx**2 + dy**2
                    b = 2 * (dx * (cx - px) + dy * (cy - py))
                    c = (cx - px) ** 2 + (cy - py) ** 2 - r**2

                    disc = b**2 - 4 * a * c
                    if disc < 0:
                        print(f"Warning: no intersection for frame {frame_idx}, point {pt}")
                        continue

                    t1 = (-b - np.sqrt(disc)) / (2 * a)
                    t2 = (-b + np.sqrt(disc)) / (2 * a)

                    # Endpoints on the circle
                    x1, y1 = px + t1 * dx, py + t1 * dy
                    x2, y2 = px + t2 * dx, py + t2 * dy

                    # Interpolate 50 points between endpoints
                    xs = np.linspace(x1, x2, num_line_points)
                    ys = np.linspace(y1, y2, num_line_points)
                    line_points[frame_idx, line_idx, :, :] = np.stack([xs, ys], axis=-1)

                    mask[frame_idx, line_idx] = True

            # Save npz
            np.savez(
                npz_path,
                line_points=line_points,
                mask=mask,
            )
            print(f"Saved {npz_path}")
