import os
import glob
import numpy as np
import math

class Endgame:
    # ... other methods ...

    def add_dense_line_data(self):
        input_dir = os.path.join(self.root, f"{self.dir}_sim_format_poses")
        output_dir = os.path.join(self.root, f"{self.dir}_dense")
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

                    # Compute pairwise distances to find furthest apart points
                    dmax = -1
                    p1 = p2 = None
                    for a in range(len(vein)):
                        for b in range(a + 1, len(vein)):
                            d = np.linalg.norm(vein[a] - vein[b])
                            if d > dmax:
                                dmax = d
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
            print(f"Saved dense file: {out_path}")
