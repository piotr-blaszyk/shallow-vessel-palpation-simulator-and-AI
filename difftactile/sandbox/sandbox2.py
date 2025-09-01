import os
import glob
import numpy as np

class Endgame:
    # ... other methods ...

    def merge_npz_to_sim_format(self):
        annotations_dir = os.path.join(self.root, f"{self.dir}_annotations_line_points")
        markers_dir = os.path.join(self.root, f"{self.dir}_reordered_interpolated_markers")
        output_dir = os.path.join(self.root, f"{self.dir}_sim_format")
        os.makedirs(output_dir, exist_ok=True)

        # Collect all annotation files
        ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.npz")))
        if not ann_files:
            print("No annotation npz files found.")
            return

        for ann_path in ann_files:
            base = os.path.splitext(os.path.basename(ann_path))[0]

            # Build corresponding marker file name
            marker_path = os.path.join(markers_dir, f"{base}_markers.npz")
            if not os.path.exists(marker_path):
                print(f"Skipping {base}: marker file not found at {marker_path}")
                continue

            # Load data
            line_point_data = np.load(ann_path)
            marker_data = np.load(marker_path)

            markers = marker_data["markers"]
            markers_mask = marker_data["markers_mask"]
            vein_polyline = line_point_data["line_points"]
            vein_polyline_mask = line_point_data["mask"]

            # Expand mask to match vein_polyline shape
            a, b, c, d = vein_polyline.shape  # (num_frames, max_num_lines, num_line_points, 2)
            vein_polyline_mask = np.broadcast_to(
                vein_polyline_mask[..., None], (a, b, c)
            )

            # Save merged file
            out_path = os.path.join(output_dir, f"{base}.npz")
            np.savez(
                out_path,
                markers=markers,
                markers_mask=markers_mask,
                vein_polyline=vein_polyline,
                vein_polyline_mask=vein_polyline_mask,
            )
            print(f"Saved merged npz to {out_path}")
