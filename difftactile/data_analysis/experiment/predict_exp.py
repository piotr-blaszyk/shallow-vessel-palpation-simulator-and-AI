"""Video -> marker .npz helpers for the real-sensor preprocessing pipelines.

Historically this module also built the bird's-eye vessel map; that has moved
to `vessel_map.py` (all datasets, versioned output, chosen threshold). What
remains here are the static helpers `preprocess_silicone_data.py` and
`preprocess_meat_data.py` call to turn a recorded video into the marker .npz
files the GNN consumes. Both import this lazily, because it drags in torch via
the marker tracker's dependencies.
"""

import numpy as np

from difftactile.data_analysis.experiment.hungarian_exp import *
from difftactile.data_analysis.experiment.marker_tracker import *
from difftactile.main.constants import *
from difftactile.main.synthetic_image_generator import SyntheticImageGenerator


class PredictExp:
    """Namespace for the video -> .npz helpers (all static)."""

    def write_video_to_npz_file(marker_tracker, path):
        n = len(marker_tracker.frame_markers)
        markers_array, markers_mask = SyntheticImageGenerator.create_padded_array_with_mask(marker_tracker.frame_markers)
        vein_data = [np.array([]) for i in range(n)]
        vein_polyline, vein_polyline_mask = SyntheticImageGenerator.create_padded_array_with_mask(vein_data)
        target_id_array = np.zeros(shape=(0, 1), dtype=int)
        np.savez(
            path,
            markers=markers_array,
            markers_mask=markers_mask,
            vein_polyline=vein_polyline,
            vein_polyline_mask=vein_polyline_mask,
            target_id_array=target_id_array
        )
    
    def compute_npz_grid_search_og():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.experiment_og_raw_video,
            video_out=SYSTEM_PARAMS.files.experiment_og_processed_video,
            npz_in=SYSTEM_PARAMS.files.experiment_og_markers_npz,
            npz_out=SYSTEM_PARAMS.files.experiment_og_markers_reordered_npz,
            frame_mapping_npz_out=SYSTEM_PARAMS.files.experiment_og_frame_mapping_npz,
            video_from_cache=True,
            npz_temp=SYSTEM_PARAMS.files.experiment_og_markers_reordered_npz,
            labels_out=SYSTEM_PARAMS.files.experiment_og_ground_truth_labels_npz,
            labels_in=SYSTEM_PARAMS.files.experiment_og_ground_truth_labels_npz,
        )
    
    def compute_npz_straight():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.experiment_straight_raw_video,
            video_out=SYSTEM_PARAMS.files.experiment_straight_processed_video,
            npz_in=SYSTEM_PARAMS.files.experiment_straight_markers_npz,
            npz_out=SYSTEM_PARAMS.files.experiment_straight_markers_reordered_npz,
            frame_mapping_npz_out=SYSTEM_PARAMS.files.experiment_straight_frame_mapping_npz,
            video_from_cache=False,
            npz_temp=SYSTEM_PARAMS.files.experiment_straight_markers_reordered_npz,
            labels_out=SYSTEM_PARAMS.files.experiment_straight_ground_truth_labels_npz,
            labels_in=SYSTEM_PARAMS.files.experiment_straight_ground_truth_labels_npz,
            seconds_per_frame=1e0
        )
    
    def compute_npz_grid_search_2025_08_22():
        PredictExp.compute_npz_helper(
            video_in=SYSTEM_PARAMS.files.experiment_2025_08_22_raw_video,
            video_out=SYSTEM_PARAMS.files.experiment_2025_08_22_processed_video,
            npz_in=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_npz,
            # npz_out_reordered=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_reordered_npz,
            # frame_mapping_npz_out=SYSTEM_PARAMS.files.experiment_2025_08_22_frame_mapping_npz,
            # video_from_cache=True,
            # npz_in=SYSTEM_PARAMS.files.experiment_2025_08_22_markers_reordered_npz,
            # labels_out=SYSTEM_PARAMS.files.experiment_2025_08_22_ground_truth_labels_npz,
            # labels_in=SYSTEM_PARAMS.files.experiment_2025_08_22_ground_truth_labels_npz,
        )

    def compute_npz_helper(
        video_in,
        video_out,
        npz_out,
        npz_out_reordered=None,
        video_from_cache=False,
        npz_in=None,
        frame_mapping_npz_out=None,
        labels_out=None,
        labels_in=None,
        seconds_per_frame=None,
        foo=False,
    ):
        if not video_from_cache:
            marker_tracker = MarkerTracker()
            marker_tracker.extract_frames(
                video_in,
                frame_mapping_npz_out,
                seconds_per_frame=seconds_per_frame
            )
            PredictExp.write_video_to_npz_file(
                marker_tracker=marker_tracker,
                path=npz_out
            )
            if npz_out_reordered is not None:
                HungarianExp.reorder_exp_points(
                    input_path=npz_out,
                    output_path=npz_out_reordered
                )
            marker_tracker.create_visualization(
                out_path=video_out,
                mode="unpaired-markers",
                base_from_file=False,
                npz_in=npz_in
            )
        # player = VideoPlayer(
        #     video_in_path=video_out,
        #     markers_in_path_npz=npz_in,
        #     labels_out_path_npz=labels_out,
        #     labels_in_path_npz=labels_in
        # )
        # player.run()
    
    def compute_npz_helper2(
        video_in,
        video_out,
        npz_in,
        npz_out,
        mode=None,
    ):
        marker_tracker = MarkerTracker()
        marker_tracker.extract_frames(video_in, mode=mode)
        HungarianExp.reorder_exp_points(
            input_path=npz_in,
            output_path=npz_out,
        )
        marker_tracker.create_visualization(
            out_path=video_out,
            mode="unpaired-markers",
            base_from_file=False,
            npz_in=npz_out,
        )
    
    def compute_npz_helper3(
        video_in,
        video_out,
        npz_out,
        mode=None,
    ):
        marker_tracker = MarkerTracker()
        marker_tracker.extract_frames(video_in, mode=mode)
        PredictExp.write_video_to_npz_file(
            marker_tracker=marker_tracker,
            path=npz_out,
        )
        marker_tracker.create_visualization(
            out_path=video_out,
            mode="unpaired-markers",
            base_from_file=False,
        )


def main():
    """Kept for `script_predict_exp`: the vessel map now lives in vessel_map.py."""
    from difftactile.data_analysis.experiment.vessel_map import main as vessel_map_main
    print("predict_exp.main() is superseded by vessel_map.py; running that instead "
          "(prefer ./docker/vessel_map.sh).")
    vessel_map_main()
