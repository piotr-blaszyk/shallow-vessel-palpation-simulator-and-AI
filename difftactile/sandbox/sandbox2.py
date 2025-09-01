npz_out = an npz file with the name being f"{input_file_name}_markers" in the f"{self.dir}_markers" folder

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

Additionally, copy each file from the f"{self.dir}_dilated" folder to the f"{self.dir}_markers" folder but rename it to f"{input_file_name}_poses".
