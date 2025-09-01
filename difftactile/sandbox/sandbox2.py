video_in = video from the f"{self.dir}_dilated" folder
video_out = video with the same name but in the f"{self.dir}_markers" folder
npz_out = an npz file with the name being f"{input_file_name}_markers" in the f"{self.dir}_markers" folder

PredictExp.compute_npz_helper(
    video_in=video_in,
    video_out=video_out,
    npz_out=npz_out,
)

Additionally, copy each file from the f"{self.dir}_dilated" folder to the f"{self.dir}_markers" folder but rename it to f"{input_file_name}_poses".
