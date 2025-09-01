Now I want to implement the following method.

    def merge_npz_to_sim_format(self):
        pass

I want to load all npz files from the f"{self.dir}_annotations_line_points". I also want to load those npz files whose base file name ends in "_markers" from the f"{self.dir}_reordered_interpolated_markers" directory. 