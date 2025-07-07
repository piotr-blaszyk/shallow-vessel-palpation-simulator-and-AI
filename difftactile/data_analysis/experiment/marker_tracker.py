import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import pickle
from difftactile.sensor_model.fisheye_model import *
from difftactile.main.constants import *


class MarkerTracker:
    def __init__(self, video_path, output_path):
        self.fisheye_model = FisheyeModel()
        self.video_path = video_path
        self.output_path = output_path
        self.frame_markers = []
        self.frame_mappings = []
        self.base_frame_mappings = []
        self.frames = []
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def extract_frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * SYSTEM_PARAMS.marker_tracker.seconds_per_frame)
        frame_count = 0
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(frame)
                markers, _, _ = self.fisheye_model.get_marker_image(gray)
                if len(markers) > 0:
                    self.frame_markers.append(markers)
                else:
                    self.frame_markers.append(np.array([]))
                frame_idx += 1
            frame_count += 1
        cap.release()

    def match_consecutive_frames(self):
        for i in range(len(self.frame_markers) - 1):
            current_markers = self.frame_markers[i]
            next_markers = self.frame_markers[i + 1]

            mapping = np.full((len(next_markers), 2), np.nan)
            if len(current_markers) == 0 or len(next_markers) == 0:
                self.frame_mappings.append(mapping)
                continue

            if len(next_markers) <= len(current_markers):
                markers_1, markers_2 = next_markers, current_markers
                reverse_mapping = False
            else:
                markers_1, markers_2 = current_markers, next_markers
                reverse_mapping = True

            row_ind, col_ind = linear_sum_assignment(cdist(markers_1, markers_2, metric="sqeuclidean"))
            distances = []

            for r, c in zip(row_ind, col_ind):
                dist = np.linalg.norm(markers_1[r] - markers_2[c])
                distances.append(dist)

            if distances:
                percentile_dist = np.percentile(distances, 90)
                dist_threshold = percentile_dist * 10
                for r, c, dist in zip(row_ind, col_ind, distances):
                    if dist <= dist_threshold:
                        if reverse_mapping:
                            mapping[c] = [r, dist]
                        else:
                            mapping[r] = [c, dist]

            self.frame_mappings.append(mapping)

    def compute_base_frame_mappings(self):
        base_markers = self.frame_markers[0]
        last_frame_mapping = np.full((len(self.frame_markers[-1]), 2), np.nan)
        
        for marker_idx in range(len(self.frame_markers[-1])):
            current_marker_idx = marker_idx
            valid_chain = True
            current_frame = len(self.frame_markers) - 1
            
            while current_frame > 0:
                prev_frame = current_frame - 1
                prev_frame_mapping = self.frame_mappings[prev_frame]
                if current_marker_idx >= len(prev_frame_mapping) or np.isnan(prev_frame_mapping[current_marker_idx][0]):
                    valid_chain = False
                    break
                prev_frame_marker_idx = int(prev_frame_mapping[current_marker_idx][0])
                if prev_frame_marker_idx >= len(self.frame_markers[prev_frame]):
                    valid_chain = False
                    break
                current_marker_idx = prev_frame_marker_idx
                current_frame = prev_frame
                
            if valid_chain and current_marker_idx < len(base_markers):
                dist = np.linalg.norm(self.frame_markers[-1][marker_idx] - base_markers[current_marker_idx])
                last_frame_mapping[marker_idx] = [current_marker_idx, dist]

        valid_base_indices = set()
        for map_entry in last_frame_mapping:
            if not np.isnan(map_entry[0]):
                valid_base_indices.add(int(map_entry[0]))

        for frame_idx in range(1, len(self.frame_markers)):
            current_frame_markers = self.frame_markers[frame_idx]
            mapping = np.full((len(current_frame_markers), 2), np.nan)
            
            for marker_idx in range(len(current_frame_markers)):
                current_marker_idx = marker_idx
                valid_chain = True
                current_frame = frame_idx
                
                while current_frame > 0:
                    prev_frame = current_frame - 1
                    prev_frame_mapping = self.frame_mappings[prev_frame]
                    if current_marker_idx >= len(prev_frame_mapping) or np.isnan(prev_frame_mapping[current_marker_idx][0]):
                        valid_chain = False
                        break
                    prev_frame_marker_idx = int(prev_frame_mapping[current_marker_idx][0])
                    if prev_frame_marker_idx >= len(self.frame_markers[prev_frame]):
                        valid_chain = False
                        break
                    current_marker_idx = prev_frame_marker_idx
                    current_frame = prev_frame
                
                if valid_chain and current_marker_idx < len(base_markers) and current_marker_idx in valid_base_indices:
                    dist = np.linalg.norm(current_frame_markers[marker_idx] - base_markers[current_marker_idx])
                    mapping[marker_idx] = [current_marker_idx, dist]
            
            self.base_frame_mappings.append(mapping)

    def create_visualization(self, mode):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        first_frame = self.frames[0]
        out = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            2.0,
            (first_frame.shape[1], first_frame.shape[0]),
        )

        def draw_markers(frame, markers, color):
            for marker in markers:
                if len(marker) > 0:
                    point = tuple(map(int, marker))
                    cv2.circle(frame, point, 5, color, -1)
            return frame

        def draw_arrows(frame, start_markers, end_markers, mapping):
            for i, map_entry in enumerate(mapping):
                if not np.isnan(map_entry[0]):
                    start_idx = int(map_entry[0])
                    if start_idx < len(start_markers) and i < len(end_markers):
                        start_point = tuple(map(int, start_markers[start_idx]))
                        end_point = tuple(map(int, end_markers[i]))
                        cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 2)
            return frame

        for frame_idx in range(len(self.frames)):
            frame = self.frames[frame_idx].copy()
            
            if mode == "show-adjacent":
                frame = draw_markers(frame, self.frame_markers[frame_idx], (0, 255, 0))
                if frame_idx > 0:
                    prev_frame = self.frames[frame_idx - 1].copy()
                    prev_frame = draw_markers(prev_frame, self.frame_markers[frame_idx - 1], (255, 0, 0))
                    blended = cv2.addWeighted(frame, 0.7, prev_frame, 0.3, 0)
                    blended = draw_arrows(
                        blended,
                        self.frame_markers[frame_idx - 1],
                        self.frame_markers[frame_idx],
                        self.frame_mappings[frame_idx - 1]
                    )
                    out.write(blended)
                else:
                    out.write(frame)

            elif mode == "show-base":
                base_frame = self.frames[0].copy()
                base_frame = draw_markers(base_frame, self.frame_markers[0], (255, 0, 0))
                frame = draw_markers(frame, self.frame_markers[frame_idx], (0, 255, 0))
                blended = cv2.addWeighted(frame, 0.7, base_frame, 0.3, 0)
                if frame_idx > 0 and frame_idx - 1 < len(self.base_frame_mappings):
                    blended = draw_arrows(
                        blended,
                        self.frame_markers[0],
                        self.frame_markers[frame_idx],
                        self.base_frame_mappings[frame_idx - 1]
                    )
                out.write(blended)

            elif mode == "unpaired-markers":
                frame = draw_markers(frame, self.frame_markers[frame_idx], (0, 255, 0))
                out.write(frame)

        out.release()

    def save_paired_markers_to_file(self):
        num_frames = len(self.frame_markers)
        num_markers = len(self.frame_markers[0])
        markers_array = np.zeros((num_frames, num_markers, 2), dtype=np.float32)
        markers_array[0] = self.frame_markers[0]
        for frame_idx in range(1, num_frames):
            current_frame_markers = self.frame_markers[frame_idx]
            base_mapping = self.base_frame_mappings[frame_idx - 1]
            reordered_markers = np.zeros((num_markers, 2), dtype=np.float32)
            for current_idx, map_entry in enumerate(base_mapping):
                if not np.isnan(map_entry[0]):
                    base_frame_idx = int(map_entry[0])
                    reordered_markers[base_frame_idx] = current_frame_markers[
                        current_idx
                    ]
            markers_array[frame_idx] = reordered_markers
        with open(SYSTEM_PARAMS.files.markers_paired, "wb") as f:
            pickle.dump(markers_array, f)

    def process_video(self):
        self.extract_frames()
        self.match_consecutive_frames()
        self.compute_base_frame_mappings()
        self.save_paired_markers_to_file()
        self.create_visualization(mode="show-base")


class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        self.current_frame = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.root = tk.Tk()
        self.root.title("Marker Tracking Viewer")
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()
        self.frame_label = tk.Label(self.root, text=f"Frame: 0/{self.total_frames}")
        self.frame_label.pack()
        self.root.bind("<Left>", self.prev_frame)
        self.root.bind("<Right>", self.next_frame)
        self.root.bind("<Escape>", self.quit)
        self.show_frame()

    def show_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            self.canvas.config(width=img.width, height=img.height)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo
            self.frame_label.config(
                text=f"Frame: {self.current_frame}/{self.total_frames}"
            )

    def next_frame(self, event):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame()

    def prev_frame(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()

    def quit(self, event):
        self.root.quit()

    def run(self):
        self.root.mainloop()
        self.cap.release()


def process_and_view_video(input_path, output_path):
    tracker = MarkerTracker(input_path, output_path)
    tracker.process_video()
    player = VideoPlayer(tracker.output_path)
    player.run()


def main():
    process_and_view_video(
        SYSTEM_PARAMS.files.system_id_video, SYSTEM_PARAMS.files.system_id_output_video
    )
    player = VideoPlayer(SYSTEM_PARAMS.files.system_id_output_video)
    player.run()
