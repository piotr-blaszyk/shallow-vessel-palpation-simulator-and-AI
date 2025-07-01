import cv2
import numpy as np
import matplotlib.pyplot as plt
from fisheye_model import *
import tkinter as tk
from PIL import Image, ImageTk
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import json
import pickle

from difftactile.main.constants import *

class MarkerTracker:
    def __init__(self, video_path, output_path=None):
        """
        Initialize the marker tracker with input video path and optional output path.
        
        Args:
            video_path (str): Path to the input video file
            output_path (str, optional): Path for output video file
        """
        self.fisheye_model = FisheyeModel()
        self.video_path = video_path
        self.output_path = None
        self.frame_markers = []  # List to store markers for each frame
        self.frame_mappings = []  # List to store mappings between consecutive frames
        self.base_frame_mappings = []  # List to store mappings to frame 0
        self.frames = []  # List to store actual frames
        
        # LK optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
    def extract_frames(self, calculate_markers=True):
        """
        Extract frames from video at 1.0 second intervals.
        
        Args:
            calculate_markers (bool): If True, calculate markers from frames. If False, load from json file.
        """
        json_path = SYSTEM_PARAMS.files.marker_tracker_markers
        manual_annotations_path = SYSTEM_PARAMS.files.marker_tracker_manual_annotations
        
        # Load manual annotations if they exist
        manual_annotations = {}
        if manual_annotations_path.exists():
            with open(manual_annotations_path, 'r') as f:
                manual_annotations = json.load(f)
                print(f"Loaded manual annotations from {manual_annotations_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1.0)  # Number of frames to skip for 1.0s interval
        
        frame_count = 0
        frame_idx = 0  # Keep track of processed frame index
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(frame)
                markers, _, _ = self.fisheye_model.get_marker_image(gray)
                
                if len(markers) > 0:  # Only append if markers were detected
                    self.frame_markers.append(markers)
                else:
                    self.frame_markers.append(np.array([]))
                    
                frame_idx += 1
                
            frame_count += 1
            
        cap.release()
        
        if not calculate_markers:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    markers_list = json.load(f)
                    # Convert the loaded list back to numpy arrays
                    self.frame_markers = [np.array(markers) for markers in markers_list]
                                
                    print(f"Loaded markers from {json_path}")
                return
            else:
                print(f"Warning: {json_path} not found")
        
        if True:
            # Convert numpy arrays to lists for JSON serialization
            markers_list = [markers.tolist() if markers.size > 0 else [] for markers in self.frame_markers]
            # Save markers to json file
            with open(json_path, 'w') as f:
                json.dump(markers_list, f)
                print(f"Saved markers to {json_path}")

        # Add manual annotations to loaded markers
        for frame_idx in manual_annotations:
            idx = int(frame_idx)
            if idx < len(self.frame_markers):
                manual_points = np.array(manual_annotations[frame_idx])
                if len(self.frame_markers[idx]) > 0:
                    self.frame_markers[idx] = np.vstack((self.frame_markers[idx], manual_points))
                else:
                    self.frame_markers[idx] = manual_points
        
        print(f"Added manual annotations")
    
    def match_consecutive_frames(self):
        """Match markers between consecutive frames using the Hungarian algorithm (linear_sum_assignment), minimizing sum of squared distances."""
        for i in range(len(self.frame_markers) - 1):
            current_markers = self.frame_markers[i]
            next_markers = self.frame_markers[i + 1]

            if len(current_markers) == 0 or len(next_markers) == 0:
                # No markers to match
                mapping = np.full((len(next_markers), 2), np.nan)
                self.frame_mappings.append(mapping)
                continue

            # Compute cost matrix (squared Euclidean distances)
            cost_matrix = cdist(next_markers, current_markers, metric='sqeuclidean')

            # Hungarian algorithm for optimal assignment (minimize sum of squared distances)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Initialize mapping matrix with NaN
            mapping = np.full((len(next_markers), 2), np.nan)

            # Fill mapping with assignments, store original (unsquared) distance
            for r, c in zip(row_ind, col_ind):
                dist = np.linalg.norm(next_markers[r] - current_markers[c])
                mapping[r] = [c, dist]

            self.frame_mappings.append(mapping)

    def compute_base_frame_mappings(self):
        """Compute mappings between each frame and frame 0."""
        base_markers = self.frame_markers[0]
        
        for frame_idx in range(1, len(self.frame_markers)):
            # Initialize mapping with NaN
            mapping = np.full((len(self.frame_markers[frame_idx]), 2), np.nan)
            
            # Track each marker through the chain of mappings
            for marker_idx in range(len(self.frame_markers[frame_idx])):
                current_marker_idx = marker_idx
                valid_chain = True
                
                # Follow the chain of mappings back to frame 0
                for prev_frame_idx in range(frame_idx - 1, -1, -1):
                    if np.isnan(self.frame_mappings[prev_frame_idx][current_marker_idx][0]):
                        valid_chain = False
                        break
                    current_marker_idx = int(self.frame_mappings[prev_frame_idx][current_marker_idx][0])
                
                if valid_chain:
                    mapping[marker_idx] = [current_marker_idx, 
                                         np.linalg.norm(self.frame_markers[frame_idx][marker_idx] - 
                                                      base_markers[current_marker_idx])]
                    
            self.base_frame_mappings.append(mapping)
        
    def create_visualization(self, show_adjacent=False):
        """Create visualization video with marker tracking.
        Args:
            show_adjacent (bool): If True, visualize adjacent frames (n-1 and n). If False, visualize base frame and frame n.
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        first_frame = self.frames[0]
        out = cv2.VideoWriter(str(self.output_path), fourcc, 2.0, 
                            (first_frame.shape[1], first_frame.shape[0]))
        
        if show_adjacent:
            for frame_idx in range(len(self.frames)):
                frame = self.frames[frame_idx].copy()

                # Draw current frame markers in green
                for marker in self.frame_markers[frame_idx]:
                    cv2.circle(frame, tuple(map(int, marker)), 5, (0, 255, 0), -1)

                if frame_idx > 0:
                    prev_frame = self.frames[frame_idx - 1].copy()
                    # Draw previous frame markers in blue
                    for marker in self.frame_markers[frame_idx - 1]:
                        cv2.circle(prev_frame, tuple(map(int, marker)), 5, (255, 0, 0), -1)
                    # Create blended image
                    blended = cv2.addWeighted(frame, 0.7, prev_frame, 0.3, 0)
                    # Draw displacement arrows between consecutive frames
                    mapping = self.frame_mappings[frame_idx - 1]
                    for i, map_entry in enumerate(mapping):
                        if not np.isnan(map_entry[0]):
                            start_point = tuple(map(int, self.frame_markers[frame_idx - 1][int(map_entry[0])]))
                            end_point = tuple(map(int, self.frame_markers[frame_idx][i]))
                            cv2.arrowedLine(blended, start_point, end_point, (0, 0, 255), 2)
                    out.write(blended)
                else:
                    # For the first frame, just write the frame with its markers
                    out.write(frame)
        else:
            # Load the paired markers data
            with open(SYSTEM_PARAMS.files.markers_paired, 'rb') as f:
                markers_array = pickle.load(f)
            
            for frame_idx in range(len(self.frames)):
                frame = self.frames[frame_idx].copy()
                base_frame = self.frames[0].copy()
                
                # Draw base frame markers in blue with indices
                for marker_idx, marker in enumerate(markers_array[0]):
                    point = tuple(map(int, marker))
                    cv2.circle(base_frame, point, 5, (255, 0, 0), -1)
                    cv2.putText(base_frame, str(marker_idx), 
                              (point[0] + 10, point[1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw current frame markers in green with indices
                for marker_idx, marker in enumerate(markers_array[frame_idx]):
                    point = tuple(map(int, marker))
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(marker_idx), 
                              (point[0] + 10, point[1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Create blended image
                blended = cv2.addWeighted(frame, 0.7, base_frame, 0.3, 0)
                
                # Draw displacement arrows from base frame to current frame
                if frame_idx > 0:
                    for marker_idx in range(len(markers_array[0])):
                        start_point = tuple(map(int, markers_array[0][marker_idx]))
                        end_point = tuple(map(int, markers_array[frame_idx][marker_idx]))
                        cv2.arrowedLine(blended, start_point, end_point, (0, 0, 255), 2)
                
                out.write(blended)
        
        out.release()
        
    def save_paired_markers_to_file(self):
        """
        Saves the paired marker coordinates to a pickle file.
        The data is transformed into a 3D numpy array with shape (num_frames, num_markers_per_frame, 2).
        The markers in frames 1 to n are reordered to match the order of markers in the base frame.
        """
        num_frames = len(self.frame_markers)
        num_markers = len(self.frame_markers[0])  # Number of markers in base frame
        
        # Initialize the 3D array
        markers_array = np.zeros((num_frames, num_markers, 2), dtype=np.float32)
        
        # Fill base frame (frame 0) markers directly
        markers_array[0] = self.frame_markers[0]
        
        # For each subsequent frame, reorder markers based on base frame mapping
        for frame_idx in range(1, num_frames):
            current_frame_markers = self.frame_markers[frame_idx]
            base_mapping = self.base_frame_mappings[frame_idx - 1]  # -1 because base_frame_mappings starts from frame 1
            
            # Create reordered markers array for this frame
            reordered_markers = np.zeros((num_markers, 2), dtype=np.float32)
            
            # For each marker in the current frame
            for current_idx, map_entry in enumerate(base_mapping):
                if not np.isnan(map_entry[0]):
                    base_frame_idx = int(map_entry[0])
                    # Place the current marker in the position corresponding to its base frame match
                    reordered_markers[base_frame_idx] = current_frame_markers[current_idx]
            
            markers_array[frame_idx] = reordered_markers
        
        # Save to pickle file
        with open(SYSTEM_PARAMS.files.markers_paired, 'wb') as f:
            pickle.dump(markers_array, f)

    def process_video(self):
        """Process the video file through all steps."""
        print("Extracting frames...")
        self.extract_frames()
        print("Matching consecutive frames...")
        self.match_consecutive_frames()
        print("Computing base frame mappings using Lucas-Kanade optical flow...")
        self.compute_base_frame_mappings()
        print("Saving markers to file")
        self.save_paired_markers_to_file()
        print("Creating visualization...")
        self.create_visualization()
        print("Processing complete!")

class VideoPlayer:
    def __init__(self, video_path):
        """Initialize video player with the given video path."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        self.current_frame = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store clicked points
        self.manual_annotations = {}  # frame_idx -> list of [x, y] coordinates
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Marker Tracking Viewer")
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()
        
        # Add frame counter label
        self.frame_label = tk.Label(self.root, text=f"Frame: 0/{self.total_frames}")
        self.frame_label.pack()
        
        # Bind keyboard events
        self.root.bind('<Left>', self.prev_frame)
        self.root.bind('<Right>', self.next_frame)
        self.root.bind('<Escape>', self.quit)
        if False:
            self.root.bind('y', self.save_annotations)  # Add binding for 'y' key
        
        # Bind mouse click event
        self.canvas.bind('<Button-1>', self.on_click)
        
        # Show first frame
        self.show_frame()
        
    def show_frame(self):
        """Display the current frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas size if needed
            self.canvas.config(width=img.width, height=img.height)
            
            # Update image
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep reference
            
            # Update frame counter
            self.frame_label.config(text=f"Frame: {self.current_frame}/{self.total_frames}")
    
    def next_frame(self, event):
        """Show next frame."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame()
    
    def prev_frame(self, event):
        """Show previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()
    
    def quit(self, event):
        """Close the video player."""
        self.root.quit()
    
    def on_click(self, event):
        """Handle mouse click event."""
        x, y = event.x, event.y
        print(f"Frame: {self.current_frame}, Clicked coordinates: ({x}, {y})")
        
        # Store the clicked coordinates
        if self.current_frame not in self.manual_annotations:
            self.manual_annotations[self.current_frame] = []
        self.manual_annotations[self.current_frame].append([x, y])
    
    def save_annotations(self, event):
        """Save manual annotations to a JSON file, merging with existing annotations if present."""
        # Convert frame indices from int to str for JSON serialization
        annotations_dict = {str(k): v for k, v in self.manual_annotations.items()}
        
        json_path = SYSTEM_PARAMS.files.marker_tracker_manual_annotations
        
        # Load existing annotations if they exist
        existing_annotations = {}
        if json_path.exists():
            with open(json_path, 'r') as f:
                existing_annotations = json.load(f)
                print(f"Loaded existing annotations from {json_path}")
        
        # Merge existing annotations with new ones
        # For each frame in new annotations, append the new points to existing points or create new entry
        for frame_idx, points in annotations_dict.items():
            if frame_idx in existing_annotations:
                existing_annotations[frame_idx].extend(points)
            else:
                existing_annotations[frame_idx] = points
        
        # Save merged annotations back to file
        with open(json_path, 'w') as f:
            json.dump(existing_annotations, f)
        print(f"Saved merged annotations to {json_path}")
        
        # Clear current annotations after saving
        self.manual_annotations = {}
    
    def run(self):
        """Start the video player."""
        self.root.mainloop()
        self.cap.release()

def process_and_view_video(input_path, output_path=None):
    """
    Process a video file and launch the interactive viewer.
    
    Args:
        input_path (str): Path to input video file
        output_path (str, optional): Path for output video file
    """
    # Process video
    tracker = MarkerTracker(input_path, output_path)
    tracker.process_video()
    
    # Launch viewer
    player = VideoPlayer(tracker.output_path)
    player.run()


if __name__ == '__main__':
    process_and_view_video(SYSTEM_PARAMS.files.system_id_video, SYSTEM_PARAMS.files.system_id_output_video)

    player = VideoPlayer(SYSTEM_PARAMS.files.system_id_output_video)
    player.run()
