import cv2

from difftactile.main.constants import *

# Input and output video paths
input_video_path = SYSTEM_PARAMS.files.system_id_video
output_video_path = SYSTEM_PARAMS.files.processed_video

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# We want 1 frame per second in the output
target_fps = 1
frames_to_skip = original_fps - 1  # Keep 1 frame, skip the rest in each second

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height), isColor=False)

print(f"Processing video with dimensions {frame_width}x{frame_height}")
print(f"Original FPS: {original_fps}, Target FPS: {target_fps}")
print(f"Total frames to process: {total_frames}")

frame_count = 0
saved_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Only process frames at 1-second intervals
    if frame_count % original_fps == 0:
        # Convert frame to grayscale if it's not already
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, processed_frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        
        # Write the processed frame
        out.write(processed_frame)
        saved_frames += 1
        
        if saved_frames % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames, Saved {saved_frames} frames")
    
    frame_count += 1

# Release everything
cap.release()
out.release()

print(f"Video processing completed! Saved {saved_frames} frames at 1 FPS") 