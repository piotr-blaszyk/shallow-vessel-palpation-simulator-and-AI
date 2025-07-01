import numpy as np
import cv2
import pickle
import signal
import sys

from ..tasks.constants import *

def signal_handler(sig, frame):
    print('\nClosing visualization...')
    cv2.destroyAllWindows()
    sys.exit(0)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Load the marker positions from pickle file
with open(SYSTEM_PARAMS.files.sim_markers_initial_positions, 'rb') as f:
    marker_positions = pickle.load(f)

# Read the image
image = cv2.imread(SYSTEM_PARAMS.files.vitactip_photo_default_state)

# Draw markers on the image
marker_color = (0, 0, 255)  # Red color in BGR
marker_size = 5
marker_thickness = -1  # Filled circle

for position in marker_positions:
    # Convert position to integer coordinates
    x, y = position.astype(int)
    cv2.circle(image, (x, y), marker_size, marker_color, marker_thickness)

# Display the image
cv2.imshow('Markers Visualization', image)

print("Press Ctrl+C in terminal to exit...")
while True:
    # Update the window with a short wait
    if cv2.waitKey(100) & 0xFF == 27:  # Also allow ESC key to exit
        break

cv2.destroyAllWindows() 