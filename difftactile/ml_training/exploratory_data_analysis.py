import pickle
import numpy as np
import cv2

def load_sim_data():
    with open("../tasks/output/marker_snapshots_and_labels.pkl", "rb") as f:
        data = pickle.load(f)
    predict_markers_snapshots = data["predict_markers_snapshots"]
    virtual_markers_snapshots = data["virtual_markers_snapshots"]
    displacements = predict_markers_snapshots - virtual_markers_snapshots
    ground_truth_labels = data["ground_truth_labels"]

    if True:
        print("predict_markers_snapshots shape:", predict_markers_snapshots.shape)
        print("virtual_markers_snapshots shape:", virtual_markers_snapshots.shape)
        print("ground_truth_labels shape:", ground_truth_labels.shape)
    
    return predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels

def visualize_markers(predict_markers, virtual_markers, ground_truth_labels):
    img_height = 480
    img_width = 640
    current_idx = 0
    num_snapshots = predict_markers.shape[0]

    while True:
        # Create a blank black image
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Scale markers to image dimensions
        pred_markers = predict_markers[current_idx].copy()
        virt_markers = virtual_markers[current_idx].copy()

        # Draw virtual markers as dots (red)
        for point in virt_markers:
            cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), 2)

        # Draw displacement arrows
        displacements = pred_markers - virt_markers
        for start, displacement in zip(pred_markers, displacements):
            end = start + displacement
            cv2.arrowedLine(img, 
                          (int(start[0]), int(start[1])),
                          (int(end[0]), int(end[1])),
                          (0, 165, 255), 2)  # Orange arrows

        # Add text for snapshot index and ground truth label
        cv2.putText(img, f"Snapshot: {current_idx}/{num_snapshots-1}", 
                   (10, img_height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Label: {ground_truth_labels[current_idx]}", 
                   (10, img_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the image
        cv2.imshow('Marker Displacement Visualization', img)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('j'):  # Previous snapshot
            current_idx = (current_idx - 1) % num_snapshots
        elif key == ord('l'):  # Next snapshot
            current_idx = (current_idx + 1) % num_snapshots
        elif key == ord('q'):  # Quit
            break

    cv2.destroyAllWindows()

def main():
    predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels = load_sim_data()
    visualize_markers(predict_markers_snapshots, virtual_markers_snapshots, ground_truth_labels)

if __name__ == "__main__":
    main()
