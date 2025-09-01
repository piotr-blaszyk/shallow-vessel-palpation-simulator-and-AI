import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from difftactile.main.constants import *


def load_data():
    # Load experimental data
    with open(SYSTEM_PARAMS.files.vascular_tumour_press_results, 'rb') as f:
        data = pickle.load(f)
    
    # Load initial marker positions
    with open(SYSTEM_PARAMS.files.init_marker_positions, 'rb') as f:
        init_marker_positions = pickle.load(f)
    
    marker_positions = data['marker_positions']  # Shape: (num_points, num_markers, 2)
    class_labels = data['class_labels']  # Shape: (num_points,)
    
    # Scale marker positions from 1920x1080 to 640x480
    scale_x = 640 / 1920
    scale_y = 480 / 1080
    marker_positions[..., 0] *= scale_x
    marker_positions[..., 1] *= scale_y
    
    # Process labels as in test_ml_model_on_real_data.py
    valid_mask = np.isin(class_labels, [0, 3, 4, 5])
    marker_positions = marker_positions[valid_mask]
    class_labels = class_labels[valid_mask]
    
    # Map labels
    new_labels = np.zeros_like(class_labels)
    new_labels[np.isin(class_labels, [3, 5])] = 1
    
    return marker_positions, init_marker_positions, new_labels

def compute_displacement_metrics(current_markers, init_markers, row_ind, col_ind):
    """Compute displacement vectors and their magnitudes."""
    displacements = current_markers[row_ind] - init_markers[col_ind]
    magnitudes = np.linalg.norm(displacements, axis=1)
    return displacements, magnitudes

def compute_histogram_bins(marker_positions, init_markers):
    """Compute histogram bins based on all frames."""
    all_magnitudes = []
    
    for frame_markers in marker_positions:
        row_ind, col_ind = match_markers(frame_markers, init_markers)
        _, magnitudes = compute_displacement_metrics(frame_markers, init_markers, row_ind, col_ind)
        all_magnitudes.extend(magnitudes)
    
    min_mag = min(all_magnitudes)
    max_mag = max(all_magnitudes)
    
    bins = np.linspace(min_mag, max_mag, 31)  # 31 edges for 30 bins
    return bins

def compute_histogram_entropy(magnitudes, bins):
    """Compute histogram and its Shannon entropy."""
    hist, _ = np.histogram(magnitudes, bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return entropy(hist)

def extract_features(marker_positions, init_markers, hist_bins):
    """Extract features (mean x-displacement, mean y-displacement, entropy) for all frames."""
    features = []
    
    for frame_markers in marker_positions:
        row_ind, col_ind = match_markers(frame_markers, init_markers)
        displacements, magnitudes = compute_displacement_metrics(frame_markers, init_markers, row_ind, col_ind)
        
        # Compute mean displacements
        mean_displacement = np.mean(displacements, axis=0)
        
        # Compute entropy
        entropy_val = compute_histogram_entropy(magnitudes, hist_bins)
        
        features.append([mean_displacement[0], mean_displacement[1], entropy_val])
    
    return np.array(features)

def train_and_evaluate_knn():
    """Train and evaluate k-NN classifier using leave-one-out cross-validation."""
    # Load and prepare data
    marker_positions, init_markers, labels = load_data()
    hist_bins = compute_histogram_bins(marker_positions, init_markers)
    
    # Extract features
    features = extract_features(marker_positions, init_markers, hist_bins)
    
    # Initialize metrics storage
    all_confusion_matrices = []
    all_f1_scores = []
    all_accuracies = []
    all_recalls = []
    all_predictions = []
    
    # Perform leave-one-out cross-validation
    loo = LeaveOneOut()
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(features), 1):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test)
        all_predictions.extend(y_pred)
        
        if fold % 5 == 0:  # Print progress every 5 folds
            print(f"Processed fold {fold}/{len(features)}")
    
    # Calculate final metrics using all predictions
    final_conf_matrix = confusion_matrix(labels, all_predictions)
    final_f1 = f1_score(labels, all_predictions)
    final_accuracy = accuracy_score(labels, all_predictions)
    final_recall = recall_score(labels, all_predictions)
    
    print("\nFinal Results (Leave-One-Out Cross-Validation):")
    print(f"Confusion Matrix:\n{final_conf_matrix}")
    print(f"F1 Score: {final_f1:.3f}")
    print(f"Accuracy: {final_accuracy:.3f}")
    print(f"Recall: {final_recall:.3f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(final_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Leave-One-Out Cross-Validation)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def match_markers(current_markers, init_markers):
    """Match markers using the Hungarian algorithm."""
    cost_matrix = cdist(current_markers, init_markers, metric='sqeuclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def draw_frame(frame_idx, marker_positions, init_markers, hist_bins):
    """Draw current frame with markers, displacement arrows, and metrics."""
    # Create a larger image to accommodate histogram
    img = np.zeros((680, 640, 3), dtype=np.uint8)
    
    current_markers = marker_positions[frame_idx]
    
    # Match markers and compute metrics
    row_ind, col_ind = match_markers(current_markers, init_markers)
    displacements, magnitudes = compute_displacement_metrics(current_markers, init_markers, row_ind, col_ind)
    
    # Compute histogram and entropy
    hist, _ = np.histogram(magnitudes, bins=hist_bins)
    entropy_score = compute_histogram_entropy(magnitudes, hist_bins)
    
    # Draw markers and arrows in the top part
    # Draw initial markers (red)
    for pos in init_markers:
        cv2.circle(img, tuple(pos.astype(int)), 3, (0, 0, 255), -1)
    
    # Draw current markers (green) and displacement arrows (blue)
    for curr_idx, init_idx in zip(row_ind, col_ind):
        curr_pos = tuple(current_markers[curr_idx].astype(int))
        init_pos = tuple(init_markers[init_idx].astype(int))
        
        cv2.circle(img, curr_pos, 3, (0, 255, 0), -1)
        cv2.arrowedLine(img, init_pos, curr_pos, (255, 0, 0), 1, tipLength=0.2)
    
    # Draw histogram in the bottom part
    hist_height = 150
    hist_normalized = hist * (hist_height / hist.max())
    bin_width = 600 // len(hist)
    
    for i, h in enumerate(hist_normalized):
        x = i * bin_width + 20
        y = 650
        cv2.rectangle(img, (x, y), (x + bin_width - 2, y - int(h)), (0, 255, 255), -1)
    
    # Add text information
    cv2.putText(img, f'Frame: {frame_idx}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Entropy: {entropy_score:.3f}', (10, 520), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    mean_displacement = np.mean(displacements, axis=0)
    cv2.putText(img, f'Mean X-disp: {mean_displacement[0]:.2f}', (10, 550), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'Mean Y-disp: {mean_displacement[1]:.2f}', (10, 580), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img

def compute_and_print_sorted_magnitudes(marker_positions, init_markers):
    """Compute and print mean deformation magnitudes for each frame in real-world coordinates."""
    # Array to store mean magnitudes for each frame
    mean_magnitudes = []
    
    for frame_idx, frame_markers in enumerate(marker_positions):
        row_ind, col_ind = match_markers(frame_markers, init_markers)
        
        # Convert to real-world coordinates
        real_frame_markers = frame_markers.copy()
        
        real_init_markers = init_markers.copy()
        
        # Compute displacements in real-world coordinates
        displacements = real_frame_markers[row_ind] - real_init_markers[col_ind]
        magnitudes = np.linalg.norm(displacements, axis=1)
        
        # Compute mean magnitude for this frame
        mean_mag = np.mean(magnitudes)
        mean_magnitudes.append((frame_idx, mean_mag))
    
    # Sort by magnitude
    sorted_magnitudes = sorted(mean_magnitudes, key=lambda x: x[1])
    
    print("\nMean Deformation Magnitudes (in pixels, sorted ascending):")
    print("Format: (Frame Index, Mean Magnitude)")
    for frame_idx, mag in sorted_magnitudes:
        print(f"Frame {frame_idx}: {mag:.2f}")
    
    return sorted_magnitudes

def main():
    # First, compute and print sorted magnitudes
    marker_positions, init_markers, labels = load_data()
    compute_and_print_sorted_magnitudes(marker_positions, init_markers)
    
    # Then proceed with training and visualization
    train_and_evaluate_knn()
    
    # Visualization
    num_frames = len(marker_positions)
    frame_idx = 0
    
    hist_bins = compute_histogram_bins(marker_positions, init_markers)
    
    print("\nVisualization Controls:")
    print("j: Previous frame")
    print("l: Next frame")
    print("q: Quit")
    
    while True:
        img = draw_frame(frame_idx, marker_positions, init_markers, hist_bins)
        cv2.imshow('Marker Matching Visualization', img)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('j'):  # Previous frame
            frame_idx = (frame_idx - 1) % num_frames
        elif key == ord('l'):  # Next frame
            frame_idx = (frame_idx + 1) % num_frames
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
