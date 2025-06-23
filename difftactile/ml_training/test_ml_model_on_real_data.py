import torch
import numpy as np
import pickle
from fully_connected_neural_network import TumorClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

def load_and_preprocess_data():
    # Load experimental data
    with open('../sensor_model/vascular-tumour-press-experimental-results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Load initial marker positions
    with open('../sensor_model/init-marker-positions.pkl', 'rb') as f:
        init_marker_positions = pickle.load(f)
    
    marker_positions = data['marker_positions']  # Shape: (num_points, num_markers, 2)
    class_labels = data['class_labels']  # Shape: (num_points,)
    
    # Scale marker positions from 1920x1080 to 640x480
    scale_x = 640 / 1920
    scale_y = 480 / 1080
    marker_positions[..., 0] *= scale_x  # Scale x coordinates
    marker_positions[..., 1] *= scale_y  # Scale y coordinates
    
    # Calculate displacements relative to initial positions
    # Expand init_marker_positions to match marker_positions shape
    init_markers_expanded = np.expand_dims(init_marker_positions, axis=0)  # Shape: (1, num_markers, 2)
    init_markers_expanded = np.repeat(init_markers_expanded, marker_positions.shape[0], axis=0)  # Shape: (num_points, num_markers, 2)
    
    # Calculate displacements
    displacements = marker_positions - init_markers_expanded  # Shape: (num_points, num_markers, 2)
    
    # Map labels: {0, 4} -> 0; {3, 5} -> 1; discard others
    valid_mask = np.isin(class_labels, [0, 3, 4, 5])
    displacements = displacements[valid_mask]
    class_labels = class_labels[valid_mask]
    
    # Map labels
    new_labels = np.zeros_like(class_labels)
    new_labels[np.isin(class_labels, [3, 5])] = 1
    
    # Flatten displacements to match training data format
    X = displacements.reshape(displacements.shape[0], -1)  # Shape: [n, num_markers*2]
    y = new_labels
    
    print(f"Processed data shape: {X.shape}")
    print(f"Number of samples per class: {np.bincount(y)}")
    print(f"Marker positions scaled from 1920x1080 to 640x480")
    print(f"Using displacements relative to initial marker positions as input")
    
    return X, y

def main():
    # Load preprocessing parameters
    with open('saved_models/preprocessing_params.pkl', 'rb') as f:
        params = pickle.load(f)
    input_dim = params['input_dim']
    scaler = params['scaler']
    
    # Initialize and load model
    model = TumorClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load('saved_models/tumor_classifier_weights.pt'))
    model.eval()
    
    # Load and preprocess experimental data
    X, y = load_and_preprocess_data()
    
    # Apply the same scaling as training
    X_scaled = scaler.transform(X)
    
    # Create dataset and dataloader
    test_dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize trainer and test
    trainer = pl.Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Clear any previous predictions and labels
    model.test_predictions = []
    model.test_labels = []
    
    # Run test
    print("\nTesting model on experimental data:")
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
