import pickle
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

class TumorClassifier(pl.LightningModule):
    def __init__(self, input_dim=254):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, x):
        return torch.sigmoid(self.layers(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(y_hat, y.int()), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
    with open("output/marker_snapshots_and_labels.pkl", "rb") as f:
        data = pickle.load(f)
    predict_markers_snapshots = data["predict_markers_snapshots"]
    virtual_markers_snapshots = data["virtual_markers_snapshots"]
    ground_truth_labels = data["ground_truth_labels"]
    print("predict_markers_snapshots shape:", predict_markers_snapshots.shape)
    print("virtual_markers_snapshots shape:", virtual_markers_snapshots.shape)
    print("ground_truth_labels shape:", ground_truth_labels.shape)

    # sys.exit()
    
    # Launch the visualization
    if False:
        visualize_markers(predict_markers_snapshots, virtual_markers_snapshots, ground_truth_labels)

    # Preprocess your data first (outside this script)
    # displacements = deformed_marker_snapshots - default_marker_positions
    displacements = predict_markers_snapshots - virtual_markers_snapshots
    X = displacements.reshape(displacements.shape[0], -1)  # Shape: [n, 254]
    y = ground_truth_labels  # Shape: [n]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Train the model
    model = TumorClassifier()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
