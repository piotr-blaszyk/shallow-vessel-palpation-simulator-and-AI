import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from exploratory_data_analysis import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Recall

from difftactile.main.constants import *


class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (2, 32, 32)
            nn.Conv2d(2, 16, 3, padding=1),  # 16x32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x16
            
            nn.Conv2d(16, 32, 3, padding=1),  # 32x16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x8x8
            
            nn.Conv2d(32, 64, 3, padding=1),  # 64x8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x4x4
            
            nn.Flatten(),
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1)
        )
        # Initialize metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_confusion = ConfusionMatrix(task="binary", num_classes=2)
        
        # Store predictions and labels for confusion matrix
        self.test_predictions = []
        self.test_labels = []

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy(y_hat, y.int()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(y_hat, y.int()), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Store predictions and labels for confusion matrix
        predictions = (y_hat > 0.5).cpu().numpy().astype(int)
        labels = y.cpu().numpy().astype(int)
        
        self.test_predictions.extend(predictions)
        self.test_labels.extend(labels)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy(y_hat, y.int()), prog_bar=True)
        self.log('test_f1', self.test_f1(y_hat, y.int()), prog_bar=True)
        self.log('test_recall', self.test_recall(y_hat, y.int()), prog_bar=True)

    def on_test_end(self):
        # Convert lists to numpy arrays
        predictions = np.array(self.test_predictions, dtype=int)
        labels = np.array(self.test_labels, dtype=int)
        
        # Print test set statistics
        print("\nTest Set Statistics:")
        print(f"Total samples: {len(predictions)}")
        print(f"Predicted positives: {np.sum(predictions)}")
        print(f"Actual positives: {np.sum(labels)}")
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\nMetrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
        # Calculate and visualize confusion matrix
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        cm_percentage = cm.astype('float') / cm.sum() * 100
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Reds')
        
        # Add title and labels
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add class labels
        classes = ['No Tumor', 'Tumor']
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)'
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="cyan")
        
        plt.tight_layout()
        plt.show()
        
        # Clear stored predictions and labels
        self.test_predictions = []
        self.test_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

# Preprocessing: Convert marker displacements to heatmaps
def displacements_to_heatmaps(displacements, default_positions, grid_size=32):
    """
    Convert marker displacements to 2-channel heatmaps
    Channel 0: x-displacement magnitude
    Channel 1: y-displacement magnitude
    """
    heatmaps = np.zeros((len(displacements), 2, grid_size, grid_size))
    
    # Normalize positions to [0, 1]
    min_vals = default_positions.min(axis=0)
    max_vals = default_positions.max(axis=0)
    norm_pos = (default_positions - min_vals) / (max_vals - min_vals)
    # Convert positions to grid coordinates
    grid_pos = (norm_pos * (grid_size - 1)).astype(int)

    for i, disp in enumerate(displacements):
        # Compute displacement magnitude
        mag = np.linalg.norm(disp, axis=1)
        # Assign to heatmap channels
        for j, (x, y) in enumerate(grid_pos):
            heatmaps[i, 0, x, y] = disp[j, 0]  # x-displacement
            heatmaps[i, 1, x, y] = disp[j, 1]  # y-displacement
            
    return heatmaps

def main():
    predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels = load_sim_data()

    # Convert to heatmaps
    X = displacements_to_heatmaps(displacements, virtual_markers_snapshots[0], grid_size=32)

    # First split: training vs. (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, ground_truth_labels, test_size=0.4, random_state=42)
    
    # Second split: validation vs. test (each gets half of the remaining 40%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Train the model
    model = CNNClassifier()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Train and validate
    trainer.fit(model, train_loader, val_loader)
    
    # Save the trained model and preprocessing parameters
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), 'saved_models/cnn_classifier_weights.pt')
    
    # Save preprocessing parameters
    with open(SYSTEM_PARAMS.cnn_preprocessing_params, 'wb') as f:
        pickle.dump({
            'grid_size': 32,
            'input_channels': 2,  # Number of input channels (x and y displacements)
            'default_positions': virtual_markers_snapshots[0]  # Save reference positions
        }, f)
    
    print("Model weights saved to saved_models/cnn_classifier_weights.pt")
    print("Preprocessing parameters saved")
    
    # Test the model and display confusion matrix
    print(f"\nTest dataset size: {len(test_dataset)}")
    model.test_predictions = []  # Clear any previous predictions
    model.test_labels = []  # Clear any previous labels
    trainer.test(model, test_loader)

if __name__ == '__main__':
    main()