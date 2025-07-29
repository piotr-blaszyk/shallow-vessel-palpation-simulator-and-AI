import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle

from exploratory_data_analysis import *
from difftactile.main.constants import *

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
        self.test_accuracy = Accuracy(task="binary")
        self.test_predictions = []
        self.test_labels = []

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Store predictions and labels for confusion matrix
        predictions = (y_hat > 0.5).cpu().numpy().astype(int)
        labels = y.cpu().numpy().astype(int)
        
        # Print batch-level debugging information
        # print(f"Batch {batch_idx}: Processing {len(predictions)} samples")
        
        self.test_predictions.extend(predictions)
        self.test_labels.extend(labels)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy(y_hat, y.int()), prog_bar=True)

    def on_test_end(self):
        # Convert lists to numpy arrays
        predictions = np.array(self.test_predictions, dtype=int)
        labels = np.array(self.test_labels, dtype=int)
        
        # Print debugging information
        print("\nTest Set Statistics:")
        print(f"Total samples: {len(predictions)}")
        print(f"Predicted positives: {np.sum(predictions)}")
        print(f"Actual positives: {np.sum(labels)}")
        print(f"Unique predictions: {np.unique(predictions)}")
        print(f"Unique labels: {np.unique(labels)}")
        
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
        
        # Calculate confusion matrix with explicit labels
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        
        # Calculate percentages
        cm_percentage = cm.astype('float') / cm.sum() * 100
        
        # Create a more readable display
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
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)'
                ax.text(j, i, text,
                       ha="center", va="center",
                       color="cyan")
        
        plt.tight_layout()
        plt.show()
        
        # Clear the stored predictions and labels
        self.test_predictions = []
        self.test_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main():
    predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels = load_sim_data()
    X = displacements.reshape(displacements.shape[0], -1)  # Shape: [n, 254]
    y = ground_truth_labels  # Shape: [n]

    # First split: separate test set (80:20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: split remaining data into train and validation (75:25, which gives us 60:20:20 overall)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

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
    
    # Save the trained model and scaler separately
    os.makedirs('saved_models', exist_ok=True)
    # Save model state dict
    torch.save(model.state_dict(), 'saved_models/tumor_classifier_weights.pt')
    # Save scaler and input dimension separately
    with open(SYSTEM_PARAMS.files.preprocessing_params, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'input_dim': X.shape[1]
        }, f)
    print("Model weights saved to saved_models/tumor_classifier_weights.pt")
    print("Preprocessing parameters saved")
    
    # Test the model and display confusion matrix
    print(f"\nTest dataset size: {len(test_dataset)}")
    model.test_predictions = []  # Clear any previous predictions
    model.test_labels = []  # Clear any previous labels
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
