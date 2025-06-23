import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Recall
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from exploratory_data_analysis import *

class GNNClassifier(pl.LightningModule):
    def __init__(self, node_dim=4):
        super().__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        # Initialize metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_recall = Recall(task="binary")
        
        # Store predictions for confusion matrix
        self.test_predictions = []
        self.test_labels = []

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)

    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(F.relu(self.conv3(x, edge_index)))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification head
        return torch.sigmoid(self.classifier(x))

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch).squeeze()
        loss = F.binary_cross_entropy(y_hat, batch.y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy(y_hat, batch.y.int()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch).squeeze()
        loss = F.binary_cross_entropy(y_hat, batch.y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(y_hat, batch.y.int()), prog_bar=True)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.batch).squeeze()
        loss = F.binary_cross_entropy(y_hat, batch.y)
        
        # Store predictions and labels for confusion matrix
        predictions = (y_hat > 0.5).cpu().numpy().astype(int)
        labels = batch.y.cpu().numpy().astype(int)
        
        self.test_predictions.extend(predictions)
        self.test_labels.extend(labels)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy(y_hat, batch.y.int()), prog_bar=True)
        self.log('test_f1', self.test_f1(y_hat, batch.y.int()), prog_bar=True)
        self.log('test_recall', self.test_recall(y_hat, batch.y.int()), prog_bar=True)

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
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    # Preprocessing (run once before training)
    predict_markers_snapshots, virtual_markers_snapshots, displacements, ground_truth_labels = load_sim_data()
    
    # Build graph dataset:
    graphs = []
    for i in range(len(predict_markers_snapshots)):
        # Node features: [displacement_x, displacement_y] + [default_x, default_y]
        node_features = np.hstack([
            displacements[i], 
            virtual_markers_snapshots[0],
        ])
        
        # Create kNN graph (k=8)
        pos_tensor = torch.tensor(virtual_markers_snapshots[0], dtype=torch.float)
        dist_matrix = torch.cdist(pos_tensor, pos_tensor)
        k = 8
        _, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
        
        # Create edge_index
        src = torch.repeat_interleave(torch.arange(len(pos_tensor)), k)
        dst = indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)
        
        # Create PyG Data object
        graphs.append(Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor([ground_truth_labels[i]], dtype=torch.float)
        ))

    # First split: training vs. (validation + test)
    train_graphs, temp_graphs = train_test_split(graphs, test_size=0.4, random_state=42)
    
    # Second split: validation vs. test
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # Initialize and train model
    model = GNNClassifier()
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Train and validate
    trainer.fit(model, train_loader, val_loader)
    
    # Save the model and preprocessing parameters
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), 'saved_models/gnn_classifier_weights.pt')
    
    # Save preprocessing parameters
    with open('saved_models/gnn_preprocessing_params.pkl', 'wb') as f:
        pickle.dump({
            'node_dim': 4,
            'k_neighbors': 8,
            'default_positions': virtual_markers_snapshots[0]
        }, f)
    
    print("Model weights saved to saved_models/gnn_classifier_weights.pt")
    print("Preprocessing parameters saved to saved_models/gnn_preprocessing_params.pkl")
    
    # Test the model
    print(f"\nTest dataset size: {len(test_graphs)}")
    model.test_predictions = []  # Clear any previous predictions
    model.test_labels = []  # Clear any previous labels
    trainer.test(model, test_loader)

if __name__ == '__main__':
    main()