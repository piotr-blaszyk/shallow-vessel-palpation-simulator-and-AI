import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from graph_neural_network import GNNClassifier
from torch_geometric.data import Data, DataLoader

from difftactile.main.constants import *


def load_and_preprocess_data():
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
    
    # Load GNN preprocessing parameters
    with open(SYSTEM_PARAMS.files.gnn_preprocessing_params, 'rb') as f:
        params = pickle.load(f)
    
    # Create graph dataset
    graphs = []
    for i in range(len(displacements)):
        # Node features: [displacement_x, displacement_y] + [default_x, default_y]
        node_features = np.hstack([
            displacements[i],
            init_marker_positions
        ])
        
        # Create kNN graph
        pos_tensor = torch.tensor(init_marker_positions, dtype=torch.float)
        dist_matrix = torch.cdist(pos_tensor, pos_tensor)
        k = params['k_neighbors']
        _, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
        
        # Create edge_index
        src = torch.repeat_interleave(torch.arange(len(pos_tensor)), k)
        dst = indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)
        
        # Create PyG Data object
        graphs.append(Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor([new_labels[i]], dtype=torch.float)
        ))
    
    print(f"Processed data shape: {len(graphs)} graphs")
    print(f"Number of samples per class: {np.bincount(new_labels)}")
    print(f"Marker positions scaled from 1920x1080 to 640x480")
    print(f"Created k-NN graphs with k={params['k_neighbors']}")
    
    return graphs

def main():
    # Load preprocessing parameters
    with open(SYSTEM_PARAMS.files.gnn_preprocessing_params, 'rb') as f:
        params = pickle.load(f)
    
    # Initialize and load model
    model = GNNClassifier(node_dim=params['node_dim'])
    model.load_state_dict(torch.load('saved_models/gnn_classifier_weights.pt'))
    model.eval()
    
    # Load and preprocess experimental data
    graphs = load_and_preprocess_data()
    
    # Create dataloader
    test_loader = DataLoader(graphs, batch_size=32)
    
    # Initialize trainer and test
    trainer = pl.Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cuda:0',
        devices=1,
        enable_progress_bar=True,
        deterministic=True
    )
    
    # Clear any previous predictions and labels
    model.test_predictions = []
    model.test_labels = []
    
    # Run test
    print("\nTesting GNN model on experimental data:")
    trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
