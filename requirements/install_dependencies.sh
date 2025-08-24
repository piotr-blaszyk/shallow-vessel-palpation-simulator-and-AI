#!/bin/zsh

# Exit on error
set -e

echo "Installing PyTorch dependencies..."

# Install PyTorch and related packages
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
uv pip install lightning
uv pip install numpy scikit-learn
uv pip install shapely
uv pip install opencv-python
uv pip install matplotlib
uv pip install -U 'tensorboardX'
uv pip install -U 'tensorboard'
uv pip install seaborn
uv pip install open3d
uv pip install torch_geometric_temporal
uv pip install ruff

echo "All dependencies installed successfully!" 