#!/bin/zsh

# Exit on error
set -e

echo "Installing dependencies..."

# Install PyTorch and related packages

uv pip install tqdm
uv pip install opencv-python
uv pip install taichi
uv pip install pyvista
uv pip install meshio
uv pip install snakeviz
uv pip install vedo
uv pip install bayesian-optimization
uv pip install ruff
uv pip install ipython
uv pip install shapely
uv pip install trimesh
uv pip install mesh-to-sdf
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
uv pip install lightning
uv pip install numpy scikit-learn
uv pip install matplotlib
uv pip install -U 'tensorboardX'
uv pip install -U 'tensorboard'
uv pip install open3d

echo "All dependencies installed successfully!" 