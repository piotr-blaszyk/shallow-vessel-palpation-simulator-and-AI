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

echo "All dependencies installed successfully!" 