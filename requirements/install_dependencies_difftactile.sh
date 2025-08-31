#!/bin/zsh

# Exit on error
set -e

echo "Installing dependencies..."

# Install PyTorch and related packages

uv pip install pyvista
uv pip install meshio
uv pip install snakeviz
uv pip install vedo
uv pip install bayesian-optimization
uv pip install ruff

echo "All dependencies installed successfully!" 