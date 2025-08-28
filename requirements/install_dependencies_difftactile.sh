#!/bin/zsh

# Exit on error
set -e

echo "Installing dependencies..."

# Install PyTorch and related packages

uv pip install pyvista
uv pip install meshio

echo "All dependencies installed successfully!" 