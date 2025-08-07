# ViTacTip CadQuery Mesh Generator

This is a CadQuery-based implementation of the ViTacTip tactile sensor mesh generator, equivalent to the original GMSH implementation.

## Overview

The `ViTacTipMeshGenerator` class replicates the core geometric functionality of the GMSH mesh generator for creating optical tactile sensor meshes. It supports:

- Single material or two-material configurations (shell + gel)
- Spherical cap (dome) with cylindrical stem geometry
- Optional biomimetic tips
- Volume calculations in SI units
- STL export functionality

## Installation

Install the required dependencies:

```bash
pip install -r requirements_cadquery.txt
```

## Usage

### Basic Usage

```python
from vitactip_cadquery import ViTacTipMeshGenerator

# Create mesh generator with default parameters
generator = ViTacTipMeshGenerator()

# Generate mesh
result = generator.generate_vitactip_mesh()

# Export to STL
generator.export_mesh(result, "vitactip_mesh.stl")
```

### Custom Parameters

```python
# Define custom system parameters
custom_params = {
    "vitactip": {
        "number_of_biomimetic_tips": 1,
        "number_of_materials": 2
    },
    "gmsh_mm": {
        "stem_wall_radius_outer": 200.0,  # mm
        "stem_wall_radius_inner": 190.0,  # mm
        "cap_height": 60.0,  # mm
        "stem_height": 300.0,  # mm
        "shell_thickness": 10.0,  # mm
        "dist_eps": 1.0  # mm
    }
}

# Create generator with custom parameters
generator = ViTacTipMeshGenerator(custom_params)
result = generator.generate_vitactip_mesh()
```

## Key Features

### Geometry Generation

The implementation creates the following geometric components:

1. **Spherical Cap (Dome)**: Creates the tactile sensing surface
2. **Cylindrical Stem**: Provides structural support
3. **Shell and Gel**: For two-material configurations
4. **Biomimetic Tips**: Optional protrusions for enhanced sensing

### Volume Calculations

The generator calculates volumes in SI units (m³) for:
- Shell volume (outer volume - inner volume)
- Gel volume (inner volume)
- Total volume

### Export Options

- STL file export for 3D printing or further processing
- Support for both single and multi-material configurations

## Differences from GMSH Implementation

1. **No Mesh Refinement**: CadQuery doesn't include the mesh refinement options available in GMSH
2. **Simplified Biomimetic Tips**: The tip creation is simplified compared to the GMSH fragment operation
3. **No Physical Groups**: CadQuery doesn't create physical groups for different materials like GMSH
4. **Direct STL Export**: Exports directly to STL without intermediate mesh format

## Parameters

### System Parameters Structure

```python
{
    "vitactip": {
        "number_of_biomimetic_tips": 0,  # Number of biomimetic tips
        "number_of_materials": 1          # 1 for single material, 2 for shell+gel
    },
    "gmsh_mm": {
        "stem_wall_radius_outer": 200.0,  # Outer radius of stem (mm)
        "stem_wall_radius_inner": 190.0,  # Inner radius of stem (mm)
        "cap_height": 60.0,               # Height of spherical cap (mm)
        "stem_height": 300.0,             # Height of cylindrical stem (mm)
        "shell_thickness": 10.0,          # Thickness of shell (mm)
        "dist_eps": 1.0                   # Distance epsilon (mm)
    }
}
```

## Example Output

When running the generator, you'll see output like:

```
Volume: 1.234e-06 m³
Mesh exported to vitactip_cadquery.stl
ViTacTip mesh generation completed using CadQuery!
```

## File Structure

- `vitactip_cadquery.py`: Main implementation file
- `requirements_cadquery.txt`: Python dependencies
- `README_cadquery.md`: This documentation file

## Limitations

1. **No Mesh Refinement**: Unlike GMSH, CadQuery doesn't provide mesh refinement capabilities
2. **Simplified Tip Geometry**: Biomimetic tip creation is simplified
3. **No Physical Groups**: Material assignment is handled differently than in GMSH
4. **Limited Boolean Operations**: Some complex boolean operations available in GMSH are not directly available in CadQuery

## Dependencies

- `cadquery>=2.3.0`: 3D CAD modeling library
- `numpy>=1.21.0`: Numerical computing library 