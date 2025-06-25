import gmsh
import numpy as np
import math
import sys
import pickle

def generate_vitactip_mesh(tips):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ViTacTip")
    
    # Parameters
    R_outer = 33.0    # Outer radius (mm)
    R_inner = 30.0    # Inner radius (mm)
    cap_height = 6.0   # Cap height (mm)
    stem_height = 11.0 # Stem height (mm)
    y_cap_base = R_outer - cap_height  # Y-coordinate of cap base
    y_bottom = y_cap_base - stem_height  # Bottom Y-coordinate
    
    # Compute base radii
    base_radius_outer = math.sqrt(R_outer**2 - y_cap_base**2)
    base_radius_inner = math.sqrt(R_inner**2 - y_cap_base**2)
    
    # Create spherical cap volumes
    outer_cap = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    inner_cap = gmsh.model.occ.addSphere(0, 0, 0, R_inner)
    tool = gmsh.model.occ.addBox(-100, y_cap_base, -100, 200, 200, 200)
    outer_cap = gmsh.model.occ.intersect([(3, outer_cap)], [(3, tool)], removeTool=False)[0]
    inner_cap = gmsh.model.occ.intersect([(3, inner_cap)], [(3, tool)])[0]
    
    # Create stem volumes
    outer_cyl = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, base_radius_outer)
    inner_cyl = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, base_radius_inner)
    stem_wall = gmsh.model.occ.cut([(3, outer_cyl)], [(3, inner_cyl)], removeObject=False, removeTool=False)[0]
    
    shell_outer = gmsh.model.occ.fuse(outer_cap, [(3, outer_cyl)])[0]
    
    # Create gel volume
    gel_volume = gmsh.model.occ.fuse(inner_cap, [(3, inner_cyl)])[0]
    
    # Subtract gel from shell to get final shell
    shell_volume = gmsh.model.occ.cut(shell_outer, gel_volume, removeObject=False)[0]
    
    # Add biomimetic tips (points on inner surface)
    tip_tags = [gmsh.model.occ.addPoint(x, y, z, meshSize=1.0) for x, y, z in tips]

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.embed(0, tip_tags, 3, shell_volume[0][1])

    gel_volume = gmsh.model.occ.cut(shell_outer, shell_volume, removeTool=False)[0]

    gmsh.model.occ.synchronize()
    tip_group = gmsh.model.addPhysicalGroup(0, tip_tags, name="Tips")
    shell_group = gmsh.model.addPhysicalGroup(3, [x[1] for x in shell_volume], name="Shell")
    gel_group = gmsh.model.addPhysicalGroup(3, [x[1] for x in gel_volume], name="Gel")
    
    # Generate mesh
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT algorithm
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay
    gmsh.model.mesh.generate(3)
    gmsh.fltk.run()
    gmsh.finalize()

# Example usage
if __name__ == "__main__":
    # Load marker positions from pickle file
    with open('init-marker-positions-3d.pkl', 'rb') as f:
        marker_positions = pickle.load(f)

    # Transform coordinates from (x, y, z) to (x, z, y)
    tips = marker_positions[:, [0, 2, 1]]  # Reorder columns to swap y and z
    
    # Generate mesh and extract arrays
    generate_vitactip_mesh(
        tips
    )