import gmsh
import numpy as np
import math
import sys
import pickle

def dome_radius_of_curvature(radius_of_projection, height):
    r = radius_of_projection
    h = height
    return (r ** 2 + h ** 2) / (2 * h)

def generate_vitactip_mesh():
    # Load marker positions from pickle file
    with open('biomimetic-tip-cones.pkl', 'rb') as f:
        marker_positions = pickle.load(f)

    cone_tips = marker_positions['cone_tips']
    cone_base_centers = marker_positions['cone_base_centres']
    # Transform coordinates from (x, y, z) to (x, z, y)
    cone_tips = cone_tips[:, [0, 2, 1]]
    cone_base_centers = cone_base_centers[:, [0, 2, 1]]

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ViTacTip")

    # Parameters
    base_radius_outer = 20
    base_radius_inner = base_radius_outer - 2
    cap_height = 6.0   # Cap height (mm)
    stem_height = 11.0 # Stem height (mm)
    R_outer = dome_radius_of_curvature(base_radius_outer, cap_height)    # Outer radius (mm)
    R_inner = dome_radius_of_curvature(base_radius_inner, cap_height)    # Inner radius (mm)
    y_cap_base = R_outer - cap_height  # Y-coordinate of cap base
    y_bottom = y_cap_base - stem_height  # Bottom Y-coordinate
    
    # Create spherical cap volumes
    outer_cap = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    inner_cap = gmsh.model.occ.addSphere(0, 0, 0, R_inner)
    outer_cyl_tool = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, base_radius_outer)
    outer_cap = gmsh.model.occ.intersect([(3, outer_cap)], [(3, outer_cyl_tool)], removeTool=False)[0]
    inner_cap = gmsh.model.occ.intersect([(3, inner_cap)], [(3, outer_cyl_tool)])[0]
    
    # Create stem volumes
    outer_cyl = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, base_radius_outer)
    inner_cyl = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, base_radius_inner)
    stem_wall = gmsh.model.occ.cut([(3, outer_cyl)], [(3, inner_cyl)], removeObject=False, removeTool=False)[0]
    
    shell_outer = gmsh.model.occ.fuse(outer_cap, [(3, outer_cyl)])[0]
    
    # Create gel volume
    gel_volume = gmsh.model.occ.fuse(inner_cap, [(3, inner_cyl)])[0]
    
    # Subtract gel from shell to get final shell
    shell_volume = gmsh.model.occ.cut(shell_outer, gel_volume, removeTool=False)[0]
    
    # Add biomimetic tips (points on inner surface)
    tip_tags = [gmsh.model.occ.addPoint(x, y, z, meshSize=0.5) for x, y, z in cone_tips]

    cones = []
    for i in range(cone_tips.shape[0]):
        x, y, z = cone_tips[i]
        dx, dy, dz = cone_base_centers[i] - cone_tips[i]
        cone = gmsh.model.occ.addCone(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, r1=0, r2=1)
        cones.append(cone)
    
    # shell_with_biomimetic_tips = gmsh.model.occ.fuse(shell_volume, [(3, x) for x in cones], removeTool=False)[0]

    # gel_volume = gmsh.model.occ.cut(gel_volume, [(3, x) for x in cones])[0]

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.embed(0, tip_tags, 3, shell_volume[0][1])

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

def box_point():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("sandbox")
    
    gmsh.model.occ.addCone(x=0, y=0, z=0, dx=0, dy=1, dz=0, r1=0, r2=1/3)
    
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT algorithm
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay
    gmsh.model.mesh.generate(3)
    gmsh.fltk.run()

    gmsh.finalize()

# Example usage
if __name__ == "__main__":
    generate_vitactip_mesh()
