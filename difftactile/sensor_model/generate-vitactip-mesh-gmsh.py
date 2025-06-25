import gmsh
import numpy as np
import math
import sys
import pickle

def generate_vitactip_mesh(tips, visualize_layers=None):
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
    shell_volume = gmsh.model.occ.cut(shell_outer, gel_volume, removeTool=False)[0]
    
    # Add biomimetic tips (points on inner surface)
    tip_tags = [gmsh.model.occ.addPoint(x, y, z, meshSize=0.01) for x, y, z in tips]

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
    
    # Get all nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    all_nodes = np.array(node_coords).reshape(-1, 3)
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    
    # Get tetrahedrons (3D mesh)
    tetra_tags, tetra_nodes = gmsh.model.mesh.getElementsByType(4)  # 4-node tetra
    all_f2v = tetra_nodes.reshape(-1, 4) if tetra_nodes.size > 0 else np.empty((0, 4), dtype=int)
    
    # Get outer surface triangles
    outer_surf_nodes = set()
    surface_f2v = []
    for entity in gmsh.model.getEntities(2):
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, entity[1])
        if elem_types:
            for elem_type, node_tags in zip(elem_types, elem_node_tags):
                if elem_type == 2:  # Triangle
                    tri_nodes = node_tags.reshape(-1, 3)
                    surface_f2v.append(tri_nodes)
                    outer_surf_nodes.update(node_tags)
    surface_f2v = np.vstack(surface_f2v) if surface_f2v else np.empty((0, 3), dtype=int)
    
    # Assign layer indices
    min_y = np.min(all_nodes[:, 1])
    base_threshold = min_y + 1.0
    layer_idxs = np.full(len(node_tags), 4, dtype=int)  # Default: non-base gel
    
    # Layer 0: Biomimetic tips
    tip_node_tags = gmsh.model.mesh.getNodesForPhysicalGroup(0, tip_group)[0]
    for tag in tip_node_tags:
        idx = tag_to_idx[tag]
        layer_idxs[idx] = 0
    
    # Layer 1: Base region (y <= min_y + 1mm)
    base_mask = all_nodes[:, 1] <= base_threshold
    base_idxs = np.where(base_mask)[0]
    for idx in base_idxs:
        if layer_idxs[idx] != 0:  # Don't override tips
            layer_idxs[idx] = 1
    
    # Layer 2: Outer surface
    for tag in outer_surf_nodes:
        idx = tag_to_idx[tag]
        if layer_idxs[idx] not in {0, 1}:  # Higher priority layers
            layer_idxs[idx] = 2
    
    # Layer 3: Remaining shell nodes
    shell_tetra_nodes = set()
    _, shell_tetra_node_tags = gmsh.model.mesh.getElementsByType(4, 3)  # Shell volume
    for tag in shell_tetra_node_tags:
        shell_tetra_nodes.add(tag)
    for tag in shell_tetra_nodes:
        idx = tag_to_idx[tag]
        if layer_idxs[idx] == 4:  # Only unassigned nodes
            layer_idxs[idx] = 3
    
    # Visualize specified layers
    if visualize_layers:
        gmsh.fltk.initialize()
        view_tag = gmsh.view.add("Layer View")
        gmsh.view.addModelData(view_tag, 0, "ViTacTip", "NodeData", node_tags, layer_idxs.reshape(-1, 1))
        gmsh.view.option.setNumber(view_tag, "PointSize", 5)
        gmsh.view.option.setNumber(view_tag, "ColormapNumber", 14)  # Rainbow colormap
        gmsh.fltk.run()
    
    # Finalize Gmsh
    gmsh.finalize()
    
    return all_nodes, all_f2v, surface_f2v, layer_idxs

# Example usage
if __name__ == "__main__":
    # Load marker positions from pickle file
    with open('init-marker-positions-3d.pkl', 'rb') as f:
        marker_positions = pickle.load(f)

    # Transform coordinates from (x, y, z) to (x, z, y)
    tips = marker_positions[:, [0, 2, 1]]  # Reorder columns to swap y and z
    
    # Generate mesh and extract arrays
    all_nodes, all_f2v, surface_f2v, layer_idxs = generate_vitactip_mesh(
        tips, 
        visualize_layers=[0, 1, 2, 3, 4]  # Visualize all layers
    )
    
    # Print array shapes
    print("Nodes shape:", all_nodes.shape)
    print("Tetrahedrons shape:", all_f2v.shape)
    print("Surface triangles shape:", surface_f2v.shape)
    print("Layer indices shape:", layer_idxs.shape)