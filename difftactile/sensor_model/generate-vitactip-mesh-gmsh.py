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
    with open('biomimetic-tip-cones.pkl', 'rb') as f:
        marker_positions = pickle.load(f)
    cone_tips = marker_positions['cone_tips']
    cone_base_centers = marker_positions['cone_base_centres']
    cone_tips = cone_tips[:, [0, 2, 1]]
    cone_base_centers = cone_base_centers[:, [0, 2, 1]]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ViTacTip")

    stem_wall_radius_outer = 20
    stem_wall_radius_inner = 19
    cap_height = 6.0
    stem_height = 11.0
    R_outer = dome_radius_of_curvature(stem_wall_radius_outer, cap_height)
    R_inner = R_outer - 1
    y_cap_base = R_outer - cap_height
    y_bottom = y_cap_base - stem_height

    outer_ball = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    inner_ball = gmsh.model.occ.addSphere(0, 0, 0, R_inner)
    cap = gmsh.model.occ.cut([(3, outer_ball)], [(3, inner_ball)])[0]
    cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
    cap = gmsh.model.occ.intersect(cap, [(3, cyl_helper)])[0]
    stem_wall_outer = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, stem_wall_radius_outer)
    stem_wall_inner = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, stem_wall_radius_inner)
    stem_wall = gmsh.model.occ.cut([(3, stem_wall_outer)], [(3, stem_wall_inner)])[0]
    shell = gmsh.model.occ.fuse(cap, stem_wall)[0]

    outer_ball = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
    all_volume = gmsh.model.occ.intersect([(3, outer_ball)], [(3, cyl_helper)])[0]
    gel = gmsh.model.occ.cut(all_volume, shell, removeTool=False)[0]

    tips = [gmsh.model.occ.addPoint(x, y, z, meshSize=1.0) for x, y, z in cone_tips]
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.embed(0, tips, 3, gel[0][1])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(0, tips, name="tips")
    gmsh.model.addPhysicalGroup(3, [shell[0][1]], name="shell")
    gmsh.model.addPhysicalGroup(3, [gel[0][1]], name="gel")
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(3)
    gmsh.write('vitactip.msh')
    all_tetrahedra, all_nodes, node_labels, all_surface_triangles, shell_surface_triangles = get_difftactile_variables()
    gmsh.model.addPhysicalGroup(2, np.unique(shell_surface_triangles.flatten()).tolist(), name="shell_surface_triangles")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.fltk.run()
    gmsh.finalize()

def get_difftactile_variables():
    # Get all tetrahedra elements
    element_types, tetrahedra_tags, tetrahedra_vertex_tags = gmsh.model.mesh.getElements(dim=3)
    
    # Reshape the vertex tags array to get a (num_tetrahedra, 4) array
    # where each row contains the 4 node tags that make up a tetrahedron
    all_tetrahedra = tetrahedra_vertex_tags[0].reshape(-1, 4).astype(int)
    
    # Get all nodes and their coordinates
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    
    # Reshape node coordinates into (num_nodes, 3) array
    all_nodes = node_coordinates.reshape(-1, 3).astype(float)
    
    # Initialize node labels array with False
    # Each row corresponds to a node, columns are [tips, shell, gel]
    node_labels = np.zeros((len(node_tags), 3), dtype=bool)
    
    # Get all physical groups
    physical_groups = gmsh.model.getPhysicalGroups()
    
    # Create a mapping from physical group name to column index
    group_to_idx = {"tips": 0, "shell": 1, "gel": 2}
    
    # For each physical group
    for dim, tag in physical_groups:
        # Get the name of this physical group
        name = gmsh.model.getPhysicalName(dim, tag)
        if name in group_to_idx:
            # Get nodes in this physical group
            group_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
            # Create mapping from node tag to index
            node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
            # Set True for these nodes in the appropriate column
            for node_tag in group_node_tags:
                if node_tag in node_tag_to_idx:
                    node_labels[node_tag_to_idx[node_tag], group_to_idx[name]] = True
    
    # Get all surface triangles (2D elements)
    element_types_2d, triangle_tags, triangle_nodes = gmsh.model.mesh.getElements(dim=2)
    all_surface_triangles = triangle_nodes[0].reshape(-1, 3).astype(int)
    
    # Create mapping from node tag to index for all nodes
    node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
    
    # Filter triangles based on node labels
    shell_triangles_mask = []
    for triangle in all_surface_triangles:
        # Get node indices for this triangle
        node_indices = [node_tag_to_idx[tag] for tag in triangle]
        # Check if all nodes belong to shell
        all_in_shell = all(node_labels[idx, group_to_idx["shell"]] for idx in node_indices)
        # Check if no nodes belong to gel
        none_in_gel = not any(node_labels[idx, group_to_idx["gel"]] for idx in node_indices)
        # Keep triangle if both conditions are met
        shell_triangles_mask.append(all_in_shell and none_in_gel)
    
    # Apply mask to get shell surface triangles
    shell_surface_triangles = all_surface_triangles[shell_triangles_mask]
    
    return all_tetrahedra, all_nodes, node_labels, all_surface_triangles, shell_surface_triangles

def point_to_triangle_distance(point, triangle_vertices):
    # Convert inputs to numpy arrays
    point = np.array(point)
    v0, v1, v2 = np.array(triangle_vertices)
    
    # Calculate triangle centroid (arithmetic mean of vertices)
    centroid = (v0 + v1 + v2) / 3.0
    
    # Calculate Euclidean distance between point and centroid
    return np.linalg.norm(point - centroid)

def find_closest_triangle(point):
    # element_types is irrelevant
    # triangle_tags[0] is a 1d int numpy array of shape (num_triangles,)
    # triangle_vertex_tags[0] is a 1d int numpy array of shape (num_traingles * 3,)
    element_types, triangle_tags, triangle_vertex_tags = gmsh.model.mesh.getElements(dim=2)
    # node_tags is a 1d int python array of length num_particles
    # node_coordinates is a 1d float numpy array of shape (num_particles*3,)
    # parametric_coord is irrelevant
    node_tags, node_coordinates, parametric_coord = gmsh.model.mesh.getNodes()

    node_coordinates = node_coordinates.reshape(-1, 3)
    node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
    triangles = []
    node_tags_array = triangle_vertex_tags[0].reshape(-1, 3)
    min_distance = float('inf')
    closest_triangle_idx = None
    closest_triangle_nodes = None
    closest_triangle_coords = None
    for tri_idx, triangle_node_tags in enumerate(node_tags_array):
        triangle_vertices = [node_coordinates[node_tag_to_idx[tag]] for tag in triangle_node_tags]
        distance = point_to_triangle_distance(point, triangle_vertices)
        if distance < min_distance:
            min_distance = distance
            closest_triangle_idx = tri_idx
            closest_triangle_nodes = triangle_node_tags
            closest_triangle_coords = triangle_vertices
    
    if False:
        print(f"\nClosest triangle:")
        print(f"Triangle index: {closest_triangle_idx}")
        print(f"Node indices: {closest_triangle_nodes}")
        print(f"Node coordinates:")
        for i, coords in enumerate(closest_triangle_coords):
            print(f"Node {i}: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")
        print(f"Distance to point: {min_distance:.3f}")
    
    return closest_triangle_idx, closest_triangle_nodes, closest_triangle_coords, min_distance

def sandbox():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("sandbox")
    
    # Set mesh size parameters for coarse mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)  # Minimum element size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)  # Maximum element size
    
    x, y, z = (0.5, 1.5, 0.5)
    point = np.array([x, y, z])
    gmsh.model.occ.addBox(x=0, y=0, z=0, dx=1, dy=1, dz=1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    
    closest_triangle_idx, closest_triangle_nodes, closest_triangle_coords, min_distance = find_closest_triangle(point)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write('sandbox.msh')
    gmsh.fltk.run()
    gmsh.finalize()

def load_mesh():
    gmsh.initialize()
    gmsh.open("sandbox.msh")
    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    generate_vitactip_mesh()
