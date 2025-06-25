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

    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(3)
    gmsh.fltk.run()
    gmsh.finalize()

def point_to_triangle_distance(point, triangle_vertices):
    # Convert inputs to numpy arrays
    point = np.array(point)
    v0, v1, v2 = np.array(triangle_vertices)
    
    # Calculate triangle centroid (arithmetic mean of vertices)
    centroid = (v0 + v1 + v2) / 3.0
    
    # Calculate Euclidean distance between point and centroid
    return np.linalg.norm(point - centroid)

def sandbox():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("sandbox")
    x, y, z = (0.5, 1.5, 0.5)
    point = np.array([x, y, z])
    gmsh.model.occ.addBox(x=0, y=0, z=0, dx=1, dy=1, dz=1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    # element_types is irrelevant
    # triangle_tags[0] is a 1d int numpy array of shape (num_triangles,)
    # triangle_vertex_tags[0] is a 1d int numpy array of shape (num_traingles * 3,)
    element_types, triangle_tags, triangle_vertex_tags = gmsh.model.mesh.getElements(dim=2)
    # node_tags is a 1d int python array of length num_particles
    # node_coordinates is a 1d float numpy array of shape (num_particles*3,)
    # parametric_coord is irrelevant
    node_tags, node_coordinates, parametric_coord = gmsh.model.mesh.getNodes()
    
    # Reshape coordinates array into (n_nodes, 3) array
    node_coordinates = node_coordinates.reshape(-1, 3)
    # Create a mapping from node tags to their indices
    node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
    
    # Process triangles
    triangles = []
    node_tags_array = triangle_vertex_tags[0].reshape(-1, 3)  # Reshape into (n_triangles, 3)
    
    min_distance = float('inf')
    closest_triangle_idx = None
    closest_triangle_nodes = None
    closest_triangle_coords = None
    
    # Iterate through all triangles
    for tri_idx, triangle_node_tags in enumerate(node_tags_array):
        # Get the coordinates of triangle vertices
        triangle_vertices = [node_coordinates[node_tag_to_idx[tag]] for tag in triangle_node_tags]
        
        # Calculate distance from point to triangle
        distance = point_to_triangle_distance(point, triangle_vertices)
        
        # Update if this is the closest triangle so far
        if distance < min_distance:
            min_distance = distance
            closest_triangle_idx = tri_idx
            closest_triangle_nodes = triangle_node_tags
            closest_triangle_coords = triangle_vertices
    
    print(f"\nClosest triangle:")
    print(f"Triangle index: {closest_triangle_idx}")
    print(f"Node indices: {closest_triangle_nodes}")
    print(f"Node coordinates:")
    for i, coords in enumerate(closest_triangle_coords):
        print(f"Node {i}: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")
    print(f"Distance to point: {min_distance:.3f}")

    new_point_tag = gmsh.model.addDiscreteEntity(0)
    gmsh.model.mesh.addNodes(0, new_point_tag, [], [x, y, z])
    a, b, c = closest_triangle_nodes
    tetra_node_tags = [a, b, c, new_point_tag]
    new_volume_tag = gmsh.model.addDiscreteEntity(3)
    gmsh.model.mesh.addElementsByType(new_volume_tag, 3, [], tetra_node_tags)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    sandbox()
