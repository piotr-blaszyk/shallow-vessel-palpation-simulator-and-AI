import gmsh
import numpy as np
import math
import sys
import pickle
import os
import itertools

def dome_radius_of_curvature(radius_of_projection, height):
    r = radius_of_projection
    h = height
    return (r ** 2 + h ** 2) / (2 * h)

def generate_vitactip_mesh():
    with open('biomimetic-tip-points.pkl', 'rb') as f:
        biomimetic_tip_points = pickle.load(f)
    A_points = biomimetic_tip_points['A_points']
    B_points = biomimetic_tip_points['B_points']
    A_points = A_points[:, [0, 2, 1]]
    B_points = B_points[:, [0, 2, 1]]
    
    # Find the point closest to (0,0) in the xz plane for A_points
    distances_A = np.sqrt(A_points[:, 0]**2 + A_points[:, 2]**2)  # xz plane distance
    closest_A_idx = np.argmin(distances_A)
    A_points = A_points[closest_A_idx:closest_A_idx+1]  # Keep only the closest point
    B_points = B_points[closest_A_idx:closest_A_idx+1]  # Keep only the closest point
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)
    gmsh.model.add("ViTacTip")

    stem_wall_radius_outer = 20
    stem_wall_radius_inner = 19
    cap_height = 6.0
    stem_height = 11.0
    R_outer = dome_radius_of_curvature(stem_wall_radius_outer, cap_height)
    R_inner = R_outer - 1
    y_cap_base = R_outer - cap_height
    y_bottom = y_cap_base - stem_height
    geometry_data = {
        'stem_wall_radius_outer': stem_wall_radius_outer,
        'stem_wall_radius_inner': stem_wall_radius_inner,
        'cap_height': cap_height,
        'stem_height': stem_height,
        'R_outer': R_outer,
        'R_inner': R_inner,
        'y_cap_base': y_cap_base,
        'y_bottom': y_bottom
    }

    outer_ball = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    inner_ball = gmsh.model.occ.addSphere(0, 0, 0, R_inner)
    cap = gmsh.model.occ.cut([(3, outer_ball)], [(3, inner_ball)])[0]
    cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
    cap = gmsh.model.occ.intersect(cap, [(3, cyl_helper)])[0]

    A_point_geometric_tags = [gmsh.model.occ.addPoint(x, y, z, meshSize=0.25) for x, y, z in A_points]
    # B_point_tags = [gmsh.model.occ.addPoint(x, y, z, meshSize=0.25) for x, y, z in B_points]

    A = A_points[0]
    B = B_points[0]
    AB = B-A
    tip_cylinder = gmsh.model.occ.addCylinder(0, A[1], 0, AB[0], AB[1], AB[2], 0.25)
    # gmsh.model.occ.synchronize()
    # gmsh.model.mesh.embed(dim=0, tags=A_point_geometric_tags, inDim=3, inTag=tip_cylinder)
    gmsh.model.occ.synchronize()
    fragments = gmsh.model.occ.fragment(cap, [(3, tip_cylinder)])[0]

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    
    new_fragments = []
    for dim, tag in fragments:
        node_tags, node_coordinates, parametric_coord = gmsh.model.mesh.getNodes(dim=dim, tag=tag, includeBoundary=True)
        node_coordinates = node_coordinates.reshape(-1, 3)
        assert node_coordinates.shape[0] > 0
        magnitudes = np.linalg.norm(node_coordinates, axis=1)
        max_magnitude_index = np.argmax(magnitudes)
        point_with_largest_magnitude = node_coordinates[max_magnitude_index]
        largest_magnitude = magnitudes[max_magnitude_index]
        magnitudes.sort()
        if largest_magnitude>R_outer+1:
            gmsh.model.occ.remove(dimTags=[(dim, tag)], recursive=True)
            continue
        new_fragments.append((dim, tag))
    
    stem_wall_outer = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, stem_wall_radius_outer)
    stem_wall_inner = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height, 0, stem_wall_radius_inner)
    stem_wall = gmsh.model.occ.cut([(3, stem_wall_outer)], [(3, stem_wall_inner)])[0]
    shell = gmsh.model.occ.fuse(new_fragments, stem_wall)[0]

    outer_ball = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    cyl_helper = gmsh.model.occ.addCylinder(0, y_bottom, 0, 0, stem_height * 2, 0, stem_wall_radius_outer)
    all_volume = gmsh.model.occ.intersect([(3, outer_ball)], [(3, cyl_helper)])[0]
    gel = gmsh.model.occ.cut(all_volume, shell, removeTool=False)[0]

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write('vitactip.msh')
    gmsh.model.addPhysicalGroup(3, [x[1] for x in shell], name="shell")
    gmsh.model.addPhysicalGroup(3, [x[1] for x in gel], name="gel")
    get_difftactile_variables(geometry_data, A_points)
    gmsh.fltk.run()
    gmsh.finalize()

def get_difftactile_variables(geometry_data, A_points):
    # Unpack all geometry variables
    stem_wall_radius_outer = geometry_data['stem_wall_radius_outer']
    stem_wall_radius_inner = geometry_data['stem_wall_radius_inner']
    cap_height = geometry_data['cap_height']
    stem_height = geometry_data['stem_height']
    R_outer = geometry_data['R_outer']
    R_inner = geometry_data['R_inner']
    y_cap_base = geometry_data['y_cap_base']
    y_bottom = geometry_data['y_bottom']

    element_types, tetrahedra_tags, tetrahedra_vertex_tags = gmsh.model.mesh.getElements(dim=3)
    all_tetrahedra = tetrahedra_vertex_tags[0].reshape(-1, 4).astype(int)
    tetrahedra_tags = tetrahedra_tags[0]
    element_types_2d, triangle_tags, triangle_nodes = gmsh.model.mesh.getElements(dim=2)
    all_surface_triangles = triangle_nodes[0].reshape(-1, 3).astype(int)
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_coordinates = node_coordinates.reshape(-1, 3).astype(float)
    node_tags = np.array(node_tags)
    node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
    node_labels = np.zeros((len(node_tags), 3), dtype=bool)
    physical_groups = gmsh.model.getPhysicalGroups()
    group_to_idx = {"shell": 0, "gel": 1}

    surface_node_tags = []
    surface_coords = []
    for i in range(len(node_tags)):
        tag = node_tags[i]
        node = node_coordinates[i]
        x,y,z = node
        if y > y_cap_base:
            if np.linalg.norm(node) > R_outer - 0.1:
                surface_node_tags.append(tag)
                surface_coords.append(node)
        else:
            xz = np.array([x, z])
            if np.linalg.norm(xz) > stem_wall_radius_outer - 0.1:
                surface_node_tags.append(tag)
                surface_coords.append(node)
    surface_node_tags = np.array(surface_node_tags)
    surface_coords = np.array(surface_coords).reshape(-1, 3).astype(float)
    
    # Compute surface triangles - triangles where all nodes belong to surface_node_tags
    surface_triangles_mask = []
    for triangle in all_surface_triangles:
        all_nodes_surface = all(tag in surface_node_tags for tag in triangle)
        surface_triangles_mask.append(all_nodes_surface)
    surface_triangles = all_surface_triangles[surface_triangles_mask]

    for dim, tag in physical_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        if name in group_to_idx:
            group_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
            node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
            for node_tag in group_node_tags:
                if node_tag in node_tag_to_idx:
                    node_labels[node_tag_to_idx[node_tag], group_to_idx[name]] = True

    A = A_points[0]
    min_dist = float('inf')
    min_node_tag = None
    for x in np.unique(all_tetrahedra.flatten()):
        dist = np.linalg.norm(node_coordinates[x-1]-A)
        if dist < min_dist:
            min_dist = dist
            min_node_tag = x
    
    assert min_node_tag is not None, "no marker node found"

    marker_node_tags = np.array([min_node_tag])-1

    node_coordinates = node_coordinates[1:]
    node_labels = node_labels[1:]
    all_tetrahedra -= 1
    surface_triangles -= 1
    marker_node_tags -= 1
    all_tetrahedra_temp = []
    for tetra in all_tetrahedra:
        if -1 not in tetra:
            all_tetrahedra_temp.append(tetra)
    all_tetrahedra = np.array(all_tetrahedra_temp)
    surface_triangles_temp = []
    for tetra in surface_triangles:
        if -1 not in tetra:
            surface_triangles_temp.append(tetra)
    surface_triangles = np.array(surface_triangles_temp)

    mesh_data = {
        'all_tetrahedra': all_tetrahedra-1,
        'node_coordinates': node_coordinates,
        'node_labels': node_labels,
        'surface_node_tags': surface_node_tags-1,
        'surface_triangles': surface_triangles-1,
        'node_tags': node_tags-1,
        'group_to_idx': group_to_idx,
        'marker_node_tags': marker_node_tags,

        'y_bottom': y_bottom,
        'R_inner': R_inner,
        'R_outer': R_outer,
    }
    os.makedirs('output', exist_ok=True)
    with open('output/gmsh-mesh.pkl', 'wb') as f:
        pickle.dump(mesh_data, f)

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
    outer_ball = gmsh.model.occ.addSphere(0, 0, 0, 10)
    inner_ball = gmsh.model.occ.addSphere(0, 0, 0, 5)
    fragments = gmsh.model.occ.fragment([(3, outer_ball)], [(3, inner_ball)])[0]

    for dim, tag in fragments:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        foo = 7
    
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

def get_fragment_node_tags(fragment_entities):
    all_node_tags = []
    all_node_coords = []
    for dim, tag in fragment_entities:
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim, tag)
        all_node_tags.extend(node_tags)
        all_node_coords.extend(node_coords)
    all_node_tags = np.array(all_node_tags)
    all_node_coords = np.array(all_node_coords).reshape(-1, 3)
    unique_node_tags, unique_indices = np.unique(all_node_tags, return_index=True)
    unique_node_coords = all_node_coords[unique_indices]
    return unique_node_tags, unique_node_coords

def example_fragment_usage():
    outer_ball = gmsh.model.occ.addSphere(0, 0, 0, 10)
    inner_ball = gmsh.model.occ.addSphere(0, 0, 0, 5)
    fragments = gmsh.model.occ.fragment([(3, outer_ball)], [(3, inner_ball)])
    if fragments[0]:
        node_tags, node_coords = get_fragment_node_tags(fragments[0])
        print(f"Fragment has {len(node_tags)} unique vertices")
        print(f"First few vertices: {node_tags[:5]}")
        print(f"Vertex coordinates shape: {node_coords.shape}")

def fragments_dfs_get_tetrahedra_tags(x):
    if isinstance(x, int):
        return []
    if not (isinstance(x, tuple) or isinstance(x, list)):
        raise Exception("unexpected structure of fragments data")
    if len(x) == 2 and isinstance(x[0], int) and isinstance(x[1], int):
        return [x[1]]
    res_nested = [fragments_dfs_get_tetrahedra_tags(y) for y in x]
    return np.unique(np.array(list(itertools.chain.from_iterable(res_nested)))).tolist()

if __name__ == "__main__":
    np.set_printoptions(formatter={'float_kind': '{:.2f}'.format})
    generate_vitactip_mesh()
