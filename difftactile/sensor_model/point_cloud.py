import numpy as np
import open3d as o3d
import pickle
from collections import Counter

with open('output/gmsh-mesh.pkl', 'rb') as f:
    mesh_data = pickle.load(f)
        
# Unpack mesh data
all_tetrahedra = mesh_data['all_tetrahedra']
node_coordinates = mesh_data['node_coordinates']
node_labels = mesh_data['node_labels']
surface_node_tags = mesh_data['surface_node_tags']
surface_triangles = mesh_data['surface_triangles']
node_tags = mesh_data['node_tags']
group_to_idx = mesh_data['group_to_idx']
y_bottom = mesh_data['y_bottom']
R_inner = mesh_data['R_inner']
R_outer = mesh_data['R_outer']

print('hello')
print(node_coordinates.shape)

# with open(f'init-marker-positions-3d.pkl', 'rb') as f:
#     points = pickle.load(f)

# with open(f'../tasks/output/fem_sensor.interp_idx_flat.pkl', 'rb') as f:
#     interp_idx_flat = pickle.load(f)

# with open(f'../tasks/output/fem_sensor.cam_3d_nodes.pkl', 'rb') as f:
#     cam_3d_nodes = pickle.load(f)

# with open(f'../tasks/output/tactile_sensor.f2v.pkl', 'rb') as f:
#     tetrahedra_indices = pickle.load(f)
# tetrahedra_indices = tetrahedra_indices.astype(int)

if False:
    counter = Counter(interp_idx_flat)
    for elem, count in counter.most_common():
        print(f"{elem}: {count}")

    # Remove duplicates
    interp_idx_flat = np.unique(interp_idx_flat)

    print(f'num of 3d marker points (after de-duplication): {interp_idx_flat.shape[0]}')

    with open(f'output/fem_sensor.surface_id_np.pkl', 'rb') as f:
        surface_id_np = pickle.load(f)

    surface_nodes = points[surface_id_np]
    marker_nodes = surface_nodes[interp_idx_flat]
    marker_nodes[:, 1] = 1.0
    surface_nodes[:, 1] = 2.0
    # Print mean values along each axis
    print(f'Mean values of marker_nodes:')
    print(f'X-axis mean: {np.mean(marker_nodes[:, 0]):.4f}')
    print(f'Y-axis mean: {np.mean(marker_nodes[:, 1]):.4f}')
    print(f'Z-axis mean: {np.mean(marker_nodes[:, 2]):.4f}')
    # all_nodes[surface_id_np][np.unique(interp_idx_flat)]

if False:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector([node_coordinates[x] for x in surface_node_tags])
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd])

if True:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(node_coordinates)
    mesh.triangles = o3d.utility.Vector3iVector(surface_triangles)
    mesh.compute_vertex_normals()
    mesh = mesh.remove_duplicated_triangles()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)