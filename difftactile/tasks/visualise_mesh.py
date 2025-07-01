import numpy as np
import open3d as o3d
import pickle
from collections import Counter

from difftactile.tasks.constants import SYSTEM_PARAMS

with open(SYSTEM_PARAMS.files.gmsh_mesh, 'rb') as f:
    mesh_data = pickle.load(f)

print('hello')
print(mesh_data.node_coordinates.shape)
print(mesh_data.all_tetrahedra.shape)

with open(SYSTEM_PARAMS.files.deformed_node_coordinates.format(SYSTEM_PARAMS.visualise_mesh.frame_number), 'rb') as f:
    deformed_node_coordinates = pickle.load(f)
print(f'ts={SYSTEM_PARAMS.visualise_mesh.frame_number}; num nan: {np.sum(np.isnan(deformed_node_coordinates))}')

nan_nodes = np.where(np.any(np.isnan(deformed_node_coordinates), axis=1))[0]

# Filter out NaN nodes and create mapping
valid_nodes = np.setdiff1d(np.arange(len(deformed_node_coordinates)), nan_nodes)
old_to_new_idx = np.full(len(deformed_node_coordinates), -1)  # Initialize with -1
old_to_new_idx[valid_nodes] = np.arange(len(valid_nodes))  # Map old indices to new indices

# Filter deformed_node_coordinates
deformed_node_coordinates = deformed_node_coordinates[valid_nodes]

# Convert all tetrahedra to new indices
all_tetrahedra_new_idx = np.array([[old_to_new_idx[i] for i in tetra] for tetra in mesh_data.all_tetrahedra])

good_tetrahedra = []
for tetra in all_tetrahedra_new_idx:
    tetra_node_labels = mesh_data.node_labels[tetra]
    gel_count = np.sum(tetra_node_labels[:, mesh_data.group_to_idx.gel])
    shell_count = np.sum(tetra_node_labels[:, mesh_data.group_to_idx.shell])
    is_gel = gel_count == 4 and shell_count <= 3
    if not is_gel:
        if np.any(tetra == -1):
            continue
        a, b, c, d = tetra
        good_tetrahedra.append([a, b, c])
        good_tetrahedra.append([a, b, d])
        good_tetrahedra.append([a, c, d])
        good_tetrahedra.append([b, c, d])
good_tetrahedra = np.array(good_tetrahedra)
print(len(good_tetrahedra) // 4)

# deformed_node_coordinates = deformed_node_coordinates[1:]
# good_tetrahedra -= 1

# with open(f'../tasks/output/vitactip.interp_idx_flat.pkl', 'rb') as f:
#     interp_idx_flat = pickle.load(f)

# with open(f'../tasks/output/vitactip.cam_3d_nodes.pkl', 'rb') as f:
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

    with open(SYSTEM_PARAMS.files.vitactip_surface_id, 'rb') as f:
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
    pcd.points = o3d.utility.Vector3dVector(deformed_node_coordinates)
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd])

if True:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(deformed_node_coordinates)
    mesh.triangles = o3d.utility.Vector3iVector(good_tetrahedra)
    mesh.compute_vertex_normals()
    mesh = mesh.remove_duplicated_triangles()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)