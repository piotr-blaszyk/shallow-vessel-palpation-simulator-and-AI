import numpy as np
import open3d as o3d
import pickle

from difftactile.main.constants import *

class VisualiseMesh:
    def __init__(self):
        pass
    
    def load_points_E(self):
        with open(SYSTEM_PARAMS.files.vitactip_points_E, 'rb') as f:
            self.point_coordinates = pickle.load(f)

    def load_validation_point(self):
        with open(SYSTEM_PARAMS.files.validation_point_E, 'rb') as f:
            self.validation_point = pickle.load(f)
    
    def load_tetrahedra(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh, 'rb') as f:
            self.mesh_data = pickle.load(f)
        self.tetrahedra = self.mesh_data['all_tetrahedra']

    def load_default_undeformed_points(self):
        with open(SYSTEM_PARAMS.files.initial_vertex_positions_undeformed, 'rb') as f:
            self.point_coordinates = pickle.load(f)
        print(f'number of nan vertices: {np.sum(np.isnan(self.point_coordinates))}')
    
    def load_deformed_points(self):
        with open(SYSTEM_PARAMS.files.deformed_node_coordinates.format(119), 'rb') as f:
            self.point_coordinates = pickle.load(f)
        print(f'number of nan vertices: {np.sum(np.isnan(self.point_coordinates))}')
    
    def load_is_fixed_layer(self):
        with open(SYSTEM_PARAMS.files.is_fixed_layer, 'rb') as f:
            self.is_fixed_layer = pickle.load(f)
    
    def apply_is_fixed_layer(self):
        # Filter point coordinates to keep only those where is_fixed_layer is True
        self.point_coordinates = self.point_coordinates[self.is_fixed_layer == 1]
    
    def use_dome_surface_points(self):
        # self.point_coordinates = self.point_coordinates[self.mesh_data['dome_surface_node_tags']]
        self.point_coordinates = self.point_coordinates[[23, 31]]

    def remove_nans_and_filter_on_material(self):
        nan_nodes = np.where(np.any(np.isnan(self.point_coordinates), axis=1))[0]
        valid_nodes = np.setdiff1d(np.arange(len(self.point_coordinates)), nan_nodes)
        old_to_new_idx = np.full(len(self.point_coordinates), -1)
        old_to_new_idx[valid_nodes] = np.arange(len(valid_nodes))
        self.point_coordinates = self.point_coordinates[valid_nodes]
        all_tetrahedra_new_idx = np.array([[old_to_new_idx[i] for i in tetra] for tetra in self.mesh_data['all_tetrahedra']])
        good_tetrahedra = []
        for tetra in all_tetrahedra_new_idx:
            if np.all(tetra != -1):
                good_tetrahedra.append(tetra)
        good_tetrahedra = np.array(good_tetrahedra)
        self.tetrahedra = good_tetrahedra
    
    def merge(self):
        self.point_coordinates = np.vstack([
            self.point_coordinates,
            self.validation_point
        ])

    def visualise_point_cloud(self):
        _min = np.min(self.point_coordinates, axis=0)
        _max = np.max(self.point_coordinates, axis=0)
        diff = _max-_min

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_coordinates)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=np.max(diff)/2, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, axes])

    def visualise_mesh(self):
        tetrahedra_decomposed = []
        for tetra in self.tetrahedra:
            a, b, c, d = tetra
            tetrahedra_decomposed.append([a, b, c])
            tetrahedra_decomposed.append([a, b, d])
            tetrahedra_decomposed.append([a, c, d])
            tetrahedra_decomposed.append([b, c, d])
        tetrahedra_decomposed = np.array(tetrahedra_decomposed)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.point_coordinates)
        mesh.triangles = o3d.utility.Vector3iVector(tetrahedra_decomposed)
        mesh.compute_vertex_normals()
        mesh = mesh.remove_duplicated_triangles()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    def go(self):
        self.load_deformed_points()
        self.load_tetrahedra()
        self.visualise_mesh()


if __name__ == '__main__':
    visualise_mesh = VisualiseMesh()
    visualise_mesh.go()
