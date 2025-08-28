import numpy as np
import open3d as o3d
import pickle
import pyvista as pv

from difftactile.main.constants import *

class VisualiseMesh:
    def __init__(self):
        # self.load_tetrahedra()
        # self.load_sensor_mesh_from_npz()
        # self.load_is_fixed_layer()
        # self.apply_is_fixed_layer()
        self.load_gmsh_data_only()
    
    def load_gmsh_data_only(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh, 'rb') as f:
            mesh_data = pickle.load(f)
        with open(SYSTEM_PARAMS.files.edge_lengths_pkl, 'rb') as f:
            edge_length_data = pickle.load(f)
        self.points = mesh_data['node_coordinates']
        self.tetrahedra = mesh_data['all_tetrahedra']
        node_edge_lengths = edge_length_data['node_edge_lengths']
        tetra_edge_lengths = edge_length_data['tetra_edge_lengths']
        self.tetra_edge_lengths = np.array(tetra_edge_lengths, dtype=float)
        node_edge_lengths = [max(xs) for xs in node_edge_lengths]
        self.node_edge_lengths = np.array(node_edge_lengths, dtype=float)
        assert self.node_edge_lengths.shape[0] == self.points.shape[0]
    
    def load_sensor_mesh_from_npz(self):
        path = SYSTEM_PARAMS.files.sensor_mesh
        data = np.load(path)
        self.points = data['particles']
    
    def load_exp_unfiltered(self):
        frames_poses_data = np.load(SYSTEM_PARAMS.files.experiment_2025_08_22_output_npz)
        frames_poses = frames_poses_data['output']
        self.points = frames_poses[:, 1:4]
    
    def load_exp_filtered(self):
        data = np.load(SYSTEM_PARAMS.files.experiment_2025_08_22_filtered_positions_npz)
        points = data['points']
        self.points = points
    
    def load_points_E(self):
        with open(SYSTEM_PARAMS.files.vitactip_points_E, 'rb') as f:
            self.points = pickle.load(f)

    def load_validation_point(self):
        with open(SYSTEM_PARAMS.files.validation_point_E, 'rb') as f:
            self.validation_point = pickle.load(f)
    
    def load_tetrahedra(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh, 'rb') as f:
            self.mesh_data = pickle.load(f)
        self.tetrahedra = self.mesh_data['all_tetrahedra']

    def load_default_undeformed_points(self):
        with open(SYSTEM_PARAMS.files.initial_vertex_positions_undeformed, 'rb') as f:
            self.points = pickle.load(f)
        print(f'number of nan vertices: {np.sum(np.isnan(self.points))}')
    
    def load_deformed_points(self):
        with open(SYSTEM_PARAMS.files.deformed_node_coordinates.format(43), 'rb') as f:
            self.points = pickle.load(f)
        print(f'number of nan vertices: {np.sum(np.isnan(self.points))}')
    
    def load_is_fixed_layer(self):
        path = SYSTEM_PARAMS.files.is_fixed_layer_npz
        data = np.load(path)
        self.is_fixed_layer = data['is_fixed_layer']
    
    def apply_is_fixed_layer(self):
        self.points = self.points[self.is_fixed_layer == 1]
    
    def use_dome_surface_points(self):
        # self.point_coordinates = self.point_coordinates[self.mesh_data['dome_surface_node_tags']]
        self.points = self.points[[23, 31]]

    def remove_nans_and_filter_on_material(self):
        nan_nodes = np.where(np.any(np.isnan(self.points), axis=1))[0]
        valid_nodes = np.setdiff1d(np.arange(len(self.points)), nan_nodes)
        old_to_new_idx = np.full(len(self.points), -1)
        old_to_new_idx[valid_nodes] = np.arange(len(valid_nodes))
        self.points = self.points[valid_nodes]
        all_tetrahedra_new_idx = np.array([[old_to_new_idx[i] for i in tetra] for tetra in self.mesh_data['all_tetrahedra']])
        good_tetrahedra = []
        for tetra in all_tetrahedra_new_idx:
            if np.all(tetra != -1):
                good_tetrahedra.append(tetra)
        good_tetrahedra = np.array(good_tetrahedra)
        self.tetrahedra = good_tetrahedra
    
    def merge(self):
        self.points = np.vstack([
            self.points,
            self.validation_point
        ])

    def visualise_point_cloud(self):
        _min = np.min(self.points, axis=0)
        _max = np.max(self.points, axis=0)
        diff = _max-_min

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        if hasattr(self, 'node_edge_lengths'):
            edge_lengths = self.node_edge_lengths
            min_val = np.min(edge_lengths)
            max_val = np.max(edge_lengths)
            normalized_values = (edge_lengths - min_val) / (max_val - min_val)
            
            colors = np.zeros((len(self.points), 3))
            colors[:, 0] = normalized_values
            colors[:, 1] = 1 - normalized_values
            colors[:, 2] = 0.5
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
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
        mesh.vertices = o3d.utility.Vector3dVector(self.points)
        mesh.triangles = o3d.utility.Vector3iVector(tetrahedra_decomposed)
        mesh.compute_vertex_normals()
        mesh = mesh.remove_duplicated_triangles()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def visualise_tetrahedra_pyvista(self):
        cells = np.column_stack([np.full(len(self.tetrahedra), 4), self.tetrahedra])
        grid = pv.PolyData(self.points, cells)
        tetra_max_edge_lengths = np.max(self.tetra_edge_lengths, axis=1)
        grid.cell_data['max_edge_lengths'] = tetra_max_edge_lengths
        
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid, 
            scalars='max_edge_lengths',
            show_edges=False,  # Hide edges to see if it's a solid mesh
            cmap='viridis',
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Max Tetrahedron Edge Length'},
            opacity=1.0,  # Fully opaque
            lighting=True,  # Enable lighting for better depth perception
            smooth_shading=True  # Smooth shading for better appearance
        )
        plotter.add_axes()
        
        # Set camera position for better initial view
        plotter.camera_position = 'iso'
        
        print("Visualization tips:")
        print("- If you can see inside the mesh, there might be missing surface tetrahedra")
        print("- Try rotating the view to see all sides")
        print("- Check the diagnostic output above for mesh quality issues")
        
        plotter.show()
    
    def check_mesh_quality(self):
        """Diagnostic method to check mesh quality"""
        print(f"Mesh Statistics:")
        print(f"  Number of vertices: {len(self.points)}")
        print(f"  Number of tetrahedra: {len(self.tetrahedra)}")
        
        # Check for inverted tetrahedra
        volumes = self.compute_tetrahedra_volumes()
        inverted_count = np.sum(volumes < 0)
        print(f"  Inverted tetrahedra: {inverted_count}")
        
        # Check for surface tetrahedra
        surface_tetrahedra = self.find_surface_tetrahedra()
        print(f"  Surface tetrahedra: {len(surface_tetrahedra)}")
        
        # Check edge length statistics
        if hasattr(self, 'tetra_edge_lengths'):
            max_edge_lengths = np.max(self.tetra_edge_lengths, axis=1)
            print(f"  Max edge length - min: {np.min(max_edge_lengths):.6f}, max: {np.max(max_edge_lengths):.6f}")
    
    def compute_tetrahedra_volumes(self):
        """Compute volumes of all tetrahedra"""
        volumes = []
        for tetra in self.tetrahedra:
            v0, v1, v2, v3 = self.points[tetra]
            # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
            volume = np.abs(np.linalg.det([v1-v0, v2-v0, v3-v0])) / 6
            volumes.append(volume)
        return np.array(volumes)
    
    def find_surface_tetrahedra(self):
        """Find tetrahedra that have at least one face on the surface"""
        # This is a simplified approach - you might need more sophisticated surface detection
        surface_tetrahedra = []
        face_counts = {}
        
        for i, tetra in enumerate(self.tetrahedra):
            # Get all faces of this tetrahedron
            faces = [
                tuple(sorted([tetra[0], tetra[1], tetra[2]])),
                tuple(sorted([tetra[0], tetra[1], tetra[3]])),
                tuple(sorted([tetra[0], tetra[2], tetra[3]])),
                tuple(sorted([tetra[1], tetra[2], tetra[3]]))
            ]
            
            for face in faces:
                if face in face_counts:
                    face_counts[face] += 1
                else:
                    face_counts[face] = 1
        
        # Tetrahedra with at least one face that appears only once are surface tetrahedra
        for i, tetra in enumerate(self.tetrahedra):
            faces = [
                tuple(sorted([tetra[0], tetra[1], tetra[2]])),
                tuple(sorted([tetra[0], tetra[1], tetra[3]])),
                tuple(sorted([tetra[0], tetra[2], tetra[3]])),
                tuple(sorted([tetra[1], tetra[2], tetra[3]]))
            ]
            
            if any(face_counts[face] == 1 for face in faces):
                surface_tetrahedra.append(i)
        
        return surface_tetrahedra


def main():
    visualise_mesh = VisualiseMesh()
    
    # Run diagnostic check first
    print("Running mesh quality diagnostics...")
    visualise_mesh.check_mesh_quality()
    print()
    
    # Then visualize
    visualise_mesh.visualise_tetrahedra_pyvista()


if __name__ == '__main__':
    main()
    
