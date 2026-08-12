import os
import pickle

import numpy as np
import open3d as o3d
import pyvista as pv
import vedo
from scipy.spatial.distance import pdist
from scipy.spatial import Delaunay

from difftactile.main.constants import *
from difftactile.main.display import is_interactive, show_plotter
from difftactile.sensor_model.fisheye_model_no_taichi import *
from difftactile.main.paths import repo_path


class VisualiseMesh:
    def __init__(self):
        self.load_gmsh_data_only()
        # self.load_vitactip_mesh_video()
        self.load_is_fixed_layer()
        self.apply_is_fixed_layer()
        self.tetrahedra = self.alpha_shape_3d(
            points=self.points,
            alpha=5.0,
        )
    
    def alpha_shape_3d(self, points, alpha):
        if len(points) < 4:
            raise ValueError("Need at least 4 points")

        delaunay = Delaunay(points)
        tetrahedra = delaunay.simplices  # (num_tetrahedra, 4)

        A = points[tetrahedra[:,0]]
        B = points[tetrahedra[:,1]]
        C = points[tetrahedra[:,2]]
        D = points[tetrahedra[:,3]]

        # Relative vectors
        AB, AC, AD = B - A, C - A, D - A
        AB2 = np.sum(AB**2, axis=1)
        AC2 = np.sum(AC**2, axis=1)
        AD2 = np.sum(AD**2, axis=1)

        # Build system: M x = rhs
        M = np.stack([AB, AC, AD], axis=1)        # (n,3,3)
        rhs = np.stack([AB2, AC2, AD2], axis=1)   # (n,3)

        # Solve per tetrahedron
        try:
            invM = np.linalg.inv(M)               # (n,3,3)
            centers = np.einsum("nij,nj->ni", invM, rhs / 2.0)
        except np.linalg.LinAlgError:
            centers = np.full_like(rhs, np.inf)

        radii = np.linalg.norm(centers, axis=1)

        mask = radii < alpha
        return tetrahedra[mask]
    
    def load_silicone_exp_points(self):
        path = 'difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_sim_format_poses/metadata_0.0_0.0.npz'
        data = np.load(path)
        points = data['vein_polyline']
        points = points[0]
        points = points.reshape(-1, 2)
        points_E = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
            ps=points,
            dist_lens_to_plane=0.019-0.003
        )
        self.points = points_E
    
    def load_vein_points(self):
        path = SYSTEM_PARAMS.files.vein_points_npz
        data = np.load(path)
        self.points = data['points']
    
    def load_phantom_points(self):
        path = SYSTEM_PARAMS.files.phantom_points_npz
        data = np.load(path)
        self.points = data['points']
    
    def load_vein_mesh(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vein_pkl, 'rb') as f:
            data = pickle.load(f)
        self.points = data['node_coordinates']
        self.tetrahedra = data['all_tetrahedra']
        self.triangles = data['surface_triangles']
    
    def load_vitactip_mesh_video(self):
        path = SYSTEM_PARAMS.files.vitactip_mesh_npz
        data = np.load(path)
        self.all_points = data['all_points']
    
    def load_gmsh_surface(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vitactip_pkl, 'rb') as f:
            mesh_data = pickle.load(f)
        surface_node_tags = mesh_data['surface_node_tags']
        points = mesh_data['node_coordinates']
        # mask = np.ones(len(points), dtype=bool)
        # mask[surface_node_tags] = False
        # points = points[mask]
        self.points = points
        self.triangles = mesh_data['surface_triangles']
    
    def debug_gmsh(self):
        with open(SYSTEM_PARAMS.files.gmsh_debug_pkl, 'rb') as f:
            mesh_data = pickle.load(f)
        self.points = mesh_data['node_coordinates']
        self.tetrahedra = mesh_data['all_tetrahedra']
    
    def load_gmsh_data_only(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vitactip_pkl, 'rb') as f:
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
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vitactip_pkl, 'rb') as f:
            self.mesh_data = pickle.load(f)
        self.tetrahedra = self.mesh_data['all_tetrahedra']
    
    def load_triangles(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vitactip_pkl, 'rb') as f:
            self.mesh_data = pickle.load(f)
        self.triangles = self.mesh_data['surface_triangles']

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
        mask = self.is_fixed_layer == 0
        self.points = self.points[mask]
    
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
        
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=np.max(diff)/2, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, axes])

    def visualise_tetrahedra(self):
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
    
    def visualise_triangles(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.points)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.compute_vertex_normals()
        mesh = mesh.remove_duplicated_triangles()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    def pyvista_visualise_msh_file(self):
        path = SYSTEM_PARAMS.files.gmsh_debug_msh
        try:
            mesh = pv.read(path)
        except Exception as e:
            import meshio
            mesh_io = meshio.read(path)
            mesh = pv.wrap(meshio.Mesh(points=mesh_io.points, cells=mesh_io.cells))
        volume = mesh.extract_cells(mesh.celltypes == pv.CellType.TETRA)

        p = pv.Plotter(shape=(1, 2))
        p.subplot(0, 0)
        p.add_text("Raw tetrahedra", font_size=10)
        p.add_mesh(volume, show_edges=True, opacity=0.5)

        surface = volume.extract_surface()
        p.subplot(0, 1)
        p.add_text("Extracted surface", font_size=10)
        p.add_mesh(surface, color="lightblue", smooth_shading=True)

        show_plotter(p, repo_path("difftactile/output/mesh_raw_and_surface.png"))

    def visualise_tetrahedra_pyvista(self):
        cells = np.column_stack([np.full(len(self.tetrahedra), 4), self.tetrahedra])
        grid = pv.PolyData(self.points, cells)
        tetra_max_edge_lengths = np.max(self.tetra_edge_lengths, axis=1)
        grid.cell_data['max_edge_lengths'] = tetra_max_edge_lengths
        
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid,
            scalars='max_edge_lengths',
            cmap='viridis',
            show_scalar_bar=True,
            opacity=1.0,
            lighting=True,
        )
        plotter.add_axes()
        plotter.camera_position = 'iso'
        show_plotter(plotter, repo_path("difftactile/output/mesh_tetrahedra.png"))
    
    def visualise_tetrahedra_pyvista_for_video(self):
        num_tets = self.tetrahedra.shape[0]
        cells = np.hstack([np.full((num_tets, 1), 4), self.tetrahedra]).flatten()
        grid = pv.UnstructuredGrid(cells, np.full(num_tets, 10), self.points)  # 10 = VTK_TETRA
        surface = grid.extract_surface()
        plotter = pv.Plotter()
        plotter.add_mesh(surface, color="lightblue", show_edges=True)
        show_plotter(plotter, repo_path("difftactile/output/mesh_surface.png"))
    
    def visualise_tetrahedra_vedo(self):
        mesh = vedo.TetMesh([self.points, self.tetrahedra])
        surface = mesh.tomesh()
        # vedo.show() blocks on its own window; only open one when asked to.
        if is_interactive():
            vedo.show(surface)
        else:
            out = repo_path("difftactile/output/mesh_vedo.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            vedo.show(surface, offscreen=True).screenshot(out).close()
            print(f"3-D view written to: {out}")

    def visualize_sequence_from_tetrahedra_pyvista(self):
        dilation = 1
        nodes = self.all_points[::dilation]
        tetrahedra = self.tetrahedra
        num_frames = nodes.shape[0]
        frame_idx = [0]
        surfaces = []
        for f in range(num_frames):
            pts = nodes[f]
            num_tets = tetrahedra.shape[0]
            cells = np.hstack([np.full((num_tets, 1), 4), tetrahedra]).ravel()
            grid = pv.UnstructuredGrid(cells, np.full(num_tets, 10), pts)  # 10 = VTK_TETRA
            surfaces.append(grid.extract_surface())
        plotter = pv.Plotter()
        actor = plotter.add_mesh(surfaces[frame_idx[0]], color="lightblue", show_edges=True)
        text_actor = plotter.add_text(
            f"Frame {frame_idx[0]+1}/{num_frames}", position="upper_edge", font_size=14
        )
        def update_scene():
            nonlocal actor, text_actor
            plotter.remove_actor(actor)
            actor = plotter.add_mesh(surfaces[frame_idx[0]], color="lightblue", show_edges=True)
            plotter.remove_actor(text_actor)
            text_actor = plotter.add_text(
                f"Frame {frame_idx[0]+1}/{num_frames}", position="upper_edge", font_size=14
            )
            plotter.render()
        def keypress_j():
            frame_idx[0] = (frame_idx[0] - 1) % num_frames
            update_scene()
        def keypress_k():
            frame_idx[0] = (frame_idx[0] + 1) % num_frames
            update_scene()
        plotter.add_key_event("j", keypress_j)
        plotter.add_key_event("k", keypress_k)
        # The j/k frame stepping needs a user; non-interactively this just
        # captures the first frame rather than waiting for keys nobody presses.
        show_plotter(plotter, repo_path("difftactile/output/mesh_sequence_frame0.png"))


def camera_to_world_transform(
    x_robot: float, 
    y_robot: float, 
    z_robot: float, 
    d: float
) -> np.ndarray:
    R = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=float)
    t = np.array([x_robot, y_robot, z_robot + d], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_silicone_exp_points():
    path = 'difftactile/manual_or_experimental_data/silicone_training_data/20250901-131547_sim_format_poses/metadata_0.0_0.0.npz'
    data = np.load(path)
    points = data['vein_polyline']
    mask = data['vein_polyline_mask']
    # points = points[0][mask[0]]
    # points = points.reshape(points.shape[0], -1, 2)
    # points = points[0]
    points_E = FisheyeModelNoTaichi.project_pix_to_points_3d_plane(
        ps=points,
        dist_lens_to_plane=0.019-0.003
    )
    trans = camera_to_world_transform(
        0, 0, 0, 0.019
    )
    foo = []
    for i in range(points_E.shape[0]):
        xs = points_E[i][mask[i]]
        # max_dist = pdist(xs).max()
        points_hom = np.hstack([xs, np.ones((xs.shape[0], 1))])
        transformed_hom = points_hom @ trans.T
        transformed_points = transformed_hom[:, :3]
        foo.append(transformed_points)

    return foo


class TemporalPointCloudVisualizer:
    def __init__(self, points_3d):
        self.points_3d = points_3d
        self.num_timesteps = len(points_3d)
        self.current_t = 0

        # Create point cloud and axis mesh
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points_3d[self.current_t])

        _min = np.min(points_3d[0], axis=0)
        _max = np.max(points_3d[1], axis=0)
        diff = _max - _min
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=np.max(diff) / 2, origin=[0, 0, 0]
        )
    
    def _update(self, vis):
        self.pcd.points = o3d.utility.Vector3dVector(self.points_3d[self.current_t])
        vis.update_geometry(self.pcd)
        vis.update_renderer()

    def _next(self, vis):
        self.current_t = (self.current_t + 1) % self.num_timesteps
        self._update(vis)
        return False

    def _prev(self, vis):
        self.current_t = (self.current_t - 1) % self.num_timesteps
        self._update(vis)
        return False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(self.pcd)
        vis.add_geometry(self.axes)

        vis.register_key_callback(ord("K"), self._next)
        vis.register_key_callback(ord("J"), self._prev)

        vis.run()
        vis.destroy_window()


def main():
    visualise_mesh = VisualiseMesh()
    visualise_mesh.visualise_tetrahedra_vedo()

    # points_3d = load_silicone_exp_points()
    # viz = TemporalPointCloudVisualizer(points_3d)
    # viz.run()


if __name__ == '__main__':
    main()
    
