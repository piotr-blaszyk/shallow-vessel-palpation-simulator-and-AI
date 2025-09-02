import open3d as o3d
import numpy as np


class TemporalPointCloudVisualizer:
    def __init__(self, points_3d: np.ndarray):
        """
        points_3d: numpy array of shape (num_timesteps, num_points, 3)
        """
        assert points_3d.ndim == 3, "Shape must be (num_timesteps, num_points, 3)"
        self.points_3d = points_3d
        self.num_timesteps = points_3d.shape[0]
        self.current_t = 0

        # Create point cloud and axis mesh
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points_3d[self.current_t])

        _min = np.min(points_3d, axis=(0, 1))
        _max = np.max(points_3d, axis=(0, 1))
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

    def _set_camera(self, vis):
        ctr = vis.get_view_control()
        bbox = self.pcd.get_axis_aligned_bounding_box()

        # Look at the point cloud center
        center = bbox.get_center()
        eye = center + np.array([0, 0, 5])  # place camera along +Z
        up = np.array([0, -1, 0])           # make +Y go downward

        ctr.look_at(center, eye, up)

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(self.pcd)
        vis.add_geometry(self.axes)

        vis.register_key_callback(ord("K"), self._next)
        vis.register_key_callback(ord("J"), self._prev)

        # Set camera after geometries are added
        self._set_camera(vis)

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    # Example: 5 timesteps, 100 points each
    num_timesteps, num_points = 5, 100
    points_3d = np.random.randn(num_timesteps, num_points, 3)

    viz = TemporalPointCloudVisualizer(points_3d)
    viz.run()
