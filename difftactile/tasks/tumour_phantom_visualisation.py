import taichi as ti
import numpy as np
import json

class ContactVisualisation:
    def __init__(self):
        self.off_screen = False
        self.enable_gui1 = False
        self.enable_gui2 = True
        self.enable_gui3 = False
        self.view_phi = 0
        self.view_theta = 0
        self.table_scale = 2.0
        self.table_height = 0.0

    def init_visualisation(self):
        self.draw_pos2 = ti.Vector.field(2, float, self.vitactip.num_vertices)
        self.draw_pos3 = ti.Vector.field(2, float, self.phantom.actual_total_num_particles)
        self.draw_tableline = ti.Vector.field(3, dtype=float, shape=(2 * 4))
        self.sensor_points = ti.Vector.field(
            3, dtype=float, shape=(self.vitactip.num_vertices)
        )
        
        self.key_points = ti.Vector.field(3, dtype=ti.f32, shape=(1,), needs_grad=False)
        
        self.healthy_tissue_points = ti.Vector.field(
            3, dtype=float, shape=(self.phantom.actual_total_num_particles)
        )
        self.tumour_points = ti.Vector.field(
            3, dtype=float, shape=(self.phantom.actual_total_num_particles)
        )

    @ti.kernel
    def reset_3d_scene(self):
        self.healthy_tissue_points.fill(0)
        self.tumour_points.fill(0)

    @ti.kernel
    def draw_3d_scene(self, f: ti.i32):
        for p in range(self.phantom.actual_total_num_particles):
            if self.phantom.titles[p] == 0:
                self.healthy_tissue_points[p] = self.phantom.particle_position[f, p]
            elif self.phantom.titles[p] == 1:
                self.tumour_points[p] = self.phantom.particle_position[f, p]

        for p in range(self.vitactip.num_vertices):
            self.sensor_points[p] = self.vitactip.vertex_positions_deformed[f, p]

    def draw_markers(self, init_markers, cur_markers, gui):
        img_height = 480
        img_width = 640
        # Scale x by img_width and y by img_height
        draw_points = init_markers.copy()
        draw_points[:, 0] = draw_points[:, 0] / img_width
        draw_points[:, 1] = draw_points[:, 1] / img_height
        offset = cur_markers.copy()
        offset[:, 0] = offset[:, 0] / img_width
        offset[:, 1] = offset[:, 1] / img_height
        offset = offset - draw_points
        if not self.off_screen:
            # Draw circle outline using line segments
            circle_centre = np.array([359, 266])
            circle_radius = 180
            
            # Generate 100 points around the circle
            theta = np.linspace(0, 2*np.pi, 100)
            points_x = circle_centre[0] + circle_radius * np.cos(theta)
            points_y = circle_centre[1] + circle_radius * np.sin(theta)
            points = np.stack([points_x, points_y], axis=1)
            points[:, 0] /= img_width
            points[:, 1] /= img_height
            
            # Create begin and end points for lines (connecting consecutive points)
            begin_points = points[:-1]  # All points except the last
            end_points = points[1:]     # All points except the first
            # Add final line connecting last point to first point
            begin_points = np.vstack([begin_points, points[-1]])
            end_points = np.vstack([end_points, points[0]])
            
            # Draw the lines
            gui.lines(begin=begin_points, end=end_points, radius=1, color=0xFFFFFF)
            
            gui.circles(draw_points, radius=2, color=0xF542A1)
            gui.arrows(draw_points, offset, radius=2, color=0xE6C949)
    
    @ti.kernel
    def draw_perspective(self, f: ti.i32):
        phi, theta = (
            ti.math.radians(self.view_phi),
            ti.math.radians(self.view_theta),
        )
        c_p, s_p = ti.math.cos(phi), ti.math.sin(phi)
        c_t, s_t = ti.math.cos(theta), ti.math.sin(theta)
        offset = 0.2
        for i in range(self.vitactip.num_vertices):
            x, y, z = (
                self.vitactip.vertex_positions_deformed[f, i][0] - offset,
                self.vitactip.vertex_positions_deformed[f, i][1] - offset,
                self.vitactip.vertex_positions_deformed[f, i][2] - offset,
            )
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos2[i][0] = u + 0.2
            self.draw_pos2[i][1] = v + 0.5
        for i in range(self.phantom.actual_total_num_particles):
            x, y, z = (
                self.phantom.particle_position[f, i][0] - offset,
                self.phantom.particle_position[f, i][1] - offset,
                self.phantom.particle_position[f, i][2] - offset,
            )
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos3[i][0] = u + 0.2
            self.draw_pos3[i][1] = v + 0.5

    def draw_triangles(self, sensor, gui, f, tphi, ttheta, viz_scale, viz_offset):
        inv_trans_h = sensor.trans_h[None].inverse()
        pos_ = sensor.vertex_positions_deformed.to_numpy()[f, :]
        init_pos_ = sensor.virtual_pos.to_numpy()[f, :]
        ones = np.ones((pos_.shape[0], 1))
        hom_pos_ = np.hstack((pos_, ones))
        c_pos_ = np.matmul(inv_trans_h, hom_pos_.T).T[:, 0:3]
        hom_pos_ = np.hstack((init_pos_, ones))
        v_pos_ = np.matmul(inv_trans_h, hom_pos_.T).T[:, 0:3]
        phi, theta = np.radians(tphi), np.radians(ttheta)
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_seg_ = sensor.contact_seg.to_numpy()
        a, b, c = c_pos_[c_seg_[:, 0]], c_pos_[c_seg_[:, 1]], c_pos_[c_seg_[:, 2]]
        x = a[:, 0]
        y = a[:, 1]
        z = a[:, 2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        ua, va = xx + 0.2, y * c_t + zz * s_t + 0.5
        x = b[:, 0]
        y = b[:, 1]
        z = b[:, 2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        ub, vb = xx + 0.2, y * c_t + zz * s_t + 0.5
        x = c[:, 0]
        y = c[:, 1]
        z = c[:, 2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        uc, vc = xx + 0.2, y * c_t + zz * s_t + 0.5
        pa, pb, pc = c_pos_[c_seg_[:, 0]], c_pos_[c_seg_[:, 1]], c_pos_[c_seg_[:, 2]]
        ba, bb, bc = v_pos_[c_seg_[:, 0]], v_pos_[c_seg_[:, 1]], v_pos_[c_seg_[:, 2]]
        oa, ob, oc = pa[:, 1] - ba[:, 1], pb[:, 1] - bb[:, 1], pc[:, 1] - bc[:, 1]
        k = -1 * (oa + ob + oc) * (1 / 3) * 1.0
        gb = 0.5
        gui.triangles(
            viz_scale * np.array([ua, va]).T + viz_offset,
            viz_scale * np.array([ub, vb]).T + viz_offset,
            viz_scale * np.array([uc, vc]).T + viz_offset,
            color=ti.rgb_to_hex([k + gb, gb, gb]),
        )
        ext_f = sensor.external_force_field.to_numpy()[f, :]
        in_contact_flag = np.sum(np.abs(ext_f), axis=1) > 0
        if np.sum(in_contact_flag) > 0:
            in_c_pos = c_pos_[in_contact_flag, :]
            x = in_c_pos[:, 0]
            y = in_c_pos[:, 1]
            z = in_c_pos[:, 2]
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            ui, vi = xx + 0.2, y * c_t + zz * s_t + 0.5
            avg_pos = np.mean(in_c_pos, axis=0)
            x = avg_pos[0]
            y = avg_pos[1]
            z = avg_pos[2]
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            ua, va = xx + 0.2, y * c_t + zz * s_t + 0.5
            gui.circles(
                viz_scale * np.array([ui, vi]).T + viz_offset, radius=2, color=0xF542A1
            )
            gui.circle(
                viz_scale * np.array([ua, va]).T + viz_offset, radius=5, color=0xE6C949
            )

    def draw_table(self):
        c1 = ti.Vector([-self.table_scale, self.table_height, -self.table_scale])
        c2 = ti.Vector([-self.table_scale, self.table_height, self.table_scale])
        c3 = ti.Vector([self.table_scale, self.table_height, self.table_scale])
        c4 = ti.Vector([self.table_scale, self.table_height, -self.table_scale])
        self.draw_tableline[0] = c1
        self.draw_tableline[1] = c2
        self.draw_tableline[2] = c2
        self.draw_tableline[3] = c3
        self.draw_tableline[4] = c3
        self.draw_tableline[5] = c4
        self.draw_tableline[6] = c4
        self.draw_tableline[7] = c1

    def set_up_gui(self):
        if self.off_screen:
            return None
        else:
            screen_width = 1920
            screen_height = 1080
            grid_rows = 2
            grid_cols = 2
            window_width = screen_width // grid_cols
            window_height = screen_height // grid_rows
            window_res = (int(window_width * 0.75), int(window_height * 0.75))
            window = ti.ui.Window("high-level camera", (int(screen_width * 0.8), screen_height // 2))
            canvas = window.get_canvas()
            canvas.set_background_color((0, 0, 0))
            scene = ti.ui.Scene()
            camera = ti.ui.Camera()
            camera.projection_mode(ti.ui.ProjectionMode.Perspective)
            x, y, z = self.vitactip_tip_pose[:3]
            camera.position(x-0.5, y, z)
            camera.up(0, 0, 1)
            camera.lookat(x, y, z)
            camera.fov(5)
            if self.enable_gui1:
                gui1 = ti.GUI("low-level camera", res=window_res)
            else:
                gui1 = None
            if self.enable_gui2:
                gui2 = ti.GUI("tactile readout 1", res=window_res)
            else:
                gui2 = None
            if self.enable_gui3:
                gui3 = ti.GUI("tactile readout 2", res=window_res)
            else:
                gui3 = None
            
            self.gui_tuple = (gui1, gui2, gui3, camera, scene, window, canvas)

    def update_gui(self):
        if self.off_screen:
            return
        gui1, gui2, gui3, camera, scene, window, canvas = self.gui_tuple

        keypoint_coords = self.vitactip.get_keypoint_coordinates(0, self.keypoint_indices[-1].reshape((1,)))
        # keypoint_coords = self.trajectory_npy

        viz_scale = 0.1
        viz_offset = [0.25, 0.25]
        viz_scale_deformation_map = viz_scale * 3
        viz_offset_deformation_map = [0.5, 0.5]
        f_deformation = 0
        r1_deformation = -90
        r2_deformation = 90
        if not self.off_screen:
            self.vitactip.extract_markers(0)
            init_2d = self.vitactip.undeformed_markers.to_numpy()
            marker_2d = self.vitactip.deformed_markers.to_numpy()
            if self.enable_gui2:
                self.draw_markers(init_2d, marker_2d, gui2)
        if not self.off_screen:
            self.draw_perspective(0)
            if self.enable_gui1:
                gui1.circles(
                    viz_scale * self.draw_pos3.to_numpy() + viz_offset,
                    radius=2,
                    color=0x039DFC,
                )
                gui1.circles(
                    viz_scale * self.draw_pos2.to_numpy() + viz_offset,
                    radius=2,
                    color=0xE6C949,
                )
            if self.enable_gui3:
                self.draw_triangles(
                    self.vitactip,
                    gui3,
                    f_deformation,
                    r1_deformation,
                    r2_deformation,
                    viz_scale_deformation_map,
                    viz_offset_deformation_map,
                )
            if self.enable_gui1:
                gui1.show()
            if self.enable_gui2:
                gui2.show()
            if self.enable_gui3:
                gui3.show()
            
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            self.draw_3d_scene(0)

            sf = 100
            
            scene.particles(
                self.healthy_tissue_points,
                color=(0.0, 0.0, 1.0),
                radius=0.01 / sf,
            )
            scene.particles(
                self.tumour_points,
                color=(1.0, 1.0, 0.0),
                radius=0.02 / sf,
            )
            scene.particles(
                self.sensor_points,
                color=(0.0, 1.0, 0.0),
                radius=0.005 / sf,
            )
            
            assert keypoint_coords.shape[0] == self.key_points.shape[0], f"Set self.key_points to shape ({keypoint_coords.shape[0]},)"
            if keypoint_coords is not None:
                self.key_points.from_numpy(keypoint_coords)
                scene.particles(
                    self.key_points,
                    color=(1.0, 0.0, 0.0),
                    radius=0.05 / sf,
                )
            
            canvas.scene(scene)
            window.show()
