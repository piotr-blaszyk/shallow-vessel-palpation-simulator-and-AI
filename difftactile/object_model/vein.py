import pickle

import numpy as np
import taichi as ti

from difftactile.main.constants import *
from difftactile.object_model.common import *


@ti.data_oriented
class Vein:
    def __init__(self):
        self.pose = np.array(SYSTEM_PARAMS_COMPUTED.vein_pose, dtype=float)
        self.dist_sf = SYSTEM_PARAMS.meta.distance_scaling_factor
        r = SYSTEM_PARAMS.gmsh_mm.vein.radius
        r *= 1/1_000*self.dist_sf
        self.r = r
        h = SYSTEM_PARAMS.gmsh_mm.vein.length
        h *= 1/1_000*self.dist_sf
        self.h = h
        self.load_obj()
        self.particles_A = ti.Vector.field(
            3,
            dtype=float,
            shape=(self.particles_B.shape[0],),
            needs_grad=False,
        )
        self.transform_BA()
        self.debug_vein()
        self.initialise_centerline_particles()
    
    def initialise_centerline_particles(self):
        p1 = self.pose[:3].copy()
        vec = np.array([
            self.h,
            0,
            0,
        ], dtype=float)
        p2 = p1+vec
        points = np.linspace(p1, p2, num=50)
        self.centerline_A = ti.Vector.field(
            3,
            dtype=float,
            shape=(points.shape[0],),
            needs_grad=False,
        )
        self.centerline_A.from_numpy(points)
    
    def debug_vein(self):
        points = self.particles_A.to_numpy()
        path = SYSTEM_PARAMS.files.vein_points_npz
        np.savez(
            path,
            points=points,
        )
    
    def load_obj(self):
        with open(SYSTEM_PARAMS.files.gmsh_mesh_vein_pkl, "rb") as f:
            data = pickle.load(f)
        tetrahedra = data['all_tetrahedra']
        points = data['node_coordinates']
        triangles = data['surface_triangles']

        points *= 1/1_000*self.dist_sf

        self.particles_B = ti.Vector.field(
            3, dtype=float, shape=(points.shape[0],), needs_grad=False
        )
        self.particles_B.from_numpy(points)
        self.triangles = ti.Vector.field(
            3, 
            dtype=int, 
            shape=(triangles.shape[0],), 
            needs_grad=False,
        )
        self.triangles.from_numpy(triangles)

    def transform_BA(self):
        rotation_matrix, transformation_matrix = Common.compute_transformation_matrix(self.pose)
        self.T_BA_np = transformation_matrix.copy()
        self.T_BA = ti.Matrix.field(4, 4, ti.f32, shape=(), needs_grad=False)
        self.T_BA.from_numpy(transformation_matrix)
        self.initialise_point_cloud()
        self.compute_yz_centre()
    
    @ti.kernel
    def initialise_point_cloud(self):
        for i in range(self.particles_A.shape[0]):
            particle_B = self.particles_B[i]
            particle_A = self.T_BA[None] @ ti.Vector(
                [
                    particle_B[0],
                    particle_B[1],
                    particle_B[2],
                    1.0,
                ]
            )
            self.particles_A[i] = ti.Vector(
                [
                    particle_A[0],
                    particle_A[1],
                    particle_A[2],
                ]
            )
    
    def xy_footprint(self):
        """The vein's projection onto the xy plane: (x_min, x_max, y_min, y_max).

        ANALYTICAL, not derived from the particle cloud. The vein is a cylinder
        of radius `r` and length `h` whose pose rotates its local axis onto world
        +x (the quaternion in `vein_pose` is a 90-degree rotation about y), so
        the centreline runs from `pose[:3]` to `pose[:3] + [h, 0, 0]` - exactly
        as `initialise_centerline_particles` constructs it. Projected onto xy a
        cylinder lying flat along x is a rectangle:

            x in [x0, x0 + h]        the length of the cylinder
            y in [y0 - r, y0 + r]    its diameter, seen from above

        Verified against the mesh: the particle cloud spans x [0.095, 0.295],
        y [0.185, 0.205], z [0.050, 0.070], which is exactly this rectangle for
        r = 0.01 m, h = 0.2 m at pose [0.095, 0.195, 0.06].

        Using the analytical form rather than the particles' bounding box keeps
        the region fixed and exact - the cloud's extent depends on mesh
        resolution, and the vein does not deform, so there is nothing to gain
        from re-measuring it.
        """
        x0, y0 = float(self.pose[0]), float(self.pose[1])
        return (x0, x0 + self.h, y0 - self.r, y0 + self.r)

    def apex_point(self):
        """Point p: the top of the curved wall, at the cylinder's mid-length.

        Derived analytically from the pose, radius and length - no particles are
        consulted, so the answer does not depend on mesh resolution.

        Construction, following the definition:
          * the cylinder lies flat along +x from `pose[:3]`, so its xy projection
            is the rectangle x in [x0, x0+h], y in [y0-r, y0+r];
          * that rectangle's centre is (x0 + h/2, y0);
          * the CURVED wall (not the two flat end caps) has exactly two points
            projecting there - directly above and below the axis, at z0 + r and
            z0 - r;
          * the one with the larger z is the apex.

        So p = (x0 + h/2, y0, z0 + r) = (0.195, 0.195, 0.070) for the shipped
        geometry - the highest point of the vessel, half way along it, which is
        where a sensor sliding across would first meet it.
        """
        x0, y0, z0 = (float(self.pose[0]), float(self.pose[1]),
                      float(self.pose[2]))
        return np.array([x0 + self.h / 2.0, y0, z0 + self.r], dtype=float)

    def max_z(self):
        """The vein's highest z, analytically: the axis height plus the radius.

        The cylinder lies flat along +x at height `pose[2]`, so its topmost
        surface is one radius above the axis. Verified against the mesh: the
        particle cloud's max z is 0.07000 for pose z = 0.06 and r = 0.01.
        """
        return float(self.pose[2]) + self.r

    def compute_yz_centre(self):
        centre_B = np.array([0, 0, 0], dtype=float)
        centre_B_h = np.append(centre_B, 1.0)
        centre_A_h = self.T_BA_np @ centre_B_h
        self.centre_A = centre_A_h[:3]
        self.centre_A_yz_ti = ti.Vector.field(
            2,
            shape=(),
            dtype=float,
            needs_grad=False,
        )
        self.centre_A_yz_ti[None] = ti.Vector(self.centre_A[1:])
    
    def get_vein_mask(self, points):
        yz_distances = np.sqrt(
            (points[:, 1] - self.centre_A[1])**2 + 
            (points[:, 2] - self.centre_A[2])**2
        )
        radius_mask = yz_distances <= self.r
        length_mask = (points[:, 0] >= self.centre_A[0]) & (points[:, 0] <= self.centre_A[0] + self.h)
        return radius_mask
    
    @ti.func
    def find_closest(self, grid_p):
        cur_min_offset = SYSTEM_PARAMS.vitactip.collision_search_distance
        cur_min_idx = -1
        for k in range(self.triangles.shape[0]):
            a, b, c = self.triangles[k]
            p_1 = self.particles_A[a]
            p_2 = self.particles_A[b]
            p_3 = self.particles_A[c]
            p_c = 1 / 3 * (p_1 + p_2 + p_3)
            offset_p = (p_c - grid_p).norm(SYSTEM_PARAMS.contact.norm_eps)
            if offset_p < cur_min_offset:
                cur_min_offset = offset_p
                cur_min_idx = k
        return cur_min_idx
    
    @ti.func
    def find_sdf(
        self,
        point_position,
        point_velocity,
        triangle_index,
    ):
        a, b, c = self.triangles[triangle_index]
        p1 = self.particles_A[a]
        p2 = self.particles_A[b]
        p3 = self.particles_A[c]
        pc = 1/3*(p1+p2+p3)
        triangle_normal = ti.math.cross(
            p2 - p1, p3 - p1
        )
        triangle_normal = triangle_normal.normalized(SYSTEM_PARAMS.contact.norm_eps)
        x, y, z = pc
        yc, zc = self.centre_A_yz_ti[None]
        cylinder_outward_normal = ti.Vector([
            0,
            y-yc,
            z-zc,
        ])
        normal_direction = ti.math.sign(
            triangle_normal.dot(cylinder_outward_normal)
        )
        triangle_normal = normal_direction * triangle_normal
        point_to_vertex1 = point_position - p1
        signed_distance = point_to_vertex1.dot(triangle_normal)
        point_projected = point_position - signed_distance * triangle_normal
        surface_normal = -1 * triangle_normal
        edge1 = p3 - p1
        edge2 = p2 - p1
        point_projected_rel = point_projected - p1
        dot_edge1_edge1 = edge1.dot(edge1)
        dot_edge1_edge2 = edge1.dot(edge2)
        dot_edge1_point = edge1.dot(point_projected_rel)
        dot_edge2_edge2 = edge2.dot(edge2)
        dot_edge2_point = edge2.dot(point_projected_rel)
        inv_denominator = 1 / (
            dot_edge1_edge1 * dot_edge2_edge2 - dot_edge1_edge2 * dot_edge1_edge2
        )
        barycentric_u = (
            dot_edge2_edge2 * dot_edge1_point - dot_edge1_edge2 * dot_edge2_point
        ) * inv_denominator
        barycentric_v = (
            dot_edge1_edge1 * dot_edge2_point - dot_edge1_edge2 * dot_edge1_point
        ) * inv_denominator
        relative_velocity = point_velocity
        is_contact = (
            signed_distance < 0
            and barycentric_u >= 0
            and barycentric_v >= 0.0
            and (barycentric_u + barycentric_v <= 1)
        )
        return signed_distance, surface_normal, relative_velocity, is_contact
