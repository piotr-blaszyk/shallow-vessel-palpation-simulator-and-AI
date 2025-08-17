import taichi as ti

from difftactile.main.constants import *

class FisheyeModelTaichi:
    def __init__(self):
        pass

    @ti.func
    def project_3d_2d(self, a):
        a_norm = a.norm(1e-12)
        cos = a[2] / a_norm
        cos = ti.min(1.0, cos)
        cos = ti.max(-1.0, cos)
        theta = ti.acos(cos)
        x_normalized = ti.select(
            abs(a[0]) < 1e-10, ti.select(a[0] >= 0, 1e-10, -1e-10), a[0]
        )
        omega = ti.atan2(a[1], x_normalized) + ti.math.pi
        r_x = SYSTEM_PARAMS.fisheye_model.focal_length_x * theta
        r_y = SYSTEM_PARAMS.fisheye_model.focal_length_y * theta
        p = ti.Vector([0.0, 0.0])
        p[0] = -r_x * ti.cos(omega) + SYSTEM_PARAMS.fisheye_model.principal_point_x
        p[1] = -r_y * ti.sin(omega) + SYSTEM_PARAMS.fisheye_model.principal_point_y
        return p
    