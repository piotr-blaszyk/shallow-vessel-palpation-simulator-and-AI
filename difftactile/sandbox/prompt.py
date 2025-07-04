@ti.func
def quat_to_mat(self, trans_v):
    # Quaternion components [w, x, y, z]
    qw = self.quat[0]
    qx = self.quat[1]
    qy = self.quat[2]
    qz = self.quat[3]
    
    # Convert quaternion to rotation matrix
    # First row
    m00 = 1.0 - 2.0 * (qy*qy + qz*qz)
    m01 = 2.0 * (qx*qy - qz*qw)
    m02 = 2.0 * (qx*qz + qy*qw)
    # Second row
    m10 = 2.0 * (qx*qy + qz*qw)
    m11 = 1.0 - 2.0 * (qx*qx + qz*qz)
    m12 = 2.0 * (qy*qz - qx*qw)
    # Third row
    m20 = 2.0 * (qx*qz - qy*qw)
    m21 = 2.0 * (qy*qz + qx*qw)
    m22 = 1.0 - 2.0 * (qx*qx + qy*qy)
    
    rot_mat = ti.Matrix([[m00, m01, m02],
                         [m10, m11, m12],
                         [m20, m21, m22]])

    # Build homogeneous transformation matrix
    trans_h = ti.Matrix.identity(float, 4)
    trans_h[0:3, 0:3] = rot_mat
    trans_h[0:3, 3] = trans_v
    
    return trans_h, rot_mat