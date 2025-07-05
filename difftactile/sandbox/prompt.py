from scipy.spatial.transform import Rotation as R
import numpy as np

xs = np.array([
    [np.pi/2, 0, 0],
    [np.pi/4, np.pi/3, 0],
])

for x in xs:
    # Create a rotation object from an axis-angle
    r = R.from_rotvec(x) # Rotate 90 degrees (pi/2 rad) around Y-axis

    # Get the equivalent quaternion
    q_scipy = r.as_quat() # (x, y, z, w) order for scipy
    print(f"Quaternion components (scipy): {q_scipy}")

    # The amount of rotation can be directly obtained from the Rotation object
    rotation_angle_radians = r.magnitude()
    print(f"Amount of rotation (radians) from Rotation object: {rotation_angle_radians}")
    print(f"Amount of rotation (degrees) from Rotation object: {np.degrees(rotation_angle_radians)}")

    # Or, if you have just the quaternion (scipy.spatial.transform.Rotation stores in (x,y,z,w) order)
    w_val = q_scipy[3] # w is the last element
    amount_from_w = 2 * np.arccos(w_val)
    print(f"Amount of rotation (radians) from w component: {amount_from_w}")
    print()