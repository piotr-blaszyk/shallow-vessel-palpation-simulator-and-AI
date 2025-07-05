from pyquaternion import Quaternion
import numpy as np

# Original quaternion: 45 degrees around Z-axis
# In pyquaternion, order is (w, x, y, z)
q_original = Quaternion(axis=[0, 0, 1], degrees=45)
print(f"Original quaternion: {q_original.elements}")
print(f"Original angle (degrees): {q_original.degrees}")

# Desired new angle
desired_angle_degrees = 90
desired_angle_radians = np.radians(desired_angle_degrees)

# --- Step 1: Extract current axis and angle ---
# pyquaternion's .axis and .angle properties do this for us for unit quaternions
current_axis = q_original.axis
current_angle_radians = q_original.angle
print(f"Extracted axis: {current_axis}")
print(f"Extracted angle (radians): {current_angle_radians}")

# Handle the edge case of zero rotation (where axis is undefined)
# If the quaternion represents effectively no rotation (angle close to 0 or 360),
# the axis can be arbitrary. We'll pick a default.
if np.isclose(current_angle_radians % (2 * np.pi), 0) or np.isclose(current_angle_radians % (2 * np.pi), 2 * np.pi):
    print("Warning: Original quaternion represents effectively zero rotation. Using a default axis.")
    current_axis = np.array([1.0, 0.0, 0.0]) # Or any other default axis
    # If the desired angle is also 0, the quaternion will be (1,0,0,0) which is correct.
    # If desired_angle_radians is not 0, it will rotate around the default axis.

# --- Step 2: Desired angle is already defined ---

# --- Step 3: Construct new quaternion ---
# Using pyquaternion's constructor directly for convenience
q_scaled = Quaternion(axis=current_axis, radians=desired_angle_radians)

print(f"\nScaled quaternion: {q_scaled.elements}")
print(f"Scaled angle (degrees): {q_scaled.degrees}")
print(f"Scaled axis: {q_scaled.axis}")

# Example 2: Scaling a non-trivial rotation
q_original_complex = Quaternion(axis=[1, 1, 1], degrees=60)
print(f"\nOriginal complex quaternion: {q_original_complex.elements}")
print(f"Original complex angle (degrees): {q_original_complex.degrees}")
print(f"Original complex axis: {q_original_complex.axis}")

desired_angle_complex_degrees = 120
desired_angle_complex_radians = np.radians(desired_angle_complex_degrees)

current_axis_complex = q_original_complex.axis

q_scaled_complex = Quaternion(axis=current_axis_complex, radians=desired_angle_complex_radians)
print(f"\nScaled complex quaternion: {q_scaled_complex.elements}")
print(f"Scaled complex angle (degrees): {q_scaled_complex.degrees}")
print(f"Scaled complex axis: {q_scaled_complex.axis}")