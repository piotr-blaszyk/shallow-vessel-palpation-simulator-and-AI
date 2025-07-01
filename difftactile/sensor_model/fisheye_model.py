"""
a class to model fisheye camera
"""

import numpy as np
import cv2
from os import path as osp
import os
import math
import taichi as ti
from glob import glob
import pickle
import matplotlib.pyplot as plt

F = 230.0
FX = F
FY = F
CX = 359.0
CY = 266.0

@ti.func
def project_3d_2d(a, fx=FX, fy=FY, cx=CX, cy=CY):
    #ref. Universal Semantic Segmentation for Fisheye Urban Driving Images Ye et al.
    #a is 3d vec
    a[2] += 2.0*0.01 # distance to the image plane
    a_norm = a.norm(1e-10)
    cos = a[2] / a_norm

    cos = ti.min(1.0, cos)
    cos = ti.max(-1.0, cos)
    theta = ti.acos(cos)
    omega = ti.atan2(a[1],a[0]+1e-8) + ti.math.pi
    r_x = fx * theta
    r_y = fy * theta

    p = ti.Vector([0.0, 0.0])
    p[0] = r_x * ti.cos(omega) + cx
    p[1] = r_y * ti.sin(omega) + cy

    return p

def project_points_to_pix(a, fx=FX, fy=FY, cx=CX, cy=CY):
    #ref. Universal Semantic Segmentation for Fisheye Urban Driving Images Ye et al.
    #a is a point cloud if (n, 3)
    a[:,2] += 2.0*0.01 #(14-0.7-9)* 0.01 # distance to the image plane
    b = np.array([[0., 0., 1.]]).repeat(len(a), axis=0)
    inner_product = (a * b).sum(axis=1)
    a_norm = np.linalg.norm(a,axis=1)
    b_norm = np.linalg.norm(b,axis=1)
    cos = inner_product / (a_norm * b_norm)

    theta = np.arccos(cos)
    omega = np.arctan2(a[:,1],a[:,0]) + np.pi

    r_x = fx * theta
    r_y = fy * theta

    p = np.zeros((len(a),2))
    p[:,0] = r_x * np.cos(omega) + cx
    p[:,1] = r_y * np.sin(omega) + cy

    return p

def project_pix_to_points(p, hemisphere_radius, fx=FX, fy=FY, cx=CX, cy=CY):
    """
    Projects 2D pixel coordinates to 3D points on a hemisphere.
    Camera configuration:
        - position: (0,0,0)
        - looking at: (0,0,1) (z+ direction)
        - up vector: (0,1,0) (y axis)
    Args:
        p: numpy array of shape (n, 2) containing pixel coordinates
        hemisphere_radius: radius of the hemisphere the points will lie on
        fx, fy: focal lengths in x and y directions
        cx, cy: principal point coordinates
    Returns:
        numpy array of shape (n, 3) containing 3D points
    Raises:
        ValueError: If any projected point falls outside the valid circle on x-y plane
    """
    # Convert to normalized coordinates
    x_norm = (p[:, 0] - cx) / fx  # normalize by focal length
    y_norm = (p[:, 1] - cy) / fy  # normalize by focal length
    
    # Calculate radial distance in normalized coordinates
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    # Calculate theta (angle from z-axis)
    theta = r  # in the normalized coordinate system, r directly gives us theta
    
    # Calculate phi (azimuthal angle in x-y plane)
    phi = np.arctan2(y_norm, x_norm)
    
    # Convert to 3D coordinates on hemisphere
    points = np.zeros((len(p), 3))
    points[:, 0] = hemisphere_radius * np.sin(theta) * np.cos(phi)  # x
    points[:, 1] = hemisphere_radius * np.sin(theta) * np.sin(phi)  # y
    points[:, 2] = hemisphere_radius * np.cos(theta)  # z (forward direction)
    
    # Check if projections onto x-y plane are within the circle
    xy_projections = np.sqrt(points[:, 0]**2 + points[:, 1]**2)  # radial distances in x-y plane
    if np.any(xy_projections > hemisphere_radius):
        raise ValueError("Marker projected outside the inner surface of the spherical cap")
    
    return points

def get_marker_image(img):
    # Get the actual image dimensions
    if len(img.shape) == 3:
        source_height, source_width = img.shape[:2]
    else:
        source_height, source_width = img.shape
    
    # Scale factors for adapting from original 640x480 to current image size
    scale_x = source_width / 640.0
    scale_y = source_height / 480.0
    
    params = cv2.SimpleBlobDetector_Params()

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    # Circle parameters - scaled to match the input image resolution
    circle_center = np.array([345 * scale_x, 260 * scale_y])
    circle_radius = 170 * min(scale_x, scale_y)  # Use minimum scale to maintain circular shape

    MarkerCenter = []
    for pt in keypoints:
        point = np.array([pt.pt[0], pt.pt[1]])
        # Calculate distance from point to circle center
        distance = np.linalg.norm(point - circle_center)
        # Only add points that are inside the circle
        if distance < circle_radius:
            MarkerCenter.append([pt.pt[0], pt.pt[1]])
    MarkerCenter = np.array(MarkerCenter)

    return MarkerCenter, circle_center, circle_radius

def project_points_to_pix_cv2(points3d, K=None, D=None, rvec=None, tvec=None):
    """
    Project 3D points to 2D image coordinates using cv2.fisheye.projectPoints.
    Args:
        points3d: (num_points, 3) or (3,) array-like, the 3D points in camera coordinates.
        K: (3,3) camera intrinsic matrix. If None, uses default fx, fy, cx, cy.
        D: (4,) distortion coefficients. If None, uses default values.
        rvec: (3,1) rotation vector. If None, assumes zero rotation.
        tvec: (3,1) translation vector. If None, assumes zero translation.
    Returns:
        points2d: (num_points, 2) numpy array, the projected 2D points on the image plane.
    """
    points3d = np.asarray(points3d, dtype=np.float64)
    if points3d.ndim == 1:
        points3d = points3d.reshape(1, 3)
    points3d = points3d.reshape(-1, 1, 3)
    if K is None:
        K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    if D is None:
        D = np.zeros((4, 1), dtype=np.float64)
    if rvec is None:
        rvec = np.zeros((3, 1), dtype=np.float64)
    if tvec is None:
        tvec = np.zeros((3, 1), dtype=np.float64)
    imgpts, _ = cv2.fisheye.projectPoints(points3d, rvec, tvec, K, D)
    res = imgpts.reshape(-1, 2)
    return res

def interactive_exploration(image_dir):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
        image_files.extend(glob(os.path.join(image_dir, ext.upper())))
    
    # Sort files for consistent ordering
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        exit()
    
    print(f"Found {len(image_files)} images")
    print("Controls: 'j' - previous image, 'l' - next image, 'q' - quit")
    
    current_index = 0
    
    while True:
        # Load current image
        img_path = image_files[current_index]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            current_index = (current_index + 1) % len(image_files)
            continue
        
        # Get marker positions and circle parameters
        marker_positions, circle_center, circle_radius = get_marker_image(img)
        
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Draw the circle outline used for marker detection (green color)
        circle_center_int = (int(circle_center[0]), int(circle_center[1]))
        circle_radius_int = int(circle_radius)
        cv2.circle(vis_img, circle_center_int, circle_radius_int, color=(0, 255, 0), thickness=2)
        
        # Draw detected markers
        for pos in marker_positions:
            # Convert positions to integers for drawing
            center = (int(pos[0]), int(pos[1]))
            # Draw a circle at each marker position (red color)
            cv2.circle(vis_img, center, radius=5, color=(0, 0, 255), thickness=2)
        
        # Add text overlay with image info
        filename = os.path.basename(img_path)
        info_text = f"Image {current_index + 1}/{len(image_files)}: {filename}"
        cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, f"Markers detected: {len(marker_positions)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the image with detected markers
        cv2.imshow("Interactive Marker Detection", vis_img)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('j'):  # Previous image
            current_index = (current_index - 1) % len(image_files)
        elif key == ord('l'):  # Next image
            current_index = (current_index + 1) % len(image_files)
    
    cv2.destroyAllWindows()

def extract_experimental_markers_and_save_to_file(image_dir):
    # Get all image files from the directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
        image_files.extend(glob(os.path.join(image_dir, ext.upper())))
    
    # Sort files for consistent ordering
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        exit()
    
    print(f"Found {len(image_files)} images")
    
    # Initialize lists to store marker positions and labels
    all_marker_positions = []
    class_labels = []
    
    # Process each image
    for img_path in image_files:
        # Load image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Get marker positions
        marker_positions, _, _ = get_marker_image(img)
        
        # Get file number from filename (format: "press-{file_num}.jpg")
        filename = os.path.basename(img_path)
        try:
            # Extract number between "press-" and ".jpg"
            file_num = int(filename.split("press-")[1].split(".")[0])
        except (IndexError, ValueError):
            print(f"Warning: Filename {filename} doesn't match expected format 'press-{{number}}.jpg'")
            continue
        
        # Determine class label based on file number
        if 0 <= file_num <= 10:
            label = 0
        elif 11 <= file_num <= 20:
            label = 1
        elif 21 <= file_num <= 30:
            label = 2
        elif 31 <= file_num <= 40:
            label = 3
        elif file_num == 41:
            label = 4
        elif file_num == 42:
            label = 5
        elif file_num == 43:
            label = 6
        else:
            print(f"Warning: File number {file_num} doesn't match any label criteria")
            continue
        
        # Store results
        all_marker_positions.append(marker_positions)
        class_labels.append(label)
        
        print(f"Processed image {filename} - Found {len(marker_positions)} markers")
    
    # Convert to numpy arrays
    all_marker_positions = np.array(all_marker_positions)
    class_labels = np.array(class_labels)
    
    # Create dictionary with results
    results = {
        'marker_positions': all_marker_positions,
        'class_labels': class_labels
    }
    
    # Save to pickle file
    output_file = "vascular-tumour-press-experimental-results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_file}")
    print(f"Marker positions shape: {all_marker_positions.shape}")
    print(f"Class labels shape: {class_labels.shape}")

def save_init_marker_positions():
    # Read the image
    img = cv2.imread('../tasks/vitactip_photo_default_state.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Could not find or open vitactip_photo_default_state.png")
    
    # Get marker positions
    marker_positions, circle_center, circle_radius = get_marker_image(img)
    
    # Save marker positions to pickle file
    with open('init-marker-positions.pkl', 'wb') as f:
        pickle.dump(marker_positions, f)
    
    print(f"Found {len(marker_positions)} markers")
    print(f"Marker positions saved to init-marker-positions.pkl")
    
    return marker_positions

def generate_marker_3d_projection():
    """
    Loads 2D marker positions from init-marker-positions.pkl,
    projects them to 3D points on a hemisphere using project_pix_to_points,
    and saves the 3D positions to init-marker-positions-3d.pkl
    """
    # Load 2D marker positions
    with open('init-marker-positions.pkl', 'rb') as f:
        marker_positions_2d = pickle.load(f)

    if False:
        with open('../tasks/output/gmsh-mesh.pkl', 'rb') as f:
            gmsh_data = pickle.load(f)
        shell_inner_r = gmsh_data['radius_of_curvature_inner']
        shell_outer_r = gmsh_data['radius_of_curvature_outer']
    else:
        shell_inner_r = 35 + 1/3
        shell_outer_r = 36 + 1/3
    
    # Project to 3D using hemisphere radius of 2.9
    A_points = project_pix_to_points(marker_positions_2d, hemisphere_radius=shell_inner_r-1)
    B_points = project_pix_to_points(marker_positions_2d, hemisphere_radius=shell_outer_r+2)

    obj = {
        'A_points': A_points,
        'B_points': B_points,
    }
    
    # Save 3D marker positions to pickle file
    with open('biomimetic-tip-points.pkl', 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    # image_dir = "../experiment-capture-completed"
    # interactive_exploration(image_dir)
    # save_init_marker_positions()
    # extract_experimental_markers_and_save_to_file(image_dir)
    
    generate_marker_3d_projection()