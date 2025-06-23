"""
a class to model fisheye camera
"""

import numpy as np
import cv2
from os import path as osp
import os
import math
import taichi as ti
import glob
import pickle

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

if __name__ == '__main__':
    # Get all image files from the directory
    image_dir = "/home/psb120/Documents/TCP-IP-Python-V4/experiment-capture-completed"
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
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