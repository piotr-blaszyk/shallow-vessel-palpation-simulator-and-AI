import glob

import cv2
import numpy as np

from difftactile.main.constants import *

# Checkerboard dimensions (inner corners)
CHECKERBOARD = (4-1, 5-1)  # Adjust to your board

# Prepare 3D object points (0,0,0), (1,0,0), ... (7,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Store object/image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load calibration images
images = glob.glob('/home/psb120/camera-calibration-2/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, 
        CHECKERBOARD, 
        None
    )
    
    if ret:
        objpoints.append(objp.copy())
        # Refine corner locations (pixel accuracy)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        )
        imgpoints.append(corners_refined)
        print(f"Found {len(corners_refined)} corners in {fname}")
        assert len(corners_refined) == CHECKERBOARD[0] * CHECKERBOARD[1]

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
    else:
       print(f"Checkerboard not found in {fname}")

cv2.destroyAllWindows()

# Fisheye calibration (requires ≥10 images)
assert len(objpoints) >= 10, "Insufficient images for calibration"

K = np.zeros((3, 3))  # Intrinsic matrix
D = np.zeros((4, 1))  # Distortion coefficients (k1, k2, k3, k4)
rvecs, tvecs = [], []  # Rotation/translation vectors

# Calibration flags
# flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND
flags = None

# Calibrate
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],  # Image resolution (width, height)
    K,
    D,
    rvecs,
    tvecs,
    flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

# Compute RMS reprojection error
mean_error = 0
num_points = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.fisheye.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], K, D
    )
    imgpoints2 = imgpoints2.reshape(-1, 2)
    imgpoints_i = imgpoints[i].reshape(-1, 2)
    error = cv2.norm(imgpoints_i, imgpoints2, cv2.NORM_L2)
    mean_error += error**2
    num_points += len(imgpoints[i])
if num_points > 0:
    rms = np.sqrt(mean_error / num_points)
    print(f"RMS Reprojection Error: {rms:.4f} pixels")
else:
    print("No points for RMS error calculation.")

# Results
print(f"Focal Length (fx, fy): {K[0,0]:.2f}, {K[1,1]:.2f}")
print(f"Principal Point (cx, cy): {K[0,2]:.2f}, {K[1,2]:.2f}")
print(f"Distortion Coefficients (k1, k2, k3, k4): {D.flatten()}")

# Save parameters
np.savez(SYSTEM_PARAMS.files.fisheye_params, K=K, D=D)

# Visualize undistortion for each calibration image
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out_path = f"{SYSTEM_PARAMS.files.output_dir}undistorted_{idx+1:02d}.png"
    cv2.imwrite(out_path, undistorted_img)
    print(f"Saved undistorted image: {out_path}")
