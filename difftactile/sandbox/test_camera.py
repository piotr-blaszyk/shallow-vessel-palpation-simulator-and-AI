import cv2
import time

def test_camera(camera_id):
    print(f"\nTesting camera {camera_id}")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera {camera_id} properties:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read frame from camera {camera_id}")
    else:
        print(f"Failed to read frame from camera {camera_id}")
    
    cap.release()

def main():
    # Test both cameras
    test_camera(0)
    test_camera(1)

if __name__ == "__main__":
    main() 