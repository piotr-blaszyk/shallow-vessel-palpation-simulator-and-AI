import numpy as np
import cv2

image_path = "difftactile/output/training_data/segmentation_mask/segmentation_mask_0_42.png"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

image_size = image.shape

# blur_kernel_size = (41, 41)
# processed_image = cv2.GaussianBlur(image, blur_kernel_size, 0)
# _, processed_image = cv2.threshold(processed_image, 10, 255, cv2.THRESH_BINARY)

cv2.imshow("Original Image", image)
# cv2.imshow("Processed Image (after blur/threshold)", processed_image)

contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_shape_image = np.zeros(image_size, dtype=np.uint8)
cv2.drawContours(filled_shape_image, contours, -1, 255, cv2.FILLED)

cv2.imshow("Filled Contours", filled_shape_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
