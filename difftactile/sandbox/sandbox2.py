# Import necessary libraries
import cv2
import numpy as np

# Read an image
img = cv2.imread("difftactile/sandbox/image.png")

# Define an array of endpoints of triangle
points = np.array([[150, 450], [450, 150], [150, 150]])

# Use fillPoly() function and give input as
# image, end points,color of polygon
# Here color of polygon will blue
cv2.fillPoly(img, pts=[points], color=(255, 0, 0))

# points -= np.array([100, 100])

# cv2.fillPoly(img, pts=[points], color=(0, 255, 0))

# Displaying the image
cv2.imshow("Triangle", img)

# wait for the user to press any key to 
# exit window
cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()