import cv2
import numpy as np

image = cv2.imread('circles.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold  image to get binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Invert binary image
binary_image_not = cv2.bitwise_not(binary_image)

# morphological opening
kernel = np.ones((35, 35), np.uint8)
opened_image = cv2.morphologyEx(binary_image_not, cv2.MORPH_OPEN, kernel)

# Invert the opened image
opened_image = cv2.bitwise_not(opened_image)

# find center of rings ie the holes
ring_centers = cv2.bitwise_xor(binary_image, opened_image)

# find num of holes
contours, _ = cv2.findContours(ring_centers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("objects with holes:", len(contours))

#total num of circles - those with holes
contours2, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("objects without holes:", len(contours2)-len(contours))
