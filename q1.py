import cv2
import numpy as np

image = cv2.imread('circle3.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold  image to get binary image
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Invert  binary image
binary_image_not = cv2.bitwise_not(binary_image)

#  morphological opening
kernel = np.ones((35, 35), np.uint8)
opened_image = cv2.morphologyEx(binary_image_not, cv2.MORPH_OPEN, kernel)

# Invert opened image
opened_image = cv2.bitwise_not(opened_image)
cv2.imshow('part b', opened_image)

# get centers of eachring
ring_centers = cv2.bitwise_xor(binary_image, opened_image)

# make rings smaller to get centroid
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
ring_centers = cv2.erode(ring_centers, erosion_kernel, iterations=1)

# get centroid in red
red_image = cv2.cvtColor(ring_centers, cv2.COLOR_GRAY2BGR)
for i in range(ring_centers.shape[0]):
    for j in range(ring_centers.shape[1]):
        if ring_centers[i, j] > 150:
            red_image[i, j] = [0, 0, 255]  # Red color in BGR format

# place red marks on original
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for i in range(red_image.shape[0]):
    for j in range(red_image.shape[1]):
        if (red_image[i, j] == [0, 0, 255]).all():
            # Set the pixel to red
            result_image[i, j] = [0, 0, 255]  # Red color in BGR format

# final result
cv2.imshow('part a', result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
