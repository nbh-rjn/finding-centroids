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
ring_centers = cv2.bitwise_xor(binary_image, opened_image)

# we erode the holes until all except the biggest are gone
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47))
biggest_hole = cv2.erode(ring_centers, erosion_kernel, iterations=1)

# then we dilate using  a kernel of the same size
# to restore the remaining hole to its original size
biggest_hole = cv2.dilate(biggest_hole, erosion_kernel, iterations=1)

image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

contours, _ = cv2.findContours(biggest_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter = int(2 * radius)

    # convert everything to int to avoid errors when using these values for indexing
    radius = int(radius)
    center = (int(x), int(y))
    p1 = (int(x) - radius, int(y))
    p2 = (int(x) + radius, int(y))

    #place lines and text
    cv2.line(image_display, p1, p2, (0, 255, 0), 2)
    cv2.putText(image_display, f"Diameter: {diameter}",
                (center[0] - radius, center[1] - 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)


cv2.imshow('part c', image_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
