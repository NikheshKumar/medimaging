# The median filtering is used to smoothen each of the the pixels by replacing its value with the median of the neighbourhood. 
# The neighbourhood evaluted is a square neighbourhood of the current pixel.


import sys
import cv2 as cv
import numpy as np
import os
 
# Read image
img_path = 'image.png'
if not os.path.exists(img_path):
    raise FileNotFoundError("File could not be read. Check the path.")

img = cv.imread(img_path)
if img is None:
    raise ValueError("cv.imread failed to load image.")


# Convert BGR to RGB for matplotlib display
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

k = 5 # size of kernel window

# Apply median Blur

median = cv2.medianBlur(img, k)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(median), plt.title('Median Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
