# Adaptive Histogram equalisation computes several histograms, each corresponding to a distinct section of the image, and uses them to redistribute the pixel intensity values.
# By limiting the amplification, we get CLAHE, by performing histogram equalization on localized regions of the image, known as tiles. This method divides the image into smaller sections and equalizes the histogram of each tile independently. 
# This method prevents over amplification of noise in honogenous regions.
# This process can significantly enhance the overall contrast of images, especially those that are poorly lit or have low contras.

!pip install numpy pandas opencv-python matplotlib

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Loading the pixel values from the CSV file

df = pd.read_csv('image.csv', header=None)
image_array = df.values.astype(np.uint8)

image_height = 100  # Example height
image_width = 100   # Example width
image_reshaped = image_array.reshape((image_height, image_width))

# Applying CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE to the grayscale image
clahe_image = clahe.apply(image_reshaped)

# Displaying the original and CLAHE enhanced images
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_reshaped, cmap='gray')
plt.axis('off')

# CLAHE Enhanced Image
plt.subplot(1, 2, 2)
plt.title('CLAHE Enhanced Image')
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.show()
