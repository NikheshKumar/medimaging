pip install numpy pandas opencv-python matplotlib

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
