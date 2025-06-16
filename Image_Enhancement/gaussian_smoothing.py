import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os

# read image
img_path = 'image.png'
if not os.path.exists(img_path):

img = cv.imread(img_path)

if img is None:
    raise ValueError("cv.imread failed to load image.")

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# apply gaussian Blur

ksize = (5, 5)  # must be odd and positive integers

sigmaX, sigmaY = 1, 1 # the standard deviation in the X and Y directions

blur = cv.GaussianBlur(img_rgb, ksize, sigmaX, sigmaY) #	cv.GaussianBlur(	src, ksize, sigmaX[, dst[, sigmaY[, borderType[, hint]]]]	) ->	dst

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
