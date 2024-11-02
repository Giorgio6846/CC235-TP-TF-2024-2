import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

colorFile = "./data/FaceImage.jpg"
depthFile = "./data/DepthImage.png"

print("Load Image")

colorArray = cv2.imread(colorFile)
depthArray = cv2.imread(depthFile)

depthArray = np.array(depthArray).astype('uint16')

colour = o3d.geometry.Image(np.array(colorArray))
depth = o3d.geometry.Image(np.array(depthArray))

print(depthArray.dtype, colorArray.dtype)

print("Depth Array", np.min(depthArray), np.max(depthArray))
print("Image Array", np.min(colorArray), np.max(colorArray))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(colour, depth)
print(rgbd_image)

plt.subplot(1, 2, 1)
plt.title("Camera grayscale image")
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title("Camera depth image")
plt.imshow(rgbd_image.depth)
plt.show()
