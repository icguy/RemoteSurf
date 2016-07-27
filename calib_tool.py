import cv2
import numpy as np

img = cv2.imread("imgs/m/WP_20160714_001.jpg")

r, c, d = img.shape

pts = []
for i in range(r):
    for j in range(c):
        if np.sum(np.abs(img[i][j] - np.array([255, 0, 0]))) == 0:
            pts.append((i,j))

print(pts)
