import cv2
import numpy as np

camMat1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])


camMat2 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

extMat1 = np.array([[1, 0, 0, 20],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])

extMat2 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])

projMat1 = np.dot(camMat1, extMat1)
projMat2 = np.dot(camMat2, extMat2)

projMat1 = np.random.rand(3, 4)
projMat2 = np.random.rand(3, 4)

numPts = 20
points = np.random.rand(4, numPts) * 10
points[3] = np.array([1] * numPts)

projPts1 = np.dot(projMat1, points)
projPts2 = np.dot(projMat2, points)

for i in range(0, numPts):
    projPts1[0][i] /= projPts1[2][i]
    projPts1[1][i] /= projPts1[2][i]
    projPts1[2][i] /= projPts1[2][i]

    projPts2[0][i] /= projPts2[2][i]
    projPts2[1][i] /= projPts2[2][i]
    projPts2[2][i] /= projPts2[2][i]

p4d = cv2.triangulatePoints(projMat1, projMat2, projPts1[0:2], projPts2[0:2])

for i in range(0, numPts):
    p4d[0][i] /= p4d[3][i]
    p4d[1][i] /= p4d[3][i]
    p4d[2][i] /= p4d[3][i]
    p4d[3][i] /= p4d[3][i]

print(p4d - points)