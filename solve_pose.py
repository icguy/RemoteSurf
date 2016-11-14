import cv2
import numpy as np
import pickle

class kp_dummy:
    def __init__(self, _pt):
        self.pt = _pt

f = open("keypoints.p")
kp_vec = pickle.load(f)
vec, des, pt = zip(*kp_vec["coords"])
des = np.array([d for d in des])

camMtx = np.array([[2241.45, 0., 1295.5],
                   [0., 2241.45, 727.5,],
                   [0., 0., 1.]])

detector = cv2.SURF()
img2 = cv2.imread("imgs/WP_20160714_005.jpg", 0)

kp2, des2 = detector.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

img_pts = []
obj_pts = []

for m in good:
    obj_pts.append(vec[m.queryIdx])
    img_pts.append(kp2[m.trainIdx].pt)


from feature_test import drawMatches
from calib_test import test
import rpy


img1 = cv2.imread("imgs/WP_20160713_002.jpg", 0)
kp1 = [kp_dummy(p) for p in pt]
# drawMatches(img1, kp1, img2, kp2, good)

flag = cv2.CV_ITERATIVE
obj_pts = np.array(obj_pts, dtype=np.float32)
obj_pts = np.array(obj_pts[:,:3])
img_pts = np.array(img_pts, dtype=np.float32)
rvec, tvec, inl = cv2.solvePnPRansac(obj_pts, img_pts, camMtx, None, flags=flag)
print(rvec, tvec, len(obj_pts)-len(inl))
rmat, jac = cv2.Rodrigues(rvec)
print rpy.rpy(rmat)
print np.dot(rmat, tvec)
tmat = np.zeros((3,4))
tmat[:3,:3] = rmat
tmat[:3,3] = tvec.T
print(tmat)
print "..................."
print np.dot(camMtx, tmat)

temp = []
for i in range(img_pts.shape[0]):
    if i in inl:
        temp.append(img_pts[i])
img_pts = np.array(temp)

temp = []
for i in range(obj_pts.shape[0]):
    if i in inl:
        temp.append(obj_pts[i])
obj_pts = np.array(temp)

test(img_pts, obj_pts, tmat, camMtx)

# WP_20160714_005
imgPtMarker = np.array([[776, 282],
                       [750, 1033],
                       [1519, 293],
                       [1508, 1047]], dtype=np.float32)
size = 14.1
sh = size / 2
objPtMarker = np.array([[-sh, -sh, 0],
                     [-sh, sh, 0],
                     [sh, -sh, 0],
                     [sh, sh, 0]], dtype=np.float32)

test(imgPtMarker, objPtMarker, tmat, camMtx)