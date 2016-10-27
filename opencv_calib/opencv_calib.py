import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

grid_size = (9, 6)
resolution = (1600, 1200)
real_size = 2.6222 # grid distance in cm

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2)
objp *= real_size


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('imgset1/*.jpg')

fnames = []
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.pyrDown(img)
    gray = cv2.pyrDown(gray)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)


    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
    if corners is not None:
        corners *= 4


    # If found, add object points, image points (after refining them)
    if ret:
        print fname
        fnames.append(fname)
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        img2 = cv2.pyrDown(img)
        cv2.drawChessboardCorners(img2, grid_size, corners / 2, ret)
        cv2.imshow('img',img2)
        cv2.waitKey(1)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, resolution, flags=cv2.CALIB_FIX_ASPECT_RATIO)
print ret
print mtx
print dist
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    img = cv2.imread(fnames[i])
    img2 = cv2.pyrDown(img)
    cv2.drawChessboardCorners(img2, grid_size, imgpoints2 / 2, True)
    cv2.imshow("img", img2)
    cv2.waitKey(0)

    mean_error += error

print "total error: ", mean_error/len(objpoints)
print len(objpoints)