import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

grid_size = (29, 19)
resolution = (1600, 1200)
real_size = 0.95 # grid distance in cm
imgs_path = 'imgset2/*.jpg'
downscale = 2

# grid_size = (9, 6)
# resolution = (1600, 1200)
# real_size = 2.6222 # grid distance in cm
# imgs_path = 'imgset1/*.jpg'
# downscale = 2

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2)
objp *= real_size


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(imgs_path)

corner_dict = {
    "imgset2\\Picture 11.jpg" : np.array([[326, 267], [1260, 272], [348, 908], [1377, 845]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 12.jpg" : np.array([[396, 359], [1144, 374], [381, 847], [1157, 853]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 13.jpg" : np.array([[458, 316], [1328, 306], [425, 849], [1409, 832]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 14.jpg" : np.array([[311, 636], [1034, 196], [646, 1084], [1438, 515]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 15.jpg" : np.array([[179, 633], [1036, 245], [506, 1105], [1498, 527]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 16.jpg" : np.array([[214, 266], [1394, 265], [169, 1038], [1442, 1040]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 17.jpg" : np.array([[320, 382], [1267, 386], [211, 896], [1385, 896]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 18.jpg" : np.array([[253, 605], [1220, 517], [135, 1095], [1409, 974]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 19.jpg" : np.array([[523, 417], [1323, 142], [432, 991], [1294, 957]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 20.jpg" : np.array([[590, 231], [1462, 297], [227, 715], [1045, 1039]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 21.jpg" : np.array([[520, 251], [1268, 515], [194, 665], [938, 1068]], dtype='float64').reshape((4, 2)),
    "imgset2\\Picture 22.jpg" : np.array([[537, 340], [1421, 438], [377, 851], [1379, 1012]], dtype='float64').reshape((4, 2)),
}

fnames = []
# images = ["imgset2\\Picture 15.jpg"]
for fname in images:
    img = cv2.imread(fname)

    if fname not in corner_dict:
        scale = 1
        gray = img
        for i in range(downscale - 1):
            gray = cv2.pyrDown(gray)
            scale *= 2
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, grid_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if corners is not None:
            corners *= scale

        print ("+ " if ret else "- ") + fname
    else:
        corners = corner_dict[fname]
        mtx = np.array([1.8435610863515064e+003, 0., 799.5, 0., 1.8435610863515064e+003, 599.5, 0., 0., 1.]).reshape((3, 3))
        obj_pts = np.array([[.0, .0, 0], [28, .0, 0], [0, 18, 0], [28, 18, 0]]) * .95
        retval, rvec, tvec = cv2.solvePnP(obj_pts, corners, mtx, None)
        corners, _ = cv2.projectPoints(objp, rvec, tvec, mtx, None)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img2 = img
        cv2.drawChessboardCorners(img2, grid_size, corners, True)
        corners = corners
        img2 = cv2.pyrDown(img2)
        cv2.imshow("", img2)
        cv2.waitKey(0)

        ret = True
        print ("+ " if ret else "- ") + fname

    if ret:
        fnames.append(fname)
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        img2 = cv2.pyrDown(img)
        cv2.drawChessboardCorners(img2, grid_size, corners / 2, ret)
        cv2.imshow('img',img2)
        cv2.waitKey(1)
        cv2.destroyWindow("img")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, resolution, flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_ZERO_TANGENT_DIST)
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
    cv2.waitKey(1)

    mean_error += error

print "total error: ", mean_error/len(objpoints)
print len(objpoints)