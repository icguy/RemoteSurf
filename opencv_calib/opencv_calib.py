import numpy as np
import cv2
import glob


np.set_printoptions(precision=5, suppress=True)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

outfile = file("out.txt", "w")

imgset1 = {
    "grid size" : (29, 19),
    "resolution" : (1600, 1200),
    "real size" : 0.95, # grid distance in cm
    "img path" : 'imgset2/*.jpg'}
imgset2 = {
    "grid size" : (9, 6),
    "resolution" : (1600, 1200),
    "real size" : 2.6222, # grid distance in cm
    "img path" : 'imgset1/*.jpg'}
imgset3 = {
    "grid size" : (9, 6),
    "resolution" : (960, 720),
    "real size" : 2.615, # grid distance in cm
    "img path" : 'D:/dokumentumok/Python/RemoteSurf/out/2016_11_18__11_51_59/*.jpg'}
imgset = imgset3

grid_size = imgset["grid size"]
resolution = imgset["resolution"]
real_size = imgset["real size"]
imgs_path = imgset["img path"]

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2)
objp *= real_size
print objp


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(imgs_path)
print images

fnames = []
# images = ["imgset2\\Picture 15.jpg"]
num = -1
for fname in images:
    num += 1
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, grid_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    outfile.write("%d \r\n %s \r\r %s \r\n" % (num , "---- corners" , str(corners)))
    print ("+ " if ret else "- ") + fname

    if ret:
        fnames.append(fname)
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        outfile.write("%d \r\n %s \r\r %s \r\n" % (num , "---- corners subpix" , str(corners)))
        imgpoints.append(corners)

        img2 = cv2.pyrDown(img)
        cv2.drawChessboardCorners(img2, grid_size, corners / 2, ret)
        cv2.imshow('img',img2)
        cv2.waitKey(1)
        cv2.destroyWindow("img")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, resolution, flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT)

outfile.write("mat, distcoeff \r\n %s \r\n .. \r\n %s" % (str(mtx), str(dist)))
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

outfile.close()