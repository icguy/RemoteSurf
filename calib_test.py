import cv2
import numpy as np
import rpy

def test(imgpts, objpts, tmat, cmat):
    numpts =  objpts.shape[0]
    proj = np.dot(cmat, tmat)

    c3d = np.zeros((4, numpts))
    c3d[:3,:] = objpts.T
    c3d[3,:] = np.ones((1, numpts))

    reproj = np.dot(proj, c3d)
    for i in range(numpts):
        w = reproj[2, i]
        for j in range(3):
            reproj[j, i] /= w

    errs = np.abs(imgpts.T - reproj[:2,:])
    max_err = np.max(errs)
    avg_err = np.average(errs)
    print("max_err: ", max_err)
    print("avg_err: ", avg_err)

if __name__ == "__main__":
    size = 14.1
    sh = size / 2

    # WP_20160713_002
    imgPt1 = np.array([[297, 706],
                     [194, 1530],
                     [1147, 733],
                     [1173, 1565]], dtype=np.float32)

    # WP_20160713_004
    imgPt2 = np.array([[374, 867],
                       [307, 1616],
                       [1075, 744],
                       [1047, 1641]], dtype=np.float32)

    # WP_20160714_001
    imgPt3 = np.array([[732, 1207],
                       [729, 423],
                       [1387, 1307],
                       [1377, 320]], dtype=np.float32)

    # WP_20160714_005
    imgPt4 = np.array([[776, 282],
                       [750, 1033],
                       [1519, 293],
                       [1508, 1047]], dtype=np.float32)



    objPt = np.array([[-sh, -sh, 0],
                     [-sh, sh, 0],
                     [sh, -sh, 0],
                     [sh, sh, 0]], dtype=np.float32)

    camMtx = np.array([[2241.45, 0., 1295.5],
                       [0., 2241.45, 727.5,],
                       [0., 0., 1.]])

    imgPt = imgPt4
    flag = cv2.CV_ITERATIVE

    retval, rvec, tvec = cv2.solvePnP(objPt, imgPt, camMtx, None, flags=flag)
    print(retval, rvec, tvec)
    rmat, jac = cv2.Rodrigues(rvec)
    print rpy.rpy(rmat)
    print np.dot(rmat, tvec)
    tmat = np.zeros((3,4))
    tmat[:3,:3] = rmat
    tmat[:3,3] = tvec.T
    print(tmat)
    print "..................."
    print np.dot(camMtx, tmat)
    test(imgPt, objPt, tmat, camMtx)

    #
    # print("img2")
    # retval, rvec, tvec = cv2.solvePnP(objPt, imgPt2, camMtx, None, flags=flag)
    # print(retval, rvec, tvec)
    # rmat, jac = cv2.Rodrigues(rvec)
    # print rpy.rpy(rmat)
    # print np.dot(rmat, tvec)
    # tmat2 = np.zeros((3,4))
    # tmat2[:3,:3] = rmat
    # tmat2[:3,3] = tvec.T
    # print(tmat2)
    #
    # print np.dot(camMtx, tmat2)
    # print "..................."
    #
    # test(imgPt2, objPt, tmat2, camMtx)
