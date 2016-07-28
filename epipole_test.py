import numpy as np
import Utils as util
import sfm_test as sfm
from pprint import pprint
from numpy.linalg import inv

def getCrossMat(t):
    tx = t[0]
    ty = t[1]
    tz = t[2]
    return np.array(
        [[0, -tz, ty],
         [tz, 0, -tx],
         [-ty, tx, 0]])

def getEssentialMat(R, t):
    tx = getCrossMat(t)
    E = np.dot(R, tx)
    return E

def getFundamentalMat(E, A1, A2):
    F = np.dot(np.dot(inv(A2.T), E), inv(A1))
    return F

def normalize(vec):
    return vec / vec[-1]

def test_epip():
    cam1 = sfm.getRandCam()
    cam2 = sfm.getRandCam()
    trf1 = sfm.getRandTrf()
    trf2 = sfm.getRandTrf()
    # trf1 = util.getTransform(0, 0, 0,  1,3, 4)
    # trf2 = util.getTransform(10, 0, 0,  5, 6, 7)
    A1 = np.eye(4)
    A2 = np.eye(4)
    A1[:3, :4] = trf1
    A2[:3, :4] = trf2

    obj_pt = np.random.rand(4, 1) * 20 - 10
    obj_pt[3] = 1
    obj_pt = np.array([10, 0, 0, 1]).T

    cam_pt1 = np.dot(A1, obj_pt)
    cam_pt1 = normalize(cam_pt1)[:3]
    print "cam_pt1\r\n",cam_pt1
    im_pt1 = np.dot(cam1, cam_pt1)
    im_pt1 = normalize(im_pt1)

    cam_pt2 = np.dot(A2, obj_pt)
    cam_pt2 = normalize(cam_pt2)[:3]
    print "cam_pt2\r\n", cam_pt2
    im_pt2 = np.dot(cam2, cam_pt2)
    im_pt2 = normalize(im_pt2)

    trf = np.dot(A2, inv(A1))
    trf = np.dot(A1, inv(A2))
    # trf = np.dot(inv(A2), (A1))
    # trf = np.dot(inv(A1), (A2))
    #
    # trf = inv(np.dot(A2, inv(A1)))
    # trf = inv(np.dot(A1, inv(A2)))
    # trf = inv(np.dot(inv(A2), (A1)))
    # trf = inv(np.dot(inv(A1), (A2)))


    R = trf[:3, :3].T
    t = trf[:3, 3]
    print "---"
    print t
    print R

    E = getEssentialMat(R, t)
    F = getFundamentalMat(E, cam1, cam2)


    print np.dot(im_pt2.T, np.dot(F, im_pt1))
    print np.dot(cam_pt2.T, np.dot(E, cam_pt1))


if __name__ == '__main__':

    # obj_pt = np.random.rand(4, 1) * 20 - 10
    # obj_pt[3] = 1
    # obj_pt2 = np.random.rand(4, 1) * 20 - 10
    # obj_pt2[3] = 1
    # print getCrossMat(obj_pt2).dot(obj_pt[:3]).T - np.cross(obj_pt2[:3].T, obj_pt[:3].T)
    #
    # obj_pt = np.random.rand(4, 1) * 20 - 10
    # obj_pt[3] = 1

    test_epip()
    # for i in range(20):
    #     test_epip()