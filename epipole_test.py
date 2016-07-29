import numpy as np
import Utils as util
import sfm_test as sfm
from pprint import pprint
from numpy.linalg import inv
from Utils import calcEssentialFundamentalMat

def normalize(vec):
    return vec / vec[-1]

def test_epip():
    cam1 = sfm.getRandCam()
    cam2 = sfm.getRandCam()
    trf1 = sfm.getRandTrf()
    trf2 = sfm.getRandTrf()

    E, F = calcEssentialFundamentalMat(cam1, cam2, trf1, trf2)

    obj_pt = np.random.rand(4, 1) * 20 - 10
    obj_pt[3] = 1
    obj_pt = np.array([10, 0, 0, 1]).T

    cam_pt1, cam_pt2, im_pt1, im_pt2 = calc_test_points(cam1, cam2, obj_pt, trf1, trf2)

    print np.dot(im_pt2.T, np.dot(F, im_pt1))
    print np.dot(cam_pt2.T, np.dot(E, cam_pt1))

def calc_test_points(cam1, cam2, obj_pt, trf1, trf2):
    A1 = np.eye(4)
    A2 = np.eye(4)
    A1[:3, :4] = trf1
    A2[:3, :4] = trf2

    cam_pt1 = np.dot(A1, obj_pt)
    cam_pt1 = normalize(cam_pt1)[:3]
    print "cam_pt1\r\n", cam_pt1
    im_pt1 = np.dot(cam1, cam_pt1)
    im_pt1 = normalize(im_pt1)

    cam_pt2 = np.dot(A2, obj_pt)
    cam_pt2 = normalize(cam_pt2)[:3]
    print "cam_pt2\r\n", cam_pt2
    im_pt2 = np.dot(cam2, cam_pt2)
    im_pt2 = normalize(im_pt2)

    return cam_pt1, cam_pt2, im_pt1, im_pt2

if __name__ == '__main__':
    test_epip()

    import cv2, FeatureLoader as FL, MarkerDetect as MD
    files = ["imgs/005.jpg", "imgs/006.jpg"]
    imgs = [cv2.imread(f) for f in files]
    fl = FL.FeatureLoader()
    kpts = [fl.loadFeatures(f, "surf") for f in files]
    kpt_list = kpts[0]

    img1 = cv2.pyrDown(imgs[0])
    img2 = cv2.pyrDown(imgs[1])

    tmats = [MD.loadMat(f) for f in files]
    

    for kpt in kpt_list:
        realscale = 2

        h, w, c = img1.shape
        out = np.zeros((h, w * 2, 3), np.uint8)
        out[:, :w, :] = img1
        out[:, w:, :] = img2

        color = (255, 255, 0)  # cyan

        pt1 = kpt.pt
        p1 = (int(pt1[0] / realscale), int(pt1[1] / realscale))
        cv2.circle(out, p1, 10, color, 1)
        cv2.imshow("match", out)
