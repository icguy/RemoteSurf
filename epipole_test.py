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

    import cv2, FeatureLoader as FL, MarkerDetect as MD, Utils as util, MatchLoader as ML
    files = ["imgs/005.jpg", "imgs/006.jpg"]
    imgs = [cv2.imread(f) for f in files]
    fl = FL.FeatureLoader()
    kpts = [fl.loadFeatures(f, "surf") for f in files]
    ml = ML.MatchLoader()
    matches = ml.matchBFCross(files[0], files[1], kpts[0][1], kpts[1][1], "surf", "mask")

    img1 = imgs[0]
    img2 = imgs[1]

    tmats = [MD.loadMat(f) for f in files]
    cam = util.camMtx
    E, F = util.calcEssentialFundamentalMat(cam, cam, tmats[0], tmats[1])

    for m in matches:
        h, w, c = img1.shape
        out = np.zeros((h, w * 2, 3), np.uint8)
        out[:, :w, :] = img1
        out[:, w:, :] = img2


        kpt1 = kpts[0][0][m.queryIdx]
        kpt2 = kpts[1][0][m.trainIdx]

        pt1 = kpt1.pt
        pt2 = kpt2.pt
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0] + w), int(pt2[1]))
        pt1_h = np.ones((3, 1))
        pt1_h[0] = pt1[0]
        pt1_h[1] = pt1[1]
        pt1_h = pt1_h.T
        n = np.dot(pt1_h, F.T).T
        nx, ny, nz = n[0], n[1], n[2]

        u1 = 0
        u2 = w
        v1 = int((-nz-nx*u1)/ny)
        v2 = int((-nz-nx*u2)/ny)

        color = (255, 255, 0)  # cyan
        thk = 8
        cv2.circle(out, p1, 40, color, thk)
        cv2.circle(out, p2, 40, color, thk)
        cv2.line(out, (u1 + w, v1), (u2 + w, v2), color, thk)

        out = cv2.pyrDown(out)
        out = cv2.pyrDown(out)

        cv2.imshow("match", out)

        print "dist from epiline", np.sqrt((nx*pt2[0] + ny*pt2[1] + nz) ** 2 / (nx * nx + ny * ny))
        if cv2.waitKey() == 27:
            break
