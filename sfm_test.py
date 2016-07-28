import reconstr
import cv2
import numpy as np
import numpy.linalg as linalg
import pickle
from rpy import getTransform

def matNorm(mat):
    return np.sum(np.abs(mat))

def getRandCam():
    foc = np.random.random() * 3000 + 1000
    cx = np.random.random() * 1000 + 1000
    cy = np.random.random() * 1000 + 1000
    return np.array([[foc, 0, cx],[0, foc, cy],[0, 0, 1]])

def getRandTrf():
    r,p,y = tuple(np.random.random(3) * 2 * np.pi - np.pi)
    tx, ty, tz = tuple(np.random.random(3) * 100 - 50)
    return getTransform(r, p, y, tx, ty, tz)


def solve_sfm(img_pts, projs):
    num_imgs = len(img_pts)

    G = np.zeros((num_imgs * 2, 3), np.float64)
    b = np.zeros((num_imgs * 2, 1), np.float64)
    for i in range(num_imgs):
        ui = img_pts[i][0]
        vi = img_pts[i][1]
        p1i = projs[i][0, :3]
        p2i = projs[i][1, :3]
        p3i = projs[i][2, :3]
        a1i = projs[i][0, 3]
        a2i = projs[i][1, 3]
        a3i = projs[i][2, 3]
        idx1 = i * 2
        idx2 = idx1 + 1

        G[idx1, :] = ui * p3i - p1i
        G[idx2, :] = vi * p3i - p2i
        b[idx1] = a1i - a3i * ui
        b[idx2] = a2i - a3i * vi
    x = np.dot(np.linalg.pinv(G), b)
    return x, G, b


#img_pts[img_idx] == img_point[3]
#projs[img_idx] = proj[3x4]

if __name__ == "__main__":
    # num_imgs = 20
    # num_pts = 20
    #
    # cam = getRandCam()
    # trfs = [getRandTrf() for i in range(num_imgs)]
    # projs = [np.dot(cam, trf) for trf in trfs]
    # obj_pts = [np.random.rand(4, 1) * 60 - 30 for i in range(num_pts)]
    # for i in range(num_pts):
    #     obj_pts[i][3] = 1
    # # img_pts[img_idx][pt_idx]
    # img_pts = [[np.dot(projs[img_idx], obj_pt) for obj_pt in obj_pts] for img_idx in range(num_imgs)]
    # for i in range(num_imgs):
    #     for j in range((num_pts)):
    #         w = img_pts[i][j][2]
    #         img_pts[i][j][0] /= w
    #         img_pts[i][j][1] /= w
    #         img_pts[i][j][2] /= w


    num_imgs = 20

    cam = getRandCam()
    trfs = [getRandTrf() for i in range(num_imgs)]
    projs = [np.dot(cam, trf) for trf in trfs]
    obj_pt = np.random.rand(4, 1) * 20 - 10
    obj_pt[3] = 1

    # img_pts[img_idx]
    img_pts = [np.dot(proj, obj_pt) for proj in projs]
    for i in range(num_imgs):
        w = img_pts[i][2]
        img_pts[i][0] /= w
        img_pts[i][1] /= w
        img_pts[i][2] /= w
    x, G, b = solve_sfm(img_pts[:][:2], projs)
    print(matNorm(np.dot(G, x) - b))
    print(matNorm(np.dot(G, obj_pt[:3,:]) - b))

    print(x)
    print(obj_pt)



