import cv2
import Utils
import numpy as np
from random import random, seed, uniform


pointdict1 = {
    "set1/Picture 22.jpg": (0, 0, 5),
    "set1/Picture 23.jpg": (0, 0, 3),
    "set1/Picture 24.jpg": (0, 0, 7),
    "set1/Picture 25.jpg": (0, 2, 5),
    "set1/Picture 26.jpg": (0, 0, 5),
    "set1/Picture 27.jpg": (0, 0, 5),
    "set1/Picture 28.jpg": (2, 0, 5),
    "set1/Picture 29.jpg": (4, 0, 5),
    "set1/Picture 30.jpg": (0, 4, 5),
    "set1/Picture 31.jpg": (0, 0, 5),
}

pointdict2 = {
    "set1/Picture 32.jpg": (0, 0, 8),
    "set1/Picture 33.jpg": (0, 0, 6),
    "set1/Picture 34.jpg": (0, 2, 8),
    "set1/Picture 35.jpg": (0, 4, 8),
    "set1/Picture 36.jpg": (2, 0, 8),
    "set1/Picture 37.jpg": (4, 0, 8),
    "set1/Picture 38.jpg": (0, 0, 4)
}

cammtx = np.array(
    [1.8435610863515064e+003, 0., 7.995e+002, 0., 1.8435610863515064e+003, 5.995e+002, 0., 0., 1.]).reshape(
    (3, 3))
dist_coeffs = np.array([1.1415471237383623e-001, -1.4601522229886266e+000, 0., 0.,
                        5.1820223903354057e+000])

def normalize(img):
    mmin, mmax = np.min(img), np.max(img)
    return np.uint8((img - mmin) * 255.0 / (mmax - mmin))

def getPts(contours):
    contours = contours[:3]
    cogs = [c.reshape((4, 2)).sum(axis=0) / 4.0 for c in contours]
    up = (cogs[0] + cogs[1]) / 2 - cogs[2]
    right = np.zeros_like(up)
    right[0] = -up[1]
    right[1] = up[0]

    allpts = []
    for c in contours:
        for r in range(c.shape[0]):
            allpts.append(c[r])
    globcog = sum(cogs) / 3
    allpts = [pt - globcog for pt in allpts]
    p00 = max(allpts, key=lambda pt: np.dot(pt, up) + np.dot(pt, -right)) + globcog
    p01 = max(allpts, key=lambda pt: np.dot(pt, up) + np.dot(pt, right)) + globcog
    p10 = max(allpts, key=lambda pt: np.dot(pt, -up) + np.dot(pt, -right)) + globcog
    p11 = max(allpts, key=lambda pt: np.dot(pt, -up) + np.dot(pt, right)) + globcog

    return [p00[0], p01[0], p10[0], p11[0]]

def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.
    Using the Kabsch algorithm with two sets of paired point P and Q,
    centered around the center-of-mass.
    Each vector set is represented as an NxD matrix, where D is the
    the dimension of the space.
    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    http://en.wikipedia.org/wiki/Kabsch_algorithm

    :param P: (N, number of points)x(D, dimension) matrix
    :param Q: (N, number of points)x(D, dimension) matrix
    :return: U -- Rotation matrix
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U

def img_test():
    pointdict = pointdict1

    objpts = np.array([[0, 0, 0], [6.043, 0, 0], [0, 6.043, 0], [6.043, 6.043, 0]])
    robot_coords = [pointdict[k] for k in pointdict.keys()]
    imgpts = []

    for k in pointdict.keys():
        img = cv2.imread(k)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = normalize(gray)
        _, dil = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        cv2.imshow("dil", cv2.pyrDown(dil))

        contours, hierarchy = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = filter_contours(contours)

        imgpts_curr = getPts(contours)

        cv2.circle(img, (int(imgpts_curr[1][0]), int(imgpts_curr[1][1])), 10, (255, 0, 0), 4)
        cv2.circle(img, (int(imgpts_curr[2][0]), int(imgpts_curr[2][1])), 10, (255, 0, 0), 4)
        cv2.circle(img, (int(imgpts_curr[3][0]), int(imgpts_curr[3][1])), 10, (255, 0, 0), 4)
        cv2.circle(img, (int(imgpts_curr[0][0]), int(imgpts_curr[0][1])), 10, (0, 0, 255), 4)
        cv2.drawContours(img, contours, -1, (0, 255, 128), 4)
        img = cv2.pyrDown(img)
        cv2.imshow("", img)
        # cv2.waitKey()

        imgpts_curr = np.array(imgpts_curr)
        imgpts.append(imgpts_curr)

    rot, toc = calc_rot(imgpts, objpts, robot_coords)

    print Utils.rpy(rot)
    print rot
    # print vrt_np
    # print voc_np
    # print voc_np - vrt_np.dot(rot.T)

    # for i in range(numpts):
    #     for j in range(i + 1, numpts):
    #         print "-", i, j
    #         print np.linalg.norm(vrt_np[i, :] - vrt_np[j, :])
    #         print np.linalg.norm(voc_np[i, :] - voc_np[j, :])

def calc_rot(imgpts, objpts, robot_coords, use_dist_coeffs = False):
    global cammtx, dist_coeffs
    numpts = len(imgpts)
    voc_np = np.zeros((numpts, 3))
    vrt_np = np.zeros((numpts, 3))
    toc = []
    for i in range(numpts):
        imgpts_i = imgpts[i]
        if objpts.shape[1] == 3:
            objpts = objpts.T
        if imgpts_i.shape[1] == 2:
            imgpts_i = imgpts_i.T
        retval, rvec, tvec = cv2.solvePnP(objpts.T, imgpts_i.T, cammtx, dist_coeffs if use_dist_coeffs else None)
        rmat, _ = cv2.Rodrigues(rvec)
        tmat = np.eye(4)
        tmat[:3, :3] = rmat
        tmat[:3, 3] = tvec.T
        toci = np.linalg.inv(tmat)
        toc.append(toci)

        voci = toci[:3, 3]

        # print voci
        # print robot_coords[i]
        voc_np[i, :] = voci.reshape((3,))
        vrt_np[i, :] = np.array(robot_coords[i][:3], dtype=float).reshape((3,))

    voc_np -= np.sum(voc_np, 0) / voc_np.shape[0]
    vrt_np -= np.sum(vrt_np, 0) / vrt_np.shape[0]
    rot = kabsch(vrt_np, voc_np)
    return rot.T, toc

def calc_trans(imgpts, objpts, robot_coords, ror, use_dist_coeffs = False):
    global cammtx, dist_coeffs

    numpts = len(imgpts)

    # pi = ror' * voci - vrti
    # Ai = rrti
    # B = ror'
    B = ror.T
    M = np.zeros((3 * numpts, 6))
    K = np.zeros((3 * numpts, 1))

    for i in range(numpts):
        imgpts_i = imgpts[i]
        if objpts.shape[1] == 3:
            objpts = objpts.T
        if imgpts_i.shape[1] == 2:
            imgpts_i = imgpts_i.T
        retval, rvec, tvec = cv2.solvePnP(objpts.T, imgpts_i.T, cammtx, dist_coeffs if use_dist_coeffs else None)
        rmat, _ = cv2.Rodrigues(rvec)
        tmat = np.eye(4)
        tmat[:3, :3] = rmat
        tmat[:3, 3] = tvec.T
        toc = np.linalg.inv(tmat)

        voci = toc[:3, 3]

        x, y, z, a, b, c = robot_coords[i]
        trti = Utils.getTransform(c, b, a, x, y, z, True)
        print i
        print trti
        print toc

        vrti = trti[:3, 3]
        pi = ror.T.dot(voci) - vrti
        Ai = trti[:3, :3]
        M[(3 * i): (3 * i + 3), :3] = Ai
        M[(3 * i): (3 * i + 3), 3:] = B
        K[(3 * i): (3 * i + 3), 0] = pi

    x = np.linalg.pinv(M).dot(K)
    return x

def calc_avg_rot(rot_matrices):
    num_matrices = len(rot_matrices)
    bigrot = np.zeros((num_matrices * 3, 3))
    bigeye = np.zeros((num_matrices * 3, 3))
    for i in range(num_matrices):
        bigrot[i:(i+3), :] = rot_matrices[i]
        bigeye[i:(i+3), :] = np.eye(3)
    return kabsch(bigeye, bigrot)

def filter_contours(contours):
    # print  len(contours)
    # h, w = dil.shape
    # area = h * w
    # contours = [c for c in contours if cv2.contourArea(c) > area / 8]
    # print  len(contours)
    contours = [cv2.approxPolyDP(c, 20, True) for c in contours]
    # print  len(contours)
    contours = [c for c in contours if c.shape[0] == 4]
    # print len(contours)
    contours = [c for c in sorted(contours, key=lambda c: cv2.contourArea(c))]
    contours = [c for c in contours if cv2.contourArea(c) > 5000]
    return contours

def add_noise(pts, amplitude):
    pts2 = [None] * len(pts)
    for i in range(len(pts)):
        pt = pts[i]
        noise = np.random.random(pt.shape) * 2 - 1
        noise *= amplitude
        pts2[i] = pt + noise
    return pts2

def test(sd = 10):
    seed(sd)
    np.random.seed(sd)
    np.set_printoptions(precision=5, suppress=True)

    num_imgs = 10
    num_obj_pts = 20
    obj_pts = np.random.random((3, num_obj_pts))
    obj_pts_homog = np.ones((4, num_obj_pts))
    obj_pts_homog[:3, :] = obj_pts

    tmats_rt = [None] * num_imgs
    rr, pp, yy = (uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10)
    # rr, pp, yy = 0, 0, 0
    img_pts = [None] * num_imgs
    robot_coords = [None] * num_imgs

    tmat_or = get_rand_trf()
    tmat_tc = get_rand_trf()

    for i in range(num_imgs):
        robot_coords[i] = map(int, (uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10))
        x, y, z = robot_coords[i]
        tmats_rt[i] =  Utils.getTransform(rr, pp, yy, x, y, z, True)
        # print tmats_rt[i]
        tmat_oc = tmat_or.dot(tmats_rt[i].dot(tmat_tc))
        tmat_co = np.linalg.inv(tmat_oc)
        # print tmat_co
        cam_pts = tmat_co.dot(obj_pts_homog)
        proj_pts = cammtx.dot(cam_pts[:3, :])
        for j in range(num_obj_pts):
            proj_pts[:, j] /= proj_pts[2, j]
        img_pts[i] = proj_pts[:2, :]

    img_pts = add_noise(img_pts, 5)

    ror_est, toc_est = calc_rot(img_pts, obj_pts, robot_coords)
    roc_est = calc_avg_rot([toci[:3, :3] for toci in toc_est])
    rrt = Utils.getTransform(rr, pp, yy, 0, 0, 0, True)[:3, :3]
    rtc_est = rrt.T.dot(ror_est.T.dot(roc_est))

    Ror, toc = calc_rot(img_pts, obj_pts, robot_coords)
    print "---"
    print ror_est
    print tmat_or
    print tmat_or[:3,:3] - ror_est
    print "-----------"

    num_imgs2 = 1000

    tmats_rt_trans = [None] * num_imgs2
    img_pts_trans = [None] * num_imgs2
    robot_coords_trans = [None] * num_imgs2

    for i in range(num_imgs2):
        robot_coords_trans[i] = map(int, (uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10))
        x, y, z, a, b, c = robot_coords_trans[i]
        tmats_rt_trans[i] = Utils.getTransform(c, b, a, x, y, z, True)
        # print tmats_rt[i]
        tmat_oc = tmat_or.dot(tmats_rt_trans[i].dot(tmat_tc))
        tmat_co = np.linalg.inv(tmat_oc)
        cam_pts = tmat_co.dot(obj_pts_homog)
        proj_pts = cammtx.dot(cam_pts[:3, :])
        for j in range(num_obj_pts):
            proj_pts[:, j] /= proj_pts[2, j]
        img_pts_trans[i] = proj_pts[:2, :]

        if i == 776:
            print ".."
            print i
            print x, y, z, a, b, c
            print tmat_co
            print tmat_oc
            print proj_pts[:2, :]
            print ".."

    img_pts = add_noise(img_pts, 5)

    # img_pts = [img_pts[i] for i in range(len(img_pts)) if i != 776]
    # robot_coords = [robot_coords[i] for i in range(len(robot_coords)) if i != 776]
    x_est = calc_trans(img_pts_trans, obj_pts, robot_coords_trans, ror_est)
    print tmat_tc
    print tmat_or
    print x_est
    print x_est[:3,0] - tmat_tc[:3, 3]
    print x_est[3:,0] - tmat_or[:3, 3]

    vtc_est = x_est[:3, :]
    vor_est = x_est[3:, :]
    tor_est = np.eye(4)
    tor_est[:3, :3] = ror_est
    tor_est[:3, 3] = vor_est.reshape((3,))
    ttc_est = np.eye(4)
    ttc_est[:3, :3] = rtc_est
    ttc_est[:3, 3] = vtc_est.reshape((3,))

    return max(np.max(x_est[:3,0] - tmat_tc[:3, 3]), np.max(x_est[3:,0] - tmat_or[:3, 3]))

def get_rand_trf():
    rand_trf = Utils.getTransform(uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10, uniform(-1, 1) * 10,
                                 uniform(-1, 1) * 10, uniform(-1, 1) * 10, True)
    return rand_trf

if __name__ == '__main__':
    # img_test()
    print test(97)
    necces_seedek = [10, 20, 48, 53, 69, 78, 96, 97]

    # seed  err             ludas
    # 10    32.6704391356   776
    # 20    197.964954352   166
    # 48    201.696790972   434
    # 53    7373426.10986   605
    # 69    276871753.915   961
    # 78    514334413.382   131
    # 96    4109638.9098    392
    # 97    352.850449091   931


    # v = np.random.random((3, 3))
    # rot, _, _ = np.linalg.svd(v)
    #
    # P = np.random.rand(10, 3)
    # centroid = np.sum(P, 0) / P.shape[0]
    # P = P - np.ones((P.shape[0], 1)).dot(centroid.reshape((1,-1)))
    #
    # Q = P.dot(rot)
    #
    # A = P.T.dot(Q)
    # V, S, W = np.linalg.svd(A)
    # d = np.linalg.det(V) * np.linalg.det(W)
    # print d
    # print V
    # if d < 0:
    #     S[2, 2] = -S[2, 2]
    #     V[:,2]= -V[:,2]
    # print V
    # U = V.dot(W)
    #
    # print "------------"
    # print rot
    # print U
    # print kabsch(P, Q)
    # print Q - P.dot(U)

    pass