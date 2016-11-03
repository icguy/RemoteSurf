import cv2
import numpy as np
import Utils

pointdict1 = {
    "set1/Picture 22.jpg" : (0, 0, 50),
    "set1/Picture 23.jpg" : (0, 0, 30),
    "set1/Picture 24.jpg" : (0, 0, 70),
    "set1/Picture 25.jpg" : (0, 20, 50),
    "set1/Picture 26.jpg" : (0, 0, 50),
    "set1/Picture 27.jpg" : (0, 0, 50),
    "set1/Picture 28.jpg" : (20, 0, 50),
    "set1/Picture 29.jpg" : (40, 0, 50),
    "set1/Picture 30.jpg" : (0, 40, 50),
    "set1/Picture 31.jpg" : (0, 0, 50),
}

pointdict2 = {
    "set1/Picture 32.jpg" : (0, 0, 80),
    "set1/Picture 33.jpg" : (0, 0, 60),
    "set1/Picture 34.jpg" : (0, 20, 80),
    "set1/Picture 35.jpg" : (0, 40, 80),
    "set1/Picture 36.jpg" : (20, 0, 80),
    "set1/Picture 37.jpg" : (40, 0, 80),
    "set1/Picture 38.jpg" : (0, 0, 40)
}

def normalize(img):
    mmin, mmax = np.min(img), np.max(img)
    return np.uint8((img - mmin) * 255.0 / (mmax - mmin))

def getPts(contours):
    contours = contours[:3]
    cogs = [c.reshape((4, 2)).sum(axis = 0) / 4.0 for c in contours]
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
    p00 = max(allpts, key = lambda pt: np.dot(pt, up) + np.dot(pt, -right)) + globcog
    p01 = max(allpts, key = lambda pt: np.dot(pt, up) + np.dot(pt, right)) + globcog
    p10 = max(allpts, key = lambda pt: np.dot(pt, -up) + np.dot(pt, -right)) + globcog
    p11 = max(allpts, key = lambda pt: np.dot(pt, -up) + np.dot(pt, right)) + globcog

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

def test():
    voc = []
    vrt = []
    pointdict = pointdict2
    for k in pointdict.keys():
        img = cv2.imread(k)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = normalize(gray)
        _, dil = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        cv2.imshow("dil", cv2.pyrDown(dil))

        contours, hierarchy = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = filter_contours(contours)

        imgpts = getPts(contours)

        cv2.circle(img, (int(imgpts[1][0]), int(imgpts[1][1])), 10, (255, 0, 0), 4)
        cv2.circle(img, (int(imgpts[2][0]), int(imgpts[2][1])), 10, (255, 0, 0), 4)
        cv2.circle(img, (int(imgpts[3][0]), int(imgpts[3][1])), 10, (255, 0, 0), 4)
        cv2.circle(img, (int(imgpts[0][0]), int(imgpts[0][1])), 10, (0, 0, 255), 4)
        cv2.drawContours(img, contours, -1, (0, 255, 128), 4)

        cammtx = np.array(
            [1.8435610863515064e+003, 0., 7.995e+002, 0., 1.8435610863515064e+003, 5.995e+002, 0., 0., 1.]).reshape(
            (3, 3))
        objpts = np.array([[0, 0, 0], [6.043, 0, 0], [0, 6.043, 0], [6.043, 6.043, 0]])
        imgpts = np.array(imgpts)
        retval, rvec, tvec = cv2.solvePnP(objpts.astype("float32"), imgpts.astype("float32"), cammtx, None)
        rmat, _ = cv2.Rodrigues(rvec)
        tmat = np.eye(4)
        tmat[:3, :3] = rmat
        tmat[:3, 3] = tvec.T
        # print tmat
        print map(lambda c: c * 180 / 3.1416, Utils.rpy(rmat))
        tmatinv = np.linalg.inv(tmat)
        voci = tmatinv[:3, 3]
        voc.append(voci)
        vrt.append(-np.array(pointdict[k], dtype=float).reshape((3, 1)))

        img = cv2.pyrDown(img)
        cv2.imshow("", img)
        # cv2.waitKey()

    numpts = len(voc)
    voc_np = np.zeros((numpts, 3))
    vrt_np = np.zeros((numpts, 3))
    for i in range(numpts):
        voc_np[i, :] = voc[i].reshape((3,))
        vrt_np[i, :] = vrt[i].reshape((3,)) / 10
    voc_np -= np.sum(voc_np, 0) / voc_np.shape[0]
    vrt_np -= np.sum(vrt_np, 0) / vrt_np.shape[0]
    rot = kabsch(vrt_np, voc_np)
    print rot
    print vrt_np
    print voc_np
    print voc_np - vrt_np.dot(rot)
    # for i in range(numpts):
    #     for j in range(i + 1, numpts):
    #         print "-", i, j
    #         print np.linalg.norm(vrt_np[i, :] - vrt_np[j, :])
    #         print np.linalg.norm(voc_np[i, :] - voc_np[j, :])

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

if __name__ == '__main__':
    # test()
    v = np.random.random((3, 3))
    rot, _, _ = np.linalg.svd(v)

    P = np.random.rand(10, 3)
    centroid = np.sum(P, 0) / P.shape[0]
    P = P - np.ones((P.shape[0], 1)).dot(centroid.reshape((1,-1)))

    Q = P.dot(rot)

    A = P.T.dot(Q)
    V, S, W = np.linalg.svd(A)
    d = np.linalg.det(V) * np.linalg.det(W)
    print d
    print V
    if d < 0:
        S[2, 2] = -S[2, 2]
        V[:,2]= -V[:,2]
    print V
    U = V.dot(W)

    print "------------"
    print rot
    print U
    print kabsch(P, Q)
    # print Q - P.dot(U)

