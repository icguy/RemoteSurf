import cv2
import numpy as np
import cPickle as pickle
from os.path import isfile
import Utils as util

MATCHER_FLANN_RATIO_07 = "flann_ratio"
MATCHER_BF_RATIO_07 = "bf_ratio"
MATCHER_BF_CROSS = "bf_cross"
MATCHER_BF_CROSS_TMATS = "bf_cross_tmats"

# dump file name: "detect/filename.detectorType.matcherType.version.p"
class MatchLoader:
    def getFileName(self, fn1, fn2, dType, mType, version):
        fn1 = fn1[fn1.rindex("/") + 1:]
        fn2 = fn2[fn2.rindex("/") + 1:]
        return "detect/match_%s.%s.%s.%s.%s.p" % (fn1, fn2, dType, mType, version)

    def loadMatches(self, filename1, filename2, detectorType, matcherType, version):
        detectorType = detectorType.lower()
        matcherType = matcherType.lower()

        fname = self.getFileName(filename1, filename2, detectorType, matcherType, version)
        if isfile(fname):
            f = open(fname, "rb")
            matches = pickle.load(f)
            f.close()
            return fname, self.deserializeMatches(matches)

        return fname, None

    def matchBFCross(self, filename1, filename2, des1, des2, detectorType, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_RATIO_07, version)
        if matches is not None and not noload:
            return matches

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        good = bf.match(np.asarray(des1, np.float32), np.asarray(des2, np.float32))

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(good), f, 2)
            f.close()

        return good

    def matchBFRatio(self, filename1, filename2, des1, des2, detectorType, ratio = 0.7, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_RATIO_07, version)
        if matches is not None and not noload:
            return matches

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Match descriptors.
        matches = bf.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        good.sort(key=lambda x: x.distance, reverse=True)

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(good), f, 2)
            f.close()

        return good

    def matchBFCrossTmats(self, filename1, filename2, des1, des2, kpts1, kpts2,
                          tmats, detectorType, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_RATIO_07, version)
        if matches is not None and not noload:
            return matches

        allmatches = []
        num1, num2 = len(kpts1), len(kpts2)
        cam = util.camMtx
        E, F = util.calcEssentialFundamentalMat(cam, cam, tmats[0], tmats[1])

        dist_thr = 20 ** 2 #distance threshold from epiline

        #match img1 against img2
        for i in range(num1):
            pt1 = kpts1[i].pt
            idx_list = []

            pt1_h = np.ones((3, 1))
            pt1_h[0] = pt1[0]
            pt1_h[1] = pt1[1]
            pt1_h = pt1_h.T
            n = np.dot(pt1_h, F.T).T
            nx, ny, nz = n[0], n[1], n[2]

            for j in range(num2):
                pt2 = kpts2[j].pt
                dist_sq = (nx * pt2[0] + ny * pt2[1] + nz) ** 2 / (nx * nx + ny * ny)
                if dist_sq < dist_thr:
                    idx_list.append(j)

            #todo draw keypoints

            des_list1 = [des1[i]]
            des_list2 = [des2[j] for j in idx_list]





        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(allmatches), f, 2)
            f.close()

        return allmatches

    def matchFLANNRatio(self, filename1, filename2, des1, des2, detectorType, ratio = 0.7, version = "0", noload = False, nosave = False):
        fname, matches = self.loadMatches(filename1, filename2, detectorType, MATCHER_BF_RATIO_07, version)
        if matches is not None and not noload:
            return matches

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        good.sort(key=lambda x: x.distance, reverse=True)

        if not nosave:
            f = open(fname, "wb")
            pickle.dump(self.serializeMatches(good), f, 2)
            f.close()

        return good

    def serializeMatches(self, matches):
        return [(m.distance, m.imgIdx, m.queryIdx, m.trainIdx) for m in matches]

    def deserializeMatches(self, serd):
        return [
            cv2.DMatch(_queryIdx = m[2], _trainIdx = m[3], _imgIdx = m[1], _distance = m[0])
            for m in serd]


def print_rand(arr, idxs):
    sel = [arr[idx] for idx in idxs]
    print(sel)

if __name__ == "__main__":
    import FeatureLoader as FL
    import random
    from pprint import pprint
    fl = FL.FeatureLoader()

    fn1 = "imgs/004.jpg"
    fn2 = "imgs/005.jpg"
    img1 = cv2.imread(fn1)
    img2 = cv2.imread(fn2)
    kp, des = fl.loadFeatures(fn1, "SURF")
    kp2, des2 = fl.loadFeatures(fn2, "SURF")
    print(len(des), len(des2))

    ml = MatchLoader()
    m = ml.matchFLANNRatio(fn1, fn2, des, des2, "surf", 0.7, "07")
    print len(m)
    print([(g.trainIdx, g.queryIdx) for g in m])



