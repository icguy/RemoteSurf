import cv2
import numpy as np
import cPickle as pickle
from os.path import isfile

MATCHER_FLANN_RATIO_07 = "flann_ratio"
MATCHER_BF_RATIO_07 = "bf_ratio"
MATCHER_BF_CROSS = "bf_cross"
ratio_thresh = 0.7

def getMatcher(name):
    global ratio_thresh

    if name == MATCHER_FLANN_RATIO_07:
        ratio_thresh = 0.7
        return matchFLANNRatio
    elif name == MATCHER_BF_RATIO_07:
        ratio_thresh = 0.7
        return matchBFRatio
    elif name == MATCHER_BF_CROSS:
        return matchBFCross
    return matchFLANNRatio

def matchBFRatio(des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors.
    matches = bf.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

    good = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    good.sort(key = lambda x: x.distance, reverse = True)
    return good

def matchFLANNRatio(des1, des2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    good.sort(key = lambda x: x.distance, reverse = True)
    return good

def matchBFCross(des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(np.asarray(des1,np.float32),np.asarray(des2,np.float32))

    return matches

# dump file name: "detect/filename.detectorType.p"
class MatchLoader:
    def __init__(self):
        pass

    def getFileName(self, fn1, fn2, dType, mType, version):
        fn1 = fn1[fn1.rindex("/") + 1:]
        fn2 = fn2[fn2.rindex("/") + 1:]
        return "detect/match_%s.%s.%s.%s.%s.p" % (fn1, fn2, dType, mType, version)

    def loadMatches(self, filename1, filename2, des1, des2, detectorType, matcherType, version ="0"):
        detectorType = detectorType.lower()
        matcherType = matcherType.lower()

        fname = self.getFileName(filename1, filename2, detectorType, matcherType, version)
        if isfile(fname):
            f = open(fname, "rb")
            matches = pickle.load(f)
            f.close()
            return self.deserializeMatches(matches)

        matcherFunc = getMatcher(matcherType)
        matches = matcherFunc(des1, des2)

        f = open(fname, "wb")
        pickle.dump(self.serializeMatches(matches), f, 2)
        f.close()

        return matches

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
    m = ml.loadMatches(fn1, fn2, des, des2, "surf", MATCHER_FLANN_RATIO_07)
    print len(m)
    print([(g.trainIdx, g.queryIdx) for g in m])



