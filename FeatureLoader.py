import cv2
import numpy as np
import cPickle as pickle
from os.path import isfile

FEATURE_SURF = "surf"

def getDetector(name):
    name = name.lower()
    if name == "surf":
        return cv2.SURF()

    return cv2.SURF()

# dump file name: "cache/filename.detectorType.p"
class FeatureLoader:
    def __init__(self):
        pass

    def getFileName(self, filename, dType):
        filename = filename[filename.rindex("/") + 1:]
        return "cache/feature_%s.%s.p" % (filename, dType)

    """
    :returns kp, des
    """
    def loadFeatures(self, filename, detectorType):
        detectorType = detectorType.lower()
        detector = getDetector(detectorType)
        fname = self.getFileName(filename, detectorType)
        if isfile(fname):
            f = open(fname, "rb")
            keyPts = pickle.load(f)
            f.close()
            return self.deserializeKeyPoints(keyPts)

        img = cv2.imread(filename, 0)
        kp, des = detector.detectAndCompute(img, None)

        f = open(fname, "wb")
        pickle.dump(self.serializeKeyPoints(kp, des), f, 2)
        f.close()

        return kp, des

    def serializeKeyPoints(self, kpts, dstors):
        return [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, des)
            for kp, des in zip(kpts, dstors)]

    def deserializeKeyPoints(self, serd):
        kpts = [cv2.KeyPoint(x=ser[0][0],y=ser[0][1],_size=ser[1], _angle=ser[2],
                            _response=ser[3], _octave=ser[4], _class_id=ser[5]) for ser in serd]
        dstors = [ser[6] for ser in serd]

        return kpts, dstors

def print_rand(arr, idxs):
    sel = [arr[idx] for idx in idxs]
    print(sel)

if __name__ == "__main__":
    import random
    fl = FeatureLoader()

    fn = "imgs/001.jpg"
    print("1")
    kp, des = fl.loadFeatures(fn, "SURF")
    kpd = fl.serializeKeyPoints(kp, des)
    idxs = [random.randint(0, len(kp)) for i in range(20)]
    print_rand(kpd, idxs)

    print("2")
    kp, des = fl.loadFeatures(fn, "SURF")
    kpd = fl.serializeKeyPoints(kp, des)
    print_rand(kpd, idxs)
    print("3")
