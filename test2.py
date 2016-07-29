import cv2
import numpy as np
import FeatureLoader
import MatchLoader
import Utils
import MarkerDetect
import cProfile

def maskKeypoints(masks, kpts):
    num = len(masks)
    print("-- masking --")
    print([len(kpl[1]) for kpl in kpts])
    for i in range(num):
        kp, des = kpts[i]
        j = 0
        while j < len(kp):
            pt = kp[j].pt
            x = int(pt[0])
            y = int(pt[1])
            if masks[i][y, x] > 100:
                j += 1
            else:
                kp.pop(j)
                des.pop(j)
    print([len(kpl[1]) for kpl in kpts])
    return kpts

def drawKpts(imgs, kpts):
    # print len(kpts[0])
    # print len(kpts[0][0])
    for i in range(len(imgs)):
        img_ = imgs[i]
        img = np.copy(img_)
        # print len(kpts[i])
        # kp, des = kpts[i]
        # print
        for j in range(0, len(kpts[i][0]), 10):
            kp = kpts[i][0][j]
            cv2.circle(img, tuple(map(int, kp.pt)), 20, (0, 255, 255), 4)
        img2 = cv2.pyrDown(img)
        img2 = cv2.pyrDown(img2)
        cv2.imshow("asd", img2)
        cv2.waitKey()


def test():
    files = ["imgs/00%d.jpg" % (i) for i in range(5, 10)]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    num = len(files)

    cst = 100 * 1000
    imgs = [cv2.imread(f) for f in files]
    fl = FeatureLoader.FeatureLoader()
    ml = MatchLoader.MatchLoader()
    kpts = [fl.loadFeatures(f, "surf") for f in files]
    kpts = maskKeypoints(masks, kpts)
    # drawKpts(imgs, kpts)
    matches = [[None] * num for i in range(num)]
    tmats = [MarkerDetect.loadMat(f) for f in files]

    for i in range(num):
        for j in range(num):
            if i == j: continue
            print(i,j)
            matches[i][j] = ml.matchBFCrossTmats(
                files[i], files[j], kpts[i][1], kpts[j][1], kpts[i][0], kpts[j][0], tmats[i], tmats[j], "surf", version="0", nosave=True)
            # Utils.drawMatchesOneByOne(imgs[i], imgs[j], kpts[i][0], kpts[j][0], matches[i][j])
            return

if __name__ == '__main__':
    cProfile.run('test()')
