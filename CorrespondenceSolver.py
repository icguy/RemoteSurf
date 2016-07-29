import cv2
import numpy as np


# def mouseCallback(evt, x, y, flags, usrdata):
#
#     pass


class CorrespondenceSolver:    
    def __init__(self, files, scale = 2):
        self.files = files
        imgs = [cv2.imread(f) for f in files]
        self.imgs = imgs
        self.scale = 2
        if scale == 2:
            self.small_imgs = [cv2.pyrDown(img) for img in imgs]
        else:
            self.scale = 4
            self.small_imgs = [cv2.pyrDown(cv2.pyrDown(img)) for img in imgs]

        self.num_imgs = len(imgs)
        self.ratio_threshold = 0.5
        self.features = None
        self.matches = None

    def drawMatch(self, idx1, idx2, pt1, pt2, good):
        realscale = self.scale

        img1 = self.small_imgs[idx1]
        img2 = self.small_imgs[idx2]

        h, w, c = img1.shape
        out = np.zeros((h, w * 2, 3), np.uint8)
        out[:,:w,:] = img1
        out[:,w:,:] = img2
    
        color = (255, 255, 0) # cyan
        if good == False:
            color = (0, 0, 255)
        elif good == True:
            color = (0, 255, 0)
    
        p1 = (int(pt1[0] / realscale), int(pt1[1] / realscale))
        p2 = (int(pt2[0] / realscale + w), int(pt2[1] / realscale))
        text = "pt1(%d, %d), pt2(%d, %d)" % (int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
        cv2.putText(out, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.circle(out, p1, 10, color, 1)
        cv2.circle(out, p2, 10, color, 1)
        cv2.line(out, p1, p2, color, 1)
        cv2.imshow("match", out)
       
    #matches: list of tuple: (imgidx1, imgidx2, (x1, y1), (x2, y2))
    def getMatchesManually(self, matches):
        cv2.namedWindow("match")
        good = [None] * len(matches)
    
        i = 0
        while i < len(matches):
            match = matches[i]
            idx1, idx2, pt1, pt2 = match
            self.drawMatch(idx1, idx2, pt1, pt2, good[i])
            while True:
                c = cv2.waitKey()
                if c == ord('a'):
                    i-=1
                    break
                elif c == ord('d'):
                    i+=1
                    break
                elif c == ord('w'):
                    good[i] = True
                    i+=1
                    break
                elif c == ord('s'):
                    good[i] = False
                    i+=1
                    break
            if i < 0:
                i = 0
    
        retval = []
        for i in range(len(good)):
            if good[i] == True:
                retval.append(matches[i])
        return retval
    
    #return kp, des
    def getKeypoints(self, idx):
        from FeatureLoader import FeatureLoader as FL
        fl = FL()
        detector = cv2.SURF()
        dType = "SURF"
        kp, des = fl.loadFeatures(self.files[idx], detector, dType)
        return kp, des
    
    #return good
    def matchFeatures(self, idx1, idx2):
        from MatchLoader import MatchLoader as ML, MATCHER_FLANN_RATIO_07

        kp1, des1 = self.features[idx1]
        kp2, des2 = self.features[idx2]
        ml = ML()
        matches = ml.matchFLANNRatio(self.files[idx1], self.files[idx2], des1, des2, "surf", 0.7, "07")
        return matches
    
    def handleNewGoodMatch(self, goodmatches, i, j, k):
        """ megnezi, hogy van-e olyan keypoint egy 3. kepen ami
            mindket megtalalt keypointtal match-ben van, ha igen, azt is match-nek tekinti
        """

        kp_idx1, kp_idx2 = self.matches[i][j][k][0]
        goodmatches.append((i, j, k))
        for ii in range(self.num_imgs):
            if ii == i or ii == j:
                continue

            match1 = None
            match2 = None

            for kk in range(len(self.matches[i][ii])):
                m = self.matches[i][ii][kk][0]
                if m[0] == kp_idx1:
                    match1 = kk, m
                    break

            for kk in range(len(self.matches[j][ii])):
                m = self.matches[j][ii][kk][0]
                if m[0] == kp_idx2:
                    match2 = kk, m
                    break

            if match1 is not None and match2 is not None and match1[1][1] == match2[1][1]:
                #talaltunk egyet

                # if self.matches[i][ii][match1[0]][1]:
                #     self.handleNewGoodMatch(goodmatches, i, ii, match1[0])
                # else:
                #     self.matches[i][ii][match1[0]][1] = True #checked
                #
                # if self.matches[j][ii][match2[0]][1]:
                #     self.handleNewGoodMatch(goodmatches, j, ii, match2[0])
                # else:
                #     self.matches[j][ii][match2[0]][1] = True #checked

                self.matches[i][ii][match1[0]][1] = True #checked
                self.matches[j][ii][match2[0]][1] = True #checked

                goodmatches.append(self.matches[i][ii][match1[0]])
                goodmatches.append(self.matches[j][ii][match2[0]])

    #self.features[img_idx] = kp, des
    #self.matches[img_idx1][img_idx2][match_idx] = [(kp_idx1, kp_idx2), checked]  where
    #              kp_idx1 is index in self.features[img_idx1], checked is boolean
    def computeCorrespondence(self):
        self.features = [self.getKeypoints(img) for img in self.imgs]
        self.matches = [None] * self.num_imgs
        for i in range(self.num_imgs):
            self.matches[i] = [None] * self.num_imgs

        matchnum = 0
        for i in range(self.num_imgs):
            for j in range(i + 1, self.num_imgs):
                kp1, des1 = self.features[i]
                kp2, des2 = self.features[j]
                ms = self.matchFeatures(i, j)
                self.matches[i][j] = [[(m.queryIdx, m.trainIdx), False] for m in ms]
                self.matches[j][i] = [[(m.trainIdx, m.queryIdx), False] for m in ms]
                matchnum += len(ms)
    
        goodmatches = []
        curr_idx = 0
        for i in range(self.num_imgs):
            for j in range(i + 1, self.num_imgs):
                ms = self.matches[i][j]
                for k in range(len(ms)):
                    curr_idx += 1
                    m, checked = ms[k]
                    if checked:
                        continue

                    print("%d/%d" % (curr_idx, matchnum))
                    ms[k][1] = True #checked
    
                    good = self.getMatchesManually(
                        [(i, j, self.features[i][0][m[0]].pt, self.features[j][0][m[1]].pt)])

                    if len(good) == 1:
                        self.handleNewGoodMatch(goodmatches, i, j, k)

        return goodmatches


def main():
    global file1, file2, file3, img1, img2, img3, csolver, match, good, kp1, kp2, g, matches
    file1 = "imgs/001_.png"
    file2 = "imgs/002_.png"
    file3 = "imgs/011_.png"
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    img3 = cv2.imread(file3)
    csolver = CorrespondenceSolver([file1, file2, file3])
    # csolver = CorrespondenceSolver([file1, file2])
    csolver.computeCorrespondence()
    from feature_test import match

    good, kp1, kp2 = match("imgs/WP_20160713_002.jpg", "imgs/WP_20160713_004.jpg", True, 10)
    matches = [(0, 1, g[0].pt, g[1].pt) for g in good]
    csolver.getMatchesManually(matches)


if __name__ == "__main__":
    main()

