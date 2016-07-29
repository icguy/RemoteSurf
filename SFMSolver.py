import cv2
import numpy as np
import FeatureLoader
import MatchLoader
import Utils
import MarkerDetect
from pprint import pprint

class CliqueExtractor:
    def getCliques(self, graph, max_num):
        triplets = []
        for k in graph:
            neigh = list(graph[k])
            num_n = len(neigh)
            for i in range(num_n):
                for j in range(i + 1, num_n):
                    n1 = neigh[i]
                    n2 = neigh[j]
                    if k in graph[n1] and k in graph[n2]:
                        triplets.append((k, n1, n2))
        num = 3
        cliques = triplets
        all_levels = [triplets]
        while num < max_num:
            cliques = self._add_level(graph, cliques)
            all_levels.append(list(cliques))
            num += 1
        return all_levels

    def _add_level(self, graph, cliques):
        newcliques = set()
        for c in cliques:
            elem = c[0]
            for n in graph[elem]:
                if self._connected_to_all(graph, n, c):
                    newclique = list(c)
                    newclique.append(n)
                    newcliques.add(tuple(sorted(newclique)))
        return newcliques

    def _connected_to_all(self, graph, node, clique):
        for n in clique:
            if node not in graph[n]:
                return False
        return True

class SFMSolver:
    def __init__(self, filenames, masks, **settings):
        self.filenames = filenames
        self.masks = masks
        self.detector = "surf"
        assert masks is None or len(filenames) == len(masks)

    def getMatches(self):
        print("-- load features --")
        files = self.filenames
        num = len(files)

        fl = FeatureLoader.FeatureLoader()
        ml = MatchLoader.MatchLoader()
        kpts = [fl.loadFeatures(f, self.detector) for f in files]

# masking
        kpts = self.maskKeypoints(kpts, num)

# match
        print("-- matching --")
        print("num imgs: %d" % num)
        matches = [[None] * num for i in range(num)]
        for i in range(num):
            print(i)
            for j in range(num):
                if i == j: continue

                MatchLoader.kpt1 = kpts[i][0]
                MatchLoader.kpt2 = kpts[j][0]
                matches[i][j] = ml.loadMatches(
                    files[i], files[j], kpts[i][1], kpts[j][1], self.detector, self.matcher, self.matcherVer)
        return matches, kpts

    def maskKeypoints(self, kpts):
        num = len(self.filenames)
        masks = self.masks
        if masks is not None:
            print("-- masking --")
            print(len(kpts[0][1]))
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
            print(len(kpts[0][1]))

    def getGraph(self, matches, kpts):
# graph
        print("-- graph --")
        num = len(self.filenames)
        graph = {}
        cnst = np.max([len(kpt_list[0]) for kpt_list in kpts])

        for i in range(num):
            for j in range(num):
                if i == j: continue
                for m in matches[i][j]:
                    id1 = (i, m.queryIdx)
                    id2 = (j, m.trainIdx)
                    if id1 not in graph:
                        graph[id1] = set()
                    graph[id1].add(id2)

#both ways
        print("graph size: %d" % len(graph))
        for k in graph.keys():
            for v in list(graph[k]):
                if v not in graph:
                    graph[v] = set()
                graph[v].add(k)
        print("graph size: %d" % len(graph))

#connectivity print
        print("-- connectivity --")
        total = 0
        i = 0
        while True:
            subg = [(k, graph[k]) for k in graph if len(graph[k]) == i]
            print(i, len(subg))
            total += len(subg)
            i += 1
            if total == len(graph):
                break
        return graph, cnst

    def extractCliques(self, graph, maxlevel = 5):
        print("levels")
        cliqueExtr = CliqueExtractor()
        all_levels = cliqueExtr.getCliques(graph, maxlevel)
        for i in range(len(all_levels)):
            level = all_levels[i]
            print(i + 3, len(level))

        return all_levels

    def getCliquePosRANSAC(self, clique, kpts, tmats, min_inliers = 3, err_thresh = 200):
        num = len(clique)
        pos = [[None] * num for n in clique]
        projMats = [np.dot(Utils.camMtx, tmat) for tmat in tmats]
        imgPts = [np.array(kpts[imgidx][0][kptidx].pt)
                        for imgidx, kptidx in clique]
        res = [[None] * num for n in clique]

        best = None
        besterr = 0
        for i in range(num):
            for j in range(i + 1, num):
                p4d = self.triangulate(
                    projMats[i], projMats[j], imgPts[i], imgPts[i])
                pos[i][j] = p4d
                #pos[j,i] = pos[i,j]

                res[i][j] = 0 #inliers
                inliers = []
                for k in range(num):
                    #reproj
                    repr = np.dot(projMats[k], p4d)
                    repr[0] /= repr[2]
                    repr[1] /= repr[2]
                    repr = repr[:2]

                    #err calc
                    diff = repr.T - imgPts[k]
                    err = np.linalg.norm(diff, 2)
                    # pprint(repr)
                    # pprint(imgPts[k])
                    # pprint(diff)
                    # print err
                    if err < err_thresh:
                        inliers.append(k)

                res[i][j] = inliers
                if best is None or len(inliers) > len(res[best[0]][best[1]]):
                    best = (i, j)

        inliers = res[best[0]][best[1]]
        if len(inliers) < min_inliers:
            return None

        sfmImgPoints = []
        sfmProjs = []
        for inl in inliers:
            sfmImgPoints.append(imgPts[inl])
            sfmProjs.append(projMats[inl])

        point =  self.solve_sfm(sfmImgPoints, sfmProjs)
        return point


                #todo: matching based on epipolar lines
                #todo: calculate p4d from all inliers (see sfm_test.py), store num inliers, error in res
                #todo: add checking of coordinate bounds to maybe class level? (eg. z is in [1, 3])
                #todo: test pose solving by running the finished algorithm on a pic with known pose.
                # asdasfsdg

    def solve_sfm(self, img_pts, projs):
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
        return x

    def triangulate(self, projMat1, projMat2, imgPts1, imgPts2):
        p4d = cv2.triangulatePoints(projMat1, projMat2, imgPts1, imgPts2)

        for i in range(0, p4d.shape[1]):
            p4d[0][i] /= p4d[3][i]
            p4d[1][i] /= p4d[3][i]
            p4d[2][i] /= p4d[3][i]
            p4d[3][i] /= p4d[3][i]

        return p4d

def draw(clique, imgs, kpts):
    m = clique
    print "-- draw --"
    print m
    for j in range(1, len(m)):
        img_idx1 = m[0][0]
        img_idx2 = m[j][0]
        kpt_idx1 = m[0][1]
        kpt_idx2 = m[j][1]
        print(img_idx1, img_idx2, kpt_idx1, kpt_idx2)

        img1 = imgs[img_idx1]
        img2 = imgs[img_idx2]

        pt1 = kpts[img_idx1][0][kpt_idx1].pt
        pt2 = kpts[img_idx2][0][kpt_idx2].pt
        Utils.drawMatch(img1, img2, pt1, pt2, scale=4)
        cv2.waitKey()
    cv2.waitKey()

def test():
    files = ["imgs/00%d.jpg" % (i) for i in range(5, 10)]
    imgs = [cv2.imread(f) for f in files]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    sfm = SFMSolver(files, masks)
    matches, kpts = sfm.getMatches()
    graph, cnst = sfm.getGraph(matches, kpts)
    all_levels = sfm.extractCliques(graph)
    tmats = [MarkerDetect.loadMat(f) for f in files]
    # print sfm.getCliquePosRANSAC(all_levels[1][0], kpts, tmats)
    points = []
    for c in all_levels[1]:
        point = sfm.getCliquePosRANSAC(c, kpts, tmats)
        if point is not None:
            points.append((c, point))
    for c, p in points:
        print p
        draw(c, imgs, kpts)
    return all_levels, cnst, graph

if __name__ == '__main__':
    test()