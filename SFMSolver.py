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
        tmats = [MarkerDetect.loadMat(f) for f in files]

# masking
        kpts = self.maskKeypoints(kpts)

# match
        print("-- matching --")
        print("num imgs: %d" % num)
        matches = [[None] * num for i in range(num)]
        for i in range(num):
            print(i)
            for j in range(num):
                if i == j: continue

                matches[i][j] = ml.matchBFCrossEpilines(
                    self.filenames[i],
                    self.filenames[j],
                    kpts[i][1],
                    kpts[j][1],
                    kpts[i][0],
                    kpts[j][0],
                    tmats[i],
                    tmats[j],
                    "surf"
                )
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
        return kpts

    def getGraph(self, matches, kpts):
# graph
        print("-- graph --")
        num = len(self.filenames)
        graph = {}

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
        return graph

    def extractCliques(self, graph, maxlevel = 5):
        print("levels")
        cliqueExtr = CliqueExtractor()
        all_levels = cliqueExtr.getCliques(graph, maxlevel)
        for i in range(len(all_levels)):
            level = all_levels[i]
            print(i + 3, len(level))

        return all_levels

    def extendCliques(self, graph, cliques, max_missing = 1):
        for clique in cliques:
            clique = set(clique)
            num = len(clique)
            for node in clique:     #for each node
                missing = []
                num_connected = 0
                for k in list(graph[node]):   #for each neighbor k of node
                    if k in clique:
                        continue
                    missing = clique - graph[k]
                    if len(missing) <= max_missing:
                        #add missing edges to k's neigh list
                        graph[k].update(missing)
                        for n in clique:
                            graph[n].add(k)

    def print_graph(self, graph, node, depth):
        nodes = self._get_subgraph(graph, node, depth)
        for n in nodes:
            print n, graph[n]

    def _get_subgraph(self, graph, node, depth):
        if depth == 0:
            return [node]

        nodes = set()
        nodes.add(node)
        for neigh in graph[node]:
            nnodes = self._get_subgraph(graph, neigh, depth - 1)
            for n in nnodes:
                nodes.add(n)
        return list(nodes)

    def getCliquePosRANSAC(self, clique, kpts, tmats, min_inliers=3, err_thresh=200):
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
                p4d = self._triangulate(
                    projMats[i], projMats[j], imgPts[i], imgPts[i])
                pos[i][j] = p4d
                # pos[j,i] = pos[i,j]

                res[i][j] = 0  # inliers
                inliers = []
                for k in range(num):
                    # reproj
                    repr = np.dot(projMats[k], p4d)
                    repr[0] /= repr[2]
                    repr[1] /= repr[2]
                    repr = repr[:2]

                    # err calc
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
            return None, None

        sfmImgPoints = []
        sfmProjs = []
        for inl in inliers:
            sfmImgPoints.append(imgPts[inl])
            sfmProjs.append(projMats[inl])

        point = self.solve_sfm(sfmImgPoints, sfmProjs)
        return point, inliers

        # todo: manually add some inliers to ransac solve
        # todo: speed up epipole-based matching (c++ maybeh?)
        # todo: add checking of coordinate bounds to maybe class level? (eg. z is in [1, 3])
        # todo: test pose solving by running the finished algorithm on a pic with known pose.

    def getCliquePosSimple(self, clique, kpts, tmats, avg_err_thresh=20, max_err_thresh = 30):
        num = len(clique)
        projMats = [np.dot(Utils.camMtx, tmat) for tmat in tmats]
        imgPts = [np.array(kpts[imgidx][0][kptidx].pt)
                  for imgidx, kptidx in clique]

        point = self.solve_sfm(imgPts, projMats)

        #reproj
        point_h = np.ones((4, 1), dtype=np.float32)
        point_h[:3,:] = point
        repr_pts = [projMat.dot(point_h).reshape(3) for projMat in projMats]
        errs = np.zeros((num,), dtype = np.float32)
        for i in range(num):
            repr_pt = repr_pts[i]
            repr_pt /= repr_pt[2]
            img_pt = imgPts[i]
            repr_pt = repr_pt[:2]
            err = np.linalg.norm(repr_pt - img_pt)
            errs[i] = err

        avg_err = np.average(errs)
        max_err = np.max(errs)
        if avg_err < avg_err_thresh and max_err < max_err_thresh:
            return point, avg_err, max_err
        else:
            return None, None, None

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

    def _triangulate(self, projMat1, projMat2, imgPts1, imgPts2):
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
    c = None
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
        c = cv2.waitKey()
    return c

def calc_repr_err(c, p, inl, tmats, kpts):
    errs = [0] * len(inl)
    p4d = np.ones((4,1))
    p4d[:3,:] = p
    for i in range(len(inl)):
        inlier = inl[i]
        img_idx, kpt_idx = c[inlier]
        kpt = np.array(kpts[img_idx][0][kpt_idx].pt, dtype=np.float32)

        tmat = tmats[img_idx]
        cmat = Utils.camMtx
        reproj = cmat.dot(tmat).dot(p4d)
        reproj /= reproj[2, 0]
        errs[i] = np.linalg.norm(reproj.T[0,:2] - kpt)
        print errs[i]
    max_err = max(errs)
    avg_err = np.average(np.array(errs))
    print max_err, avg_err

def match_to_img(file, data):
    img = cv2.imread(file)
    data_des, data_pts = zip(*data)

    fl = FeatureLoader.FeatureLoader()
    kp, des = fl.loadFeatures(file, "surf")
    ml = MatchLoader.MatchLoader()
    matches = ml.matchBFCross(file, "asd/nope.avi", des, data_des, "surf", nosave=True, noload=True)
    img_pts = []
    obj_pts = []
    for m in matches:
        img_pts.append(kp[m.queryIdx].pt)
        obj_pts.append(data_pts[m.trainIdx])

    rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(obj_pts, np.float32),
        np.asarray(img_pts, np.float32),
        Utils.camMtx,
        None,
        reprojectionError=10)
    rmat = cv2.Rodrigues(rvec)[0]

    tmat_load = MarkerDetect.loadMat(file, False)
    tmat = np.zeros((3,4))
    tmat[:3,:3] = rmat
    tmat[:3,3] = tvec.T
    tmat4x4 = np.eye(4)
    tmat4x4[:3,:] = tmat
    print "num data, img pts", len(data_des), len(des)
    print "num matches:", len(matches)
    print "num inliers: ", len(inliers)
    print "rmat", rmat
    print "tvec", tvec
    print "tmat load", tmat_load
    print "combined trf diff", tmat_load.dot(np.linalg.inv(tmat4x4))


def test():
    files = ["imgs/00%d.jpg" % (i) for i in range(5, 10)]
    imgs, kpts, points, data = calc_data_from_files(files)

    match_to_img("imgs/003.jpg", data)
    exit()

    print "num points: ", len(points)
    for c, p, a, m in points:
        print "--- new clique ---"
        print p
        print "error (avg, max): ", a, m
        if p[2] > -1.5:
            if draw(c, imgs, kpts) == 27:
                return

# pointData is list of tuple: (des, p3d)
def calc_data_from_files(files):
    imgs = [cv2.imread(f) for f in files]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    sfm = SFMSolver(files, masks)
    matches, kpts = sfm.getMatches()

    import DataCache as DC
    data = DC.getData(DC.POINTS4D)
    if data is None:
        graph = sfm.getGraph(matches, kpts)
        all_levels = sfm.extractCliques(graph, maxlevel=3)
        sfm.extendCliques(graph, all_levels[0], 1)
        all_levels = sfm.extractCliques(graph, maxlevel=3)
        # sfm.extendCliques(graph, all_levels[0], 1)
        # all_levels = sfm.extractCliques(graph)
        tmats = [MarkerDetect.loadMat(f) for f in files]
        points = []

        # for c in all_levels[0]:
        #     point, inliers = sfm.getCliquePosRANSAC(c, kpts, tmats, err_thresh=100)
        #     if point is not None:
        #         points.append((c, point, inliers))
        # print "num points: ", len(points)
        # for c, p, inl in points:
        #     print "--- new clique ---"
        #     # print p
        #     # calc_repr_err(c, p, inl, tmats, kpts)
        #     if draw(c, imgs, kpts) == 27:
        #         return

        for i in range(len(all_levels[0])):
            if i % 1000 == 0: print i, len(all_levels[0]), len(points)
            c = all_levels[0][i]
            point, avg_err, max_err = sfm.getCliquePosSimple(c, kpts, tmats, avg_err_thresh=5, max_err_thresh=10)
            if point is not None:
                points.append((c, point, avg_err, max_err))

        pointData = []
        for c, p, a, m in points:
            for node in c:
                img_idx, kpt_idx = node
                pointData.append((kpts[img_idx][1][kpt_idx], p))

        DC.saveData(DC.POINTS4D, (points, pointData))
    else:
        points, pointData = data

    return imgs, kpts, points, pointData

if __name__ == '__main__':
    test()
    exit()

    import cProfile
    cProfile.run("test()")