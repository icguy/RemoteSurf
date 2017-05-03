import cv2
import numpy as np
import FeatureLoader
import MatchLoader
import Utils
import cProfile

def extract_full(graph, max_num):
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
        cliques = add_level(graph, cliques)
        all_levels.append(list(cliques))
        num += 1
    return all_levels

def add_level(graph, cliques):
    newcliques = set()
    for c in cliques:
        elem = c[0]
        for n in graph[elem]:
            if connected_to_all(graph, n, c):
                newclique = list(c)
                newclique.append(n)
                newcliques.add(tuple(sorted(newclique)))
    return newcliques

def connected_to_all(graph, node, clique):
    for n in clique:
        if node not in graph[n]:
            return False
    return True


def __find_object():
    import DataCache as DC
    from glob import glob
    from os.path import join
    import numpy as np
    from SFMSolver import SFMSolver, find_ext_params
    from pprint import pprint
    import DataCache as DC

    print "FINDING"

    np.set_printoptions(precision=3, suppress=True)

    SFM_files_dir = "out/2017_3_8__14_51_22/"
    SFM_files = glob(join(SFM_files_dir, "*.jpg"))
    masks = []
    for f in SFM_files:
        m = f.replace(".jpg", "_mask.png")
        masks.append(m)
    sfm = SFMSolver(SFM_files, masks)

    imgs, kpts, points, data = sfm.calc_data_from_files_triang_simple()

    arr_calib = DC.getData("out/%s/arrangement_calib.p" % "2017_4_28__13_51_30")
    ttc = arr_calib["ttc"]
    tor = arr_calib["tor"]

    find_dir = "out/2017_5_3__13_17_25"
    # find_dir = "out/2017_5_3__12_53_1"
    files = glob("%s/*.jpg" % find_dir)
    print files
    # files_dir = "out/2017_4_5__15_57_20/"
    # files = glob(join(files_dir, "*.jpg"))
    files.sort()
    offset = 0
    files = files[offset:offset+5]
    results = []

    for f in files:
        # datafile = "cache/%s.p" % str(f).replace("\\", "/").replace("/", "_")
        # res = DC.getData(datafile)
        # if res is None:
        #     res = find_ext_params(f, imgs, kpts, points, data, tor, ttc)
        #     DC.saveData(datafile, res)

        res = find_ext_params(f, imgs, kpts, points, data, tor, ttc, False, False, False)
        results.append(res)

    for i in range(len(results)):
        print i, results[i]
    result = max(results, key=lambda x: x[2])
    print result
    values = {
        500: int(result[0][0] * 10),
        501: int(result[0][1] * 10),
        502: int(result[0][2] * 10) + 200,
        503: int(result[1][2]),
        504: int(result[1][1]),
        505: int(result[1][0]),
    }

    print "num inl: ", result[2]
    pprint(values)


if __name__ == '__main__':
    # a = (1, 2, 3)
    # b = (4, 5, 6)
    # c = [aa - bb for aa, bb in zip(a, b)]
    # print c

    __find_object()