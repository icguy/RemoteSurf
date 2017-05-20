from fileinput import filename

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

data1 = [(26.574, -2.466, 2.796, 172.95, -27.46, 162.4, 189),
         (30.336, 0.098, 0.266, -179.673, 0.226, 158.38, 157),
         (29.991, -1.417, 0.823, -179.524, 2.95, 174.088, 76),
         (26.48, -2.457, 2.933, 173.111, -27.105, 161.189, 183),
         (30.953, -0.756, 6.51, -178.271, 41.878, 164.869, 130),
         (31.175, -1.095, 6.814, -179.399, 31.888, 176.378, 152),
         (30.469, -1.467, 2.651, -179.627, -10.134, 179.929, 101),
         (308, -14, 249, 180, 0, 180)]

data2 = [(28.136, -4.519, 10.646, 106.25, -9.377, 145.319, 116),
(29.736, -2.86, 1.069, 99.441, -27.925, 175.934, 251),
(25.659, -5.409, 1.74, 98.545, 12.364, 160.735, 187),
(29.896, -2.852, 1.14, 99.504, -27.796, 177.206, 274),
(29.759, -4.353, 1.38, 98.259, -1.021, 171.354, 248),
(29.871, -2.489, 2.011, 99.397, -23.122, 175.709, 212),
(30.007, -4.377, 1.577, 98.121, -2.763, 166.26, 209),
(299, -51, 253, -180, 0, 100)]

data3 = [(30.964, -7.602, 12.958, 149.857, -10.951, 140.641, 61),
(34.008, -4.263, 3.313, 150.877, -1.531, 163.052, 115),
(33.669, -3.97, 10.04, 157.411, 12.741, 147.545, 71),
(31.319, -4.082, 11.978, 156.922, 4.727, 104.222, 29),
(34.292, -3.574, 1.021, 151.385, -3.774, 169.826, 100),
(34.432, -3.95, 2.417, 153.066, -10.457, 170.509, 190),
(356, -46, 265, 179, -14, 155)]

data4 = [(30.632,-0.304,-0.335,170.826,-34.773,-135.698,46),
(33.474,0.396,3.652,-178.541,15.818,-142.559,50),
(26.574,-2.466,-2.796,172.95,-27.46,-162.4,189),
(39.065,9.627,-27.112,-92.524,-62.719,122.855,8),
(30.097,-2.03,10.256,-176.964,-22.522,-136.128,24),
(29.966,1.515,4.656,-174.9,19.662,-82.133,19),
(30.531,-0.937,7.044,-179.329,42.456,-171.822,110),
(26.524,-2.863,-3.272,173.34,-26.41,-169.056,104),
(30.336,0.098,0.266,-179.673,0.226,158.38,157),
(29.991,-1.417,0.823,-179.524,2.95,174.088,76),
(29.931,-1.519,1.43,171.786,-6.779,-170.713,39),
(31.339,-0.233,6.077,-174.77,39.216,-142.146,35),
(25.189,0.072,7.471,-172.007,22.927,-170.822,27),
(26.915,-1.493,6.616,168.139,40.247,158.368,12),
(30.278,0.516,-4.034,-174.924,6.376,-76.776,27),
(28.324,0.244,0.217,116.573,-22.277,-159.219,71),
(30.914,-0.932,6.822,-179.666,37.535,-172.339,145),
(26.48,-2.457,-2.933,173.111,-27.105,-161.189,183),
(31.284,-0.435,-2.661,-177.392,-14.029,123.768,24),
(31.016,-3.06,10.788,-175.234,-29.764,-170.004,43),
(29.549,-0.936,-0.47,-177.12,7.587,-166.557,84),
(30.953,-0.756,6.51,-178.271,41.878,-164.869,130),
(25.049,-2.188,0.069,-174.792,12.429,-178.479,60),
(29.783,-0.596,5.917,172.298,7.414,-175.402,15),
(30.283,0.166,-2.502,-175.312,3.065,-143.805,28),
(30.171,0.372,1.637,-179.255,-5.259,-133.195,56),
(31.175,-1.095,6.814,-179.399,31.888,-176.378,152),
(26.537,-2.818,-2.331,171.947,-31.953,-176.55,113),
(29.912,-1.049,4.987,177.265,13.36,158.886,43),
(30.469,-1.467,2.651,-179.627,-10.134,179.929,101),
(31.773,-0.919,8.007,155.875,-62.085,-138.796,11),
(29.4,-2.053,11.278,-178.145,-8.024,152.455,26),
(26.772,-1.837,-0.962,171.422,-33.284,-162.331,49),
(29.348,-1.838,3.69,174.24,19.837,121.925,47),
         (308, -14, 249, 180, 0, 180)]

data5 = [(29.538,-4.538,5.435,95.342,-20.242,165.957,54),
(28.136,-4.519,10.646,106.25,-9.377,145.319,116),
(26.96,-5.438,-2.142,96.027,9.034,-120.704,57),
(28.823,-5.231,7.955,97.771,-38.581,153.602,36),
(27.495,-4.8,-1.236,101.659,-5.245,106.372,50),
(27.833,-3.635,3.879,104.701,14.785,143.917,31),
(30.693,-4.378,6.575,97.041,3.475,149.862,245),
(25.649,-5.266,-1.649,99.009,14.189,-159.481,155),
(29.736,-2.86,-1.069,99.441,-27.925,175.934,251),
(27.891,-5.857,17.403,97.594,40.811,133.166,25),
(31.695,-4.368,-0.22,89.275,-4.736,-98.9,59),
(30.808,-6.378,8.112,144.104,-48.969,144.632,7),
(25.659,-5.409,-1.74,98.545,12.364,-160.735,187),
(29.69,0.565,-5.362,174.551,-81.571,92.987,19),
(28.76,-4.415,0.255,97.968,5.104,150.972,88),
(29.955,-4.368,1.549,98.027,-2.603,-166.387,234),
(30.528,-4.315,6.613,95.954,4.998,143.933,216),
(25.705,-5.2,-1.594,99.934,16.182,-161.91,107),
(29.896,-2.852,-1.14,99.504,-27.796,177.206,274),
(29.181,-5.592,-0.086,96.876,-38.135,159.042,44),
(29.759,-4.353,1.38,98.259,-1.021,-171.354,248),
(28.716,-3.298,10.994,87.887,17.175,164.405,55),
(27.061,-5.753,-3.52,95.023,9.301,-95.622,53),
(32.621,-1.524,-2.088,104.895,-11.163,-87.394,38),
(31.328,-4.633,-1.692,93.484,-15.178,-145.265,37),
(29.82,-4.246,0.772,99.112,2.571,-166.624,180),
(25.134,-4.508,9.945,105.925,-23.406,59.213,13),
(24.905,-7.005,-10.09,100.471,27.33,-159.688,31),
(29.871,-2.489,-2.011,99.397,-23.122,175.709,212),
(30.291,-4.374,-0.16,98.191,-4.892,-164.747,151),
(30.007,-4.377,1.577,98.121,-2.763,-166.26,209),
(30.45,-4.391,6.568,96.714,4.299,140.12,199),
(22.98,-4.796,-1.542,95.628,32.476,127.367,12),
(29.604,-2.71,-1.423,103.81,-50.084,154.112,57),
(299, -51, 253, -180, 0, 100)]

data6 = [
(33.544,-4.612,4.799,142.904,-23.504,129.333,17),
(22.794,-1.462,27.878,166.442,-33.632,-170.768,13),
(30.964,-7.602,-12.958,149.857,-10.951,-140.641,61),
(32.716,-2.641,5.409,142.685,33.898,166.629,12),
(34.008,-4.263,3.313,150.877,-1.531,163.052,115),
(34.574,-3.292,1.115,151.619,-35.376,-170.658,58),
(28.771,-3.882,16.134,156.274,14.296,144.669,22),
(31.213,-3.044,5.026,137.459,-37.832,-163.616,46),
(34.896,-0.183,-3.268,156.539,-4.981,-137.686,38),
(32.377,-6.803,11.606,167.641,-62.956,119.31,13),
(33.786,-5.027,1.17,152.186,-12.869,142.152,47),
(33.669,-3.97,10.04,157.411,12.741,147.545,71),
(31.519,-2.607,2.735,132.773,-44.161,-128.569,52),
(32.869,-4.557,3.753,146.697,23.346,99.692,25),
(28.026,-7.309,25.775,91.701,54.033,152.401,5),
(34.377,-5.163,7.504,108.146,-12.571,-84.8,16),
(31.319,-4.082,11.978,156.922,4.727,104.222,29),
(30.457,-4.909,4.228,151.791,-26.65,133.724,7),
(35.178,-4.771,12.175,176.06,-51.437,-174.239,29),
(34.292,-3.574,1.021,151.385,-3.774,169.826,100),
(36.087,1.624,31.982,179.686,66.993,-78.5,7),
(25.731,-3.192,24.007,160.505,-25.205,-169.916,8),
(31.251,-5.763,-6.155,140.558,-23.286,-142.064,87),
(34.229,-4.661,5.481,165.346,-30.337,134.733,77),
(34.432,-3.95,2.417,153.066,-10.457,170.509,190),
(32.853,-3.742,-0.185,145.307,-49.178,174.28,32),
(33.815,-2.638,10.682,150.392,23.081,-169.299,109),
(30.901,-4.133,0.868,137.695,-31.382,-161.707,59),
(34.8,-2.713,0.7,159.583,-10.885,165.642,133),
         (356, -46, 265, 179, -14, 155)]

def rotdiff(A, B):
    # print A, B
    return np.rad2deg((np.arccos(A.T.dot(B))))
    return (3 - np.trace(A.T.dot(B))) / 6

def getdatapts(data):
    goodpos = np.array(data[-1][:3]) / 10.0
    goodrot = map(np.deg2rad, data[-1][3:])
    data = data[:-1]
    err_all = []
    errot_all = []
    inliers_all = []
    for i in range(len(data)):
        endpos = np.array(data[i][:3])
        endpos[2] = endpos[2] + 20
        endrot = map(np.deg2rad, data[i][3:6])
        inliers = data[i][-1]
        err = np.sqrt(np.sum(np.square(goodpos - endpos)))
        goodrpy = Utils.getTransform(goodrot[2], goodrot[1], goodrot[0], 0, 0, 0)[:3, :3]
        endrpy = Utils.getTransform(endrot[0], endrot[1], endrot[2], 0, 0, 0)[:3, :3]
        # print goodrpy
        # print endrpy
        errot_mat = np.diag(rotdiff(goodrpy, endrpy))
        errot = np.average(errot_mat)
        err_coord = goodpos - endpos
        # print goodrpy, endrpy
        print "%s, %.2f, %s, %.2f, %d" % (str(err_coord), (err), str(errot_mat), (errot), inliers)
        err_all.append(err)
        errot_all.append(errot)
        inliers_all.append(inliers)
    return err_all, errot_all, inliers_all

def procdata(data, filename = None):
    global err, errot, inliers

    err, errot, inliers = getdatapts(data)

    plt.figure(figsize=(9, 4))
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}
    plt.rc('font', **font)

    sub1 = plt.subplot(121)
    sub1.scatter(inliers, err)
    sub1.set_ylabel("transzlacios hiba (cm)")
    sub1.set_xlabel("inlierek szama")
    sub1.set_ylim(ymin = 0, ymax = 40)
    sub1.set_xlim(xmin = 0, xmax = 300)

    # plt.title("err")
    sub2 = plt.subplot(122)
    sub2.scatter(inliers, errot)
    sub2.set_ylabel("atlagos orientacios hiba (fok)")
    sub2.set_xlabel("inlierek szama")
    sub2.set_ylim(ymin = 0, ymax=120)
    sub2.set_xlim(xmin = 0, xmax = 300)

    plt.subplots_adjust(
        left = 0.11,
        bottom = 0.17,
        right = 0.95,
        top = 0.92,
        wspace = 0.31
    )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename+".eps", format = "eps")
        plt.savefig(filename+".png", format = "png")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.interactive(False)
    np.set_printoptions(precision=2, suppress=True)

    A = Utils.getTransform(1, 2, 3, 0, 0, 0)[:3, :3]
    print rotdiff(A, A)
    print rotdiff(A, -A)

    print "data4"
    procdata(data4, "err1")
    print "data5"
    procdata(data5, "err2")
    print "data6"
    procdata(data6, "err3")

    print "data1"
    procdata(data1)
    print "data2"
    procdata(data2)
    print "data3"
    procdata(data3)
    print "hai"
    # __find_object()