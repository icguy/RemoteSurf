import cv2
import numpy as np
import FeatureLoader
import MatchLoader
import Utils

files = ["imgs/00%d.jpg" % (i) for i in range(5, 10)]
num = len(files)

cst = 100 * 1000
imgs = [cv2.imread(f) for f in files]
fl = FeatureLoader.FeatureLoader()
ml = MatchLoader.MatchLoader()
kpts = [fl.loadFeatures(f, "surf") for f in files]
matches = [[None] * num for i in range(num)]
for i in range(num):
    for j in range(num):
        if i == j: continue
        print(i,j)
        matches[i][j] = ml.matchBFRatio(files[i], files[j], kpts[i][1], kpts[j][1], "surf", version="07")
