import cPickle as pickle
from os.path import isfile

POINTS4D = "cache/points4d.p"
POINTS4D_MULTIPLE_MATCH = "cache/points4d_multiple.p"

def getData(filename):
    if isfile(filename):
        f = open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data
    return None

def saveData(filename, data):
    f = open(filename, "wb")
    pickle.dump(data, f, 2)
    f.close()
