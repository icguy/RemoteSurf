import numpy as np
from glob import glob
import pickle

np.set_printoptions(precision=3, suppress=True)

def rotx(angle):
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([1, 0, 0, 0, c, -s, 0, s, c]).reshape((3, 3))

def roty(angle):
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([c, 0, s, 0, 1, 0, -s, 0, c]).reshape((3, 3))

def rotz(angle):
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([c, -s, 0, s, c, 0, 0, 0, 1]).reshape((3, 3))

def get_rot(A, B, C):
    ra = rotx(A)
    rb = roty(B)
    rc = rotz(C)
    trf = rc.dot(rb.dot(ra))
    return trf

A = -135
B = 0
C = 90
A, B, C = map(lambda c: c * np.pi / 180, (A, B, C))

trf = get_rot(A, B, C)
import Utils
print trf
print Utils.getTransform(C, B, A, 0, 0, 0)
print trf - Utils.getTransform(C, B, A, 0, 0, 0)[:3, :3]

for i in range(1000):
    abc = np.random.random((3, ))
    abc = (abc - 0.5) * 2 * np.pi * 2
    a, b, c = abc

    trf1 = get_rot(A, B, C)
    trf2 = Utils.getTransform(C, B, A, 0, 0, 0)[:3, :3]
    delta = trf1- trf2
    if np.linalg.norm(delta, np.inf) > 0.00001:
        print "haho"
    

for f in glob("C:/Users/user/Documents/AD/RemoteSurf/Python/RemoteSurf/out/2016_11_15__16_47_52/*.p"):
    print f
    ff = file(f)
    o = pickle.load(ff)
    ff.close()
    d1, d2 = o
    x, y, z, a, b, c = d1.values()
    a, b, c = map(lambda c: c * np.pi / 180, (a, b, c))
    trf1 = np.zeros((3, 4))
    trf1[:3, :3] = get_rot(a, b, c)
    trf1[:3, 3] = np.array([x, y, z]).T

    x, y, z, a, b, c = d2.values()
    a, b, c = map(lambda c: c * np.pi / 180, (a, b, c))
    trf2 = np.zeros((3, 4))
    trf2[:3, :3] = get_rot(a, b, c)
    trf2[:3, 3] = np.array([x, y, z]).T

    print d1.values()
    print d2.values()
    print np.linalg.norm(trf1 - trf2, np.inf)
    # raw_input()

