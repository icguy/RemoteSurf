import numpy as np

def getTrf(angle, axis):
    angle *= np.pi / 180
    s, c = np.sin(angle), np.cos(angle)
    if axis == 'x':
        return np.array([1, 0, 0, 0, c, -s, 0, s, c], dtype=np.float32).reshape((3,3))
    if axis == 'y':
        return np.array([c, 0, s, 0, 1, 0, -s, 0, c], dtype=np.float32).reshape((3,3))
    if axis == 'z':
        return np.array([c, -s, 0, s, c, 0, 0, 0, 1], dtype=np.float32).reshape((3,3))
    return None

a = 90
b = 90
c = 90

v1 =
