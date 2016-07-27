import numpy as np
import cv2

def rpy(mat):
    r = np.arctan2(mat[1,0], mat[0,0])
    if r<-np.pi: r+=np.pi
    if r>np.pi: r-=np.pi

    s1 = np.sin(r)
    c1 = np.cos(r)

    s2 = -mat[2,0]
    c2 = s1 * mat[1,0] + c1 * mat[0,0]
    p = np.arctan2(s2, c2)

    s3 = s1 * mat[0,2] - c1*mat[1,2]
    c3 = -s1*mat[0,1] + c1*mat[1,1]
    y = np.arctan2(s3, c3)

    return r, p, y

def getTransform(roll,  pitch,  yaw,  tx,  ty,  tz):
    s1 = np.sin(roll)
    c1 = np.cos(roll)
    s2 = np.sin(pitch)
    c2 = np.cos(pitch)
    s3 = np.sin(yaw)
    c3 = np.cos(yaw)

    return np.array([
        [c1*c2, c1*s2*s3-s1*c3, c1*s2*c3+s1*s3, tx],
        [s1*c2, s1*s2*s3+c1*c3, s1*s2*c3-c1*s3, ty],
        [-s2,   c2*s3,          c2*c3,          tz]
    ])

def dum(i):
    return i/10.0 * np.pi

def test():
    num = 0
    errno = 0
    for i in range(-10, 10):
        for j in range(-10, 10):
            for k in range(-10, 10):
                num += 1
                r, p, y = dum(i), dum(j), dum(k)
                t = getTransform(r, p, y, 0, 0, 0)
                rr, pp, yy = rpy(t)
                tt = getTransform(rr, pp, yy, 0, 0, 0)
                err = np.sum(np.abs(t-tt))
                if err > 0.01:
                    errno+=1
                    print(r, p, y, rr, pp, yy)
    print(num)
    print(errno)

if __name__ == "__main__":
    test()