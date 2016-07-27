import cv2
import numpy as np

camMtx = np.array([[2241.45, 0., 1295.5],
                   [0., 2241.45, 727.5,],
                   [0., 0., 1.]])

size = 14.1
sh = size / 2
objPtMarker = np.array([[-sh, -sh, 0],
                     [sh, -sh, 0],
                     [sh, sh, 0],
                     [-sh, sh, 0]], dtype=np.float32).T

def getObjPtMarkerHomogeneous():
    pts = np.ones((4,4))
    pts[:3,:] = objPtMarker
    return pts

def test_reproj(imgpts, objpts, tmat, cmat):
    numpts =  objpts.shape[0]
    proj = np.dot(cmat, tmat)

    c3d = np.zeros((4, numpts))
    c3d[:3,:] = objpts.T
    c3d[3,:] = np.ones((1, numpts))

    reproj = np.dot(proj, c3d)
    for i in range(numpts):
        w = reproj[2, i]
        for j in range(3):
            reproj[j, i] /= w

    errs = np.abs(imgpts.T - reproj[:2,:])
    max_err = np.max(errs)
    avg_err = np.average(errs)
    if(max_err > 20):
        print "WARNING! ---------------------------------------------------"
    print("max_err: ", max_err)
    print("avg_err: ", avg_err)

def drawMatch(img1, img2, pt1, pt2, good = True, scale = 2):
        realscale = 2
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        if scale == 4:
            realscale = 4
            img1 = cv2.pyrDown(img1)
            img2 = cv2.pyrDown(img2)

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

if __name__ == '__main__':
    print getObjPtMarkerHomogeneous()