import cv2
import numpy as np
from random import randrange

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    scale = 4

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    i = 0
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        i = i + 1
        alpha = int(255 * (i * 1.0 / len(matches)))
        color = (randrange(0, 255),randrange(0, 255),randrange(0, 255))

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4 * scale, color, scale)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4 * scale, color, scale)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, scale)


    out = cv2.pyrDown(out)
    out = cv2.pyrDown(out)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def match(file1, file2, draw = False, maxdraw = None, reverse = True, thresh = 0.7):

    print "......"
    print file1
    print file2

    img1 = cv2.imread(file1,0) # queryImage
    img2 = cv2.imread(file2,0) # trainImage

    # Initiate SIFT detector
    detector = cv2.SIFT()
    detector = cv2.SURF()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)
    print(len(des1))
    print(len(des2))

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < thresh * n.distance:
            good.append(m)

    good.sort(key = lambda x: x.distance, reverse = reverse)

    print "num matches: ", len(matches)
    print "num good matches: ", len(good)
    if draw:
        if maxdraw is None: maxdraw = len(good)
        img3 = drawMatches(img1,kp1,img2,kp2,good[:maxdraw])

    return good, kp1, kp2

def matchORB(file1, file2, draw = False, maxdraw = None):
    print "......"
    print file1
    print file2

    MIN_MATCH_COUNT = 10

    img1 = cv2.imread(file1,0)          # queryImage
    img2 = cv2.imread(file2,0) # trainImage

    # Initiate SIFT detector
    detector = cv2.ORB(WTA_K=4)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # store all the good matches as per Lowe's ratio test.
    good = matches

    good.sort(key = lambda x: x.distance, reverse=True)

    print "num matches: ", len(good)
    if draw:
        if maxdraw is None: maxdraw = len(good)
        img3 = drawMatches(img1,kp1,img2,kp2,good[:maxdraw])

    return len(good)

def main():
    file1 = "imgs/004.jpg"
    file2 = "imgs/005.jpg"
    good, kp1, kp2 = match(file1, file2, True)
    print([(g.trainIdx, g.queryIdx) for g in good])
    print("end")


if __name__ == "__main__":
    main()
    exit()
    import cProfile
    print(cProfile.run("main()"))