import cv2
import numpy as np
import time
import os
import sys

OUT_FOLDER = None

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def getNextFileIdx():
    i = 0
    while os.path.exists(os.path.join(OUT_FOLDER, str(i).rjust(4, '0') + ".jpg")):
        i += 1
    return i

def getFileName(idx):
    fname = os.path.join(OUT_FOLDER, str(idx).rjust(4, '0') + ".jpg")
    return fname

if __name__ == '__main__':
    if OUT_FOLDER is None:
        OUT_FOLDER = get_script_path() + "/out/"
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    fileIdx = getNextFileIdx()

    if not cap.isOpened():
        print "ERROR: webcam open failed"
    else:
        while cap.isOpened():
            r, frame = cap.read()
            if not r:
                continue
            cv2.imshow("frame", frame)
            # enter: 13, escape: 27
            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == 13:
                success = cv2.imwrite(getFileName(fileIdx), frame)
                if success:
                    print "Success. File saved: %s" %  getFileName(fileIdx)
                else:
                    print "Failed to write to: %s" %  getFileName(fileIdx)
                fileIdx += 1

