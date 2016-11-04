import cv2
import numpy as np
import time
import os
import sys

OUT_FOLDER = None

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def getNextFileIdx(out_folder):
    i = 0
    while os.path.exists(os.path.join(out_folder, str(i).rjust(4, '0') + ".jpg")):
        i += 1
    return i

def getFileName(out_folder, idx):
    fname = os.path.join(out_folder, str(idx).rjust(4, '0') + ".jpg")
    return fname

def run(out_folder):
    if out_folder is None:
        out_folder = get_script_path() + "/out/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    fileIdx = getNextFileIdx(out_folder)
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
                success = cv2.imwrite(getFileName(out_folder, fileIdx), frame)
                if success:
                    print "Success. File saved: %s" % getFileName(out_folder, fileIdx)
                else:
                    print "Failed to write to: %s" % getFileName(out_folder, fileIdx)
                fileIdx += 1

if __name__ == '__main__':
    run(OUT_FOLDER)

