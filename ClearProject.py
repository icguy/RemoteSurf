import os
import shutil

out_path = "out"
dirs = os.listdir(out_path)
for dir in dirs:
    curdir = os.path.join(out_path, dir)
    if not os.path.isdir(curdir):
        continue

    contents = os.listdir(curdir)
    if len(contents) == 0 or (len(contents) == 1 and contents[0] == "out.txt"):
        shutil.rmtree(curdir)
        print curdir
