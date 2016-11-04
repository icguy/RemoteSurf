
outfile = None

def write_log(text):
    if outfile is None: return

    f = file(outfile, "a")
    f.write(text)
    f.write("\r\n")
    f.close()
