
outfile = None
PRINT_LOG_TO_STDOUT = True

def write_log(text):
    if PRINT_LOG_TO_STDOUT: print text
    if outfile is None: return

    f = file(outfile, "a")
    f.write(text)
    f.write("\r\n")
    f.close()
