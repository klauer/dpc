import os
import sys
import tempfile


def main():
    filename = sys.argv[1]
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        for line in open(filename):
            line = line.rstrip()
            fh.write(line + '\n')
        os.rename(filename, filename + '.bak')
        os.rename(fh.name, filename)


if __name__ == '__main__':
    main()
