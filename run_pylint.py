#!/usr/bin/env python
"""
This script runs pylint on this project
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--files",nargs="+")
parser.add_argument("--patfile", type=argparse.FileType("r"))
args = parser.parse_args()


import subprocess, os.path as osp, os, fnmatch
from glob import glob

def cap(cmd):
    "call and print"
    print "\x1b[32m%s\x1b[0m"%cmd
    subprocess.call(cmd,shell=True)

def recursive_glob(basedir):
    filepat = "*.py"
    matches = []
    for root, _dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, filepat):
            matches.append(os.path.join(root, filename))   
    return matches 

def filelist_from_patterns(pats):
    # filelist = []
    fileset = set([])
    lines = [line.strip() for line in pats]
    for line in lines:
        line = line.strip()
        pat  = line[2:]
        if line.startswith("#"):
            continue
        newfiles = recursive_glob(pat) if osp.isdir(pat) else [pat]
        if line.startswith("+"):
            fileset.update(newfiles)
        elif line.startswith("-"):
            fileset.difference_update(newfiles)
        else:
            raise ValueError("line must start with + or -, got %s"%line)
    filelist = list(fileset)
    return filelist


def main():
    if args.files is not None:
        filelist = args.files
    elif args.patfile is not None:
        filelist = filelist_from_patterns(args.patfile.readlines())
    elif osp.exists("lintfiles.txt"):
        with open("lintfiles.txt","r") as fh:
            filelist = filelist_from_patterns(fh.readlines())
    else:
        filelist = glob("*.py")

    rcfile = osp.abspath(osp.join(osp.dirname(__file__), "pylintrc"))
    assert osp.exists(rcfile)
    lint = "pylint"
    if filelist is not None:
        for fname in filelist:
            cap("%s -f colorized --rcfile %s -r n %s"%(lint, rcfile, fname))
    else:
        cap("%s -f colorized  --rcfile %s -r n  *.py"%(lint,rcfile))


if __name__ == "__main__":
    main()