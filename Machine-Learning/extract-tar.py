import keras
import numpy as np
import os

fname = "/Users/bk/downloads/aclImdb_v1.tar.gz"
import tarfile
if (fname.endswith("tar.gz")):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif (fname.endswith("tar")):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()
