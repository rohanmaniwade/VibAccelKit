import numpy as np
from scipy.io import loadmat

def load_mat_timesignal(path: str):
    """
    Loads a .mat with a single 2-column array [time, signal].
    Returns (time, signal, fs).
    """
    mat = loadmat(path)
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    arr = mat[key]
    time = arr[:, 0].astype(float)
    sig  = arr[:, 1].astype(float)
    fs = 1.0 / (time[1] - time[0])
    return time, sig, fs
