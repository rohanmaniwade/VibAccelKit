import numpy as np

def combine_fds(freqs, fds_list, durations):
    """
    Σ_i fds_i(f) * T_i  (time-weighted composite FDS).
    """
    out = np.zeros_like(freqs, dtype=float)
    for fds, T in zip(fds_list, durations):
        out += fds * float(T)
    return out
