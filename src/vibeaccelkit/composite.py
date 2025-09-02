import numpy as np
from typing import Dict, Iterable, Tuple, Union

def fds_envelope(curves: dict[str, np.ndarray]) -> np.ndarray:
    """
    Return the pointwise maximum (envelope) across FDS curves.
    Use when modes are mutually exclusive in time (parallel case).
    Durations are ignored.
    """
    if not curves:
        raise ValueError("No FDS curves provided")
    X = np.vstack([np.asarray(v, float) for v in curves.values()])
    return np.nanmax(X, axis=0)

def combine_fds(freqs, fds_list, durations):
    """
    Σ_i fds_i(f) * T_i  (time-weighted composite FDS).
    """
    out = np.zeros_like(freqs, dtype=float)
    for fds, T in zip(fds_list, durations):
        out += fds * float(T)
    return out
