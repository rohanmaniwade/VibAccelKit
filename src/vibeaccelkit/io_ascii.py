# vibeaccelkit/io_ascii.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple

def load_ascii_timesignal(
    path: str | Path,
    comments: Iterable[str] = ("#", "%", ";", "!", "//"),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load a simple ASCII time history with the format:
      col0 = time [s], cols 1..N = acceleration [m/s^2] (channels)

    Returns
    -------
    t : (n,) float64
        Time vector in seconds.
    A : (n, m) float64
        Acceleration channels (m/s^2).
    fs : float
        Sampling rate [Hz], computed from median dt.
    """
    path = Path(path)
    arr = np.loadtxt(path, comments=tuple(comments))
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path.name}: expected at least 2 columns (time + channels). Got shape {arr.shape}.")

    t = arr[:, 0].astype(float)
    A = arr[:, 1:].astype(float)

    dt = np.diff(t)
    if not np.all(dt > 0):
        raise ValueError(f"{path.name}: non-monotonic time column.")
    fs = float(1.0 / np.median(dt))

    return t, A, fs


def trim_by_time(t: np.ndarray, X: np.ndarray, t0: float | None, t1: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim time series to [t0, t1] (inclusive on both ends).
    If t0/t1 is None, it’s left open.
    """
    n = len(t)
    if n == 0:
        return t, X
    i0 = 0 if t0 is None else int(np.searchsorted(t, t0, side="left"))
    i1 = n if t1 is None else int(np.searchsorted(t, t1, side="right"))
    i0 = max(0, min(i0, n))
    i1 = max(0, min(i1, n))
    return t[i0:i1], X[i0:i1, :]
