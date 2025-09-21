# src/vibeaccelkit/stats.py
import numpy as np
from scipy.integrate import trapezoid
import scipy.signal as sps
from typing import Optional, Tuple

def time_rms(x: np.ndarray, demean: bool = True) -> float:
    """Root-mean-square of a time signal. If demean=True, removes DC first."""
    x = np.asarray(x, float)
    if demean:
        x = x - np.mean(x)
    return float(np.sqrt(np.mean(x**2)))

def psd_rms(f: np.ndarray, G: np.ndarray, band: Optional[Tuple[float, float]] = None) -> float:
    """
    RMS from a ONE-SIDED PSD by integration. Optionally restrict to a band [fmin, fmax].
    Units preserved (e.g., m/s²). Assumes density units (e.g., (m/s²)²/Hz).
    """
    f = np.asarray(f, float); G = np.asarray(G, float)
    if band is not None:
        fmin, fmax = float(band[0]), float(band[1])
        m = (f >= fmin) & (f <= fmax)
        if not np.any(m):
            return 0.0
        f = f[m]; G = G[m]
    var = float(trapezoid(np.maximum(G, 0.0), f))
    return float(np.sqrt(max(var, 0.0)))

def welch_psd(sig: np.ndarray, fs: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: one-sided Welch PSD with DC removed (detrend='constant').
    Returns (f, G) suitable for psd_rms().
    """
    if nperseg is None:
        nperseg = int(max(8, fs * 2))  # ~2 s windows by default
    f, G = sps.welch(
        sig, fs=fs, window="hann", nperseg=nperseg, detrend="constant",
        return_onesided=True, scaling="density"
    )
    return f.astype(float), G.astype(float)


def make_freq_grid(freq_range: Tuple[float, ...], bins: str = "log",
                   points_per_decade: int = 24, n_points: Optional[int] = None) -> np.ndarray:
    """
    Build a frequency grid from a freq_range tuple.

    freq_range may be (fmin, fmax) or (fmin, fmax, df). When `bins=='log'`
    returns a log-spaced grid with `n_points` or computed from `points_per_decade`.
    When `bins=='linear'` and a df is provided, returns an arange using df, else
    uses linspace with `n_points` or a default of 1000 points.
    """
    fr = tuple(float(x) for x in freq_range)
    if len(fr) < 2:
        raise ValueError("freq_range must have at least (fmin, fmax)")
    fmin, fmax = fr[0], fr[1]
    if fmin <= 0 or fmax <= 0 or fmax <= fmin:
        raise ValueError("freq_range must satisfy 0 < fmin < fmax")

    if bins == "log":
        if n_points is not None:
            n = int(n_points)
        else:
            decades = np.log10(fmax) - np.log10(fmin)
            n = int(np.round(decades * float(points_per_decade))) + 1
        return np.logspace(np.log10(fmin), np.log10(fmax), n)
    else:
        # linear
        if len(fr) >= 3:
            df = fr[2]
            return np.arange(fmin, fmax + 0.5 * df, df, dtype=float)
        if n_points is not None:
            return np.linspace(fmin, fmax, int(n_points))
        return np.linspace(fmin, fmax, 1000)
