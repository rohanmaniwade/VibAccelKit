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
