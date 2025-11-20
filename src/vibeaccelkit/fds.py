from typing import Tuple
import numpy as np
from scipy.signal import welch
import scipy.signal as sps
from typing import Tuple, Literal

try:
    import rainflow
    _HAS_RAINFLOW = True
except ImportError:
    _HAS_RAINFLOW = False

# ───────── Helpers for time-domain FDS ─────────
def _sdof_rel_disp_from_base_accel(a: np.ndarray, fs: float, f0: float, zeta: float) -> np.ndarray:
    """
    Simulate relative displacement z(t) of a base-excited SDOF oscillator.
    Equation: z'' + 2ζω z' + ω^2 z = -a_base(t)
    """
    w = 2 * np.pi * f0
    A = np.array([[0.0, 1.0],
                [-w**2, -2.0*zeta*w]])
    B = np.array([[0.0], [-1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])

    sysd = sps.cont2discrete((A, B, C, D), dt=1.0/fs, method="bilinear")
    Ad, Bd, Cd, Dd, _ = sysd

    x = np.zeros(2)
    z = np.zeros_like(a)
    for i, ui in enumerate(a):
        x = Ad @ x + Bd.flatten() * ui
        z[i] = (Cd @ x + Dd * ui).item()
    return z


def _rainflow_damage(z: np.ndarray, b: float) -> float:
    if _HAS_RAINFLOW:
        cycles = rainflow.count_cycles(z)
        cycles = np.asarray(cycles, dtype=float)
        rng = cycles[:, 0]   # range
        cnt = cycles[:, -1]  # count
    else:
        peaks, _ = sps.find_peaks(z)
        valleys, _ = sps.find_peaks(-z)
        idx = np.sort(np.concatenate([peaks, valleys]))
        rng = np.abs(np.diff(z[idx]))
        cnt = np.full_like(rng, 0.5)
    return float(np.sum(cnt * 2 * (rng/2.0)**b))


# ───────── Helpers for PSD-domain FDS ─────────
def _sdof_tf_rel_disp(f: np.ndarray, f0: float, zeta: float) -> np.ndarray:
    w = 2 * np.pi * f
    wn = 2 * np.pi * f0
    H = -1.0 / (-w**2 + 2j*zeta*wn*w + wn**2)
    return H

def _spectral_moments(Sz: np.ndarray, f: np.ndarray) -> dict:
    w = 2*np.pi*f
    m0 = np.trapz(Sz, f)
    m2 = np.trapz((w**2)*Sz, f)/(2*np.pi)**2
    m4 = np.trapz((w**4)*Sz, f)/(2*np.pi)**4
    return {"m0": m0, "m2": m2, "m4": m4}

def _rice_damage_rate(sigma: float, np_: float, r: float, b: float) -> float:
    u = np.linspace(0, 8, 2000)
    q = (r/np.sqrt(2*np.pi))*np.exp(-u**2/2) + np.sqrt(1-r**2)*u*np.exp(-u**2/2)
    I = np.trapz(u**b * q, u)
    return np_ * (sigma**b) * I

def _rayleigh_damage_rate(sigma: float, np_: float, b: float) -> float:
    from scipy.special import gamma
    Eb = (2**(b/2)) * gamma(1+b/2)
    return np_ * (sigma**b) * Eb



def time_to_psd(signal_data, fs, freq_range, nperseg=None):
    """
    Compute PSD from time history using Welch's method.

    Parameters
    ----------
    signal_data : ndarray
        Time history signal.
    fs : float
        Sampling frequency [Hz].
    freq_range : tuple
        (fmin, fmax, df) frequency range for PSD output.
    nperseg : int or None
        Segment length for Welch's method (defaults: fs*2).

    Returns
    -------
    freqs : ndarray
        Frequency vector [Hz]
    psd : ndarray
        One-sided PSD [m²/s⁴/Hz]
    """
    fmin, fmax, df = freq_range
    if nperseg is None:
        nperseg = int(fs * 2)  # 2-sec segments → smoother PSD

    freqs, psd = welch(signal_data, fs=fs, nperseg=nperseg)

    # Restrict to desired band
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]

def rms_from_psd(f: np.ndarray, G: np.ndarray) -> float:
    """
    RMS from one-sided PSD using Parseval: σ^2 = ∫ G df.
    Units preserved (e.g., if G is in (m/s²)²/Hz, returns RMS in m/s²).
    """
    return float(np.sqrt(np.trapezoid(np.asarray(G, float), np.asarray(f, float))))

from .stats import make_freq_grid

def get_fds_time(
    x: np.ndarray, fs: float,
    freq_range: Tuple[float,float], damp: float,
    *, b: float = 6.0,
    bins: Literal["log","linear"]="log",
    points_per_decade: int = 24,
    n_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-history → FDS (per-second damage rate).
    """
    # frequency grid
    f0 = make_freq_grid(freq_range, bins=bins, points_per_decade=points_per_decade, n_points=n_points)

    T = len(x)/fs
    fds_rate = np.zeros_like(f0)

    for j, fn in enumerate(f0):
        z = _sdof_rel_disp_from_base_accel(x, fs, fn, damp)
        damage = _rainflow_damage(z, b)
        fds_rate[j] = damage/T   # normalize to per second
    return f0, fds_rate


import numpy as np

def get_fds(
    x: np.ndarray, fs: float,
    freq_range: Tuple[float, float], damp: float,
    *, b: float = 6.0,
    bins: Literal["log", "linear"] = "log",
    points_per_decade: int = 24,
    n_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible convenience wrapper for time-history → FDS.
    Defaults to log-spaced bins for smooth log-scale plotting.

    Parameters mirror get_fds_time with the same defaults.
    """
    return get_fds_time(
        x, fs, freq_range, damp,
        b=b, bins=bins, points_per_decade=points_per_decade, n_points=n_points,
    )

def get_fds_psd(
    Sa: np.ndarray, f: np.ndarray,
    freq_range: Tuple[float,float], damp: float,
    *, b: float = 6.0,
    method: Literal["rice","rayleigh"]="rice",
    bins: Literal["log","linear"]="log",
    points_per_decade: int = 24,
    n_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PSD → FDS (per-second damage rate) via Lalanne §4.2.
    """
    # oscillator grid
    f0 = make_freq_grid(freq_range, bins=bins, points_per_decade=points_per_decade, n_points=n_points)

    fds_rate = np.zeros_like(f0)

    for j, fn in enumerate(f0):
        H = _sdof_tf_rel_disp(f, fn, damp)
        Sz = np.abs(H)**2 * Sa

        m = _spectral_moments(Sz, f)
        sigma = np.sqrt(m["m0"])
        np_ = np.sqrt(m["m4"]/m["m2"]) / (2*np.pi)
        n0  = np.sqrt(m["m2"]/m["m0"]) / (2*np.pi)
        r   = n0/np_

        if method=="rice":
            fds_rate[j] = _rice_damage_rate(sigma, np_, r, b)
        elif method=="rayleigh":
            fds_rate[j] = _rayleigh_damage_rate(sigma, np_, b)
        else:
            raise ValueError(f"Unknown method {method}")
    return f0, fds_rate

# Compatibility alias (older scripts called fds_from_psd)
def fds_from_psd(
    Sa: np.ndarray, f: np.ndarray,
    freq_range: Tuple[float, float], damp: float,
    *, b: float = 6.0,
    method: Literal["rice", "rayleigh"] = "rice",
    bins: Literal["log", "linear"] = "log",
    points_per_decade: int = 24,
    n_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return get_fds_psd(
        Sa, f, freq_range, damp,
        b=b, method=method, bins=bins,
        points_per_decade=points_per_decade, n_points=n_points,
    )

