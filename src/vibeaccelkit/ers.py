# ers.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from .srs import get_srs  # your time-domain SRS
from .stats import make_freq_grid

_TWO_PI = 2.0 * np.pi
_EPS = 1e-30


def _Ha2_accel(f0: np.ndarray, f: np.ndarray, zeta: float) -> np.ndarray:
    """
    |H_a(j2πf)|^2 for base-acc -> absolute acceleration of SDOF mass.
    |H_a|^2 = (ω0^4 + (2ζω0ω)^2) / [ (ω0^2 - ω^2)^2 + (2ζω0ω)^2 ].
    Shapes: f0: (M,), f: (N,) -> (M,N)
    """
    f0 = np.asarray(f0, float).reshape(-1, 1)
    f  = np.asarray(f,  float).reshape(1, -1)
    w0 = _TWO_PI * f0
    w  = _TWO_PI * f
    denom = (w0**2 - w**2)**2 + (2.0 * zeta * w0 * w)**2
    numer = (w0**4) + (2.0 * zeta * w0 * w)**2
    return numer / np.maximum(denom, _EPS)

def _peak_factor_vanmarcke(m0, m2, m4, T):
    eps = 1e-30
    gamma = 0.5772156649
    nu0 = np.sqrt((m2 + eps) / (m0 + eps)) / (2*np.pi)
    N   = np.maximum(nu0 * float(T), 1.0 + 1e-6)
    L   = np.log(N)
    base = np.sqrt(2.0 * L)
    k = base + (gamma - np.log(L)) / np.maximum(base, 1e-9)

    # Tonal guard (tames very narrow-band peaks)
    eps_bw = np.clip(1.0 - (m2*m2)/np.maximum(m0*m4, eps), 0.0, 1.0)
    w = np.exp(-eps_bw / 0.03)      # 0.02–0.05 reasonable
    k_tone_cap = 1.6                # near tone crest factor
    k = (1.0 - w) * k + w * np.minimum(k, k_tone_cap)

    return np.clip(k, 0.0, 4.0)

# The earlier Lalanne spectral root-finding implementation has been removed.
# We keep two ERS pathways: ers_from_time (time-history) and ers_from_psd
# (Vanmarcke peak-factor / spectral-moments-based acceleration ERS).


def ers_from_time(signal: np.ndarray, fs: float, f0: np.ndarray, damping: float) -> np.ndarray:
    """ERS from a finite time history (maximax of absolute acceleration)."""
    x = np.asarray(signal, float) - float(np.mean(signal))
    srs_pos, srs_neg = get_srs(x, float(fs), np.asarray(f0, float), float(damping))
    return np.maximum(srs_pos, np.abs(srs_neg))

def ers_from_psd(f: np.ndarray, G: np.ndarray, f0: np.ndarray, damping: float, T: float,
                k_scale: float = 1.0) -> np.ndarray:
    """
    ERS from one-sided input PSD G(f) [(m/s^2)^2/Hz]:
      Syy = |Ha|^2 * G
    σ_y = sqrt(∫Syy df)
    k   = Vanmarcke peak factor (acceleration moments)
      ERS = k_scale * k * σ_y
    """
    f   = np.asarray(f,   float).ravel()
    G   = np.asarray(G,   float).ravel()
    f0  = np.asarray(f0,  float).ravel()
    zeta = float(damping); T = float(T)
    if f.size != G.size: raise ValueError("f and G must have same length")
    if np.any(f0 <= 0) or T <= 0: raise ValueError("f0>0 and T>0 required")

    Ha2 = _Ha2_accel(f0, f, zeta)                # (M,N)
    Syy = Ha2 * G[None, :]                       # (M,N)

    m0 = np.trapezoid(Syy,                      f[None, :], axis=1)
    m2 = np.trapezoid((2*np.pi*f[None,:])**2 * Syy, f[None, :], axis=1)
    m4 = np.trapezoid((2*np.pi*f[None,:])**4 * Syy, f[None, :], axis=1)

    sigma_y = np.sqrt(np.maximum(m0, 0.0))
    k = _peak_factor_vanmarcke(m0, m2, m4, T)
    return np.where(sigma_y < 1e-20, 0.0, float(k_scale) * k * sigma_y)

def get_ers(signal_or_psd,
            fs_or_freqs,
            freq_range_or_T,
            damping: float,
            from_psd: bool = False,
            k_scale: float | None = None,
            bins: str = "log",
            points_per_decade: int = 24,
            n_points: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ERS helper that supports log- or linear-spaced f0 grids.

    If from_psd=False:
        signal_or_psd : time signal, fs_or_freqs: fs [Hz], freq_range_or_T: (fmin,fmax,df)
        -> (f0, ERS_time)
        - When `bins=='log'` the third element of `freq_range_or_T` is ignored and
        the grid is log-spaced. Use `n_points` to control the number of points
        or `points_per_decade` to set resolution per decade.
        - When `bins=='linear'` behavior is unchanged: third element is df.

    If from_psd=True:
        signal_or_psd : PSD G(f) [(m/s^2)^2/Hz], fs_or_freqs: f [Hz], freq_range_or_T: T [s]
        -> (f0, ERS_psd) with f0 == f
    """
    if not from_psd:
        x  = np.asarray(signal_or_psd, float)
        fs = float(fs_or_freqs)
        # expect (fmin, fmax, df) but make_freq_grid will accept (fmin,fmax,df)
        f0 = make_freq_grid(tuple(map(float, freq_range_or_T)), bins=bins,
                             points_per_decade=points_per_decade, n_points=n_points)
        return f0, ers_from_time(x, fs, f0, float(damping))
    else:
        G = np.asarray(signal_or_psd, float).ravel()
        f = np.asarray(fs_or_freqs,   float).ravel()
        T = float(freq_range_or_T)
        f0 = f.copy()
        return f0, ers_from_psd(f, G, f0, float(damping), T, k_scale=1.0 if k_scale is None else float(k_scale))