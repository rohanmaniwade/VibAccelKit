# ers.py
import numpy as np
from typing import Tuple
from .srs import get_srs  # reuse your time-domain SRS engine

_TWO_PI = 2.0 * np.pi
_EPS    = 1e-30

def _Ha2_accel(f0: np.ndarray, f: np.ndarray, zeta: float) -> np.ndarray:
    """
    |H_a(j2πf)|^2 for base-acceleration -> absolute acceleration of SDOF mass.
    For base accel input a_b, z satisfies: z¨ + 2ζω0 z˙ + ω0² z = -a_b.
    Absolute acceleration y = z¨ + a_b = -2ζω0 z˙ - ω0² z  (linear in a_b).
    The resulting acceleration FRF magnitude squared is:

    |H_a|^2 = ((ω0^4 + (2ζω0ω)^2) / denom)

    where denom = (ω0² - ω²)² + (2ζω0ω)²

    Shapes: f0: (M,), f: (N,) -> return (M, N)
    """
    f0 = np.asarray(f0, float).reshape(-1, 1)
    f  = np.asarray(f,  float).reshape(1, -1)
    w0 = _TWO_PI * f0
    w  = _TWO_PI * f

    denom = (w0**2 - w**2)**2 + (2.0 * zeta * w0 * w)**2
    numer = (w0**4) + (2.0 * zeta * w0 * w)**2
    return numer / np.maximum(denom, _EPS)

def _peak_factor_vanmarcke(m0: np.ndarray, m2: np.ndarray, m4: np.ndarray, T: float) -> np.ndarray:
    """
    Vanmarcke/Cartwright–Longuet-Higgins extreme factor from spectral moments of the
    *acceleration* response process y(t). All arrays shape (M,).

    m0 = ∫ S_yy df,  m2 = ∫ (2πf)^2 S_yy df,  m4 = ∫ (2πf)^4 S_yy df
    ν0  = sqrt(m2/m0)/(2π)         # zero-crossing rate of y(t)
    N   = max(ν0*T, 1)

    k ≈ sqrt(2 ln N) + (γ - ln ln N)/sqrt(2 ln N), γ≈0.5772
    Clamp to [0, ~4] to avoid blow-ups for very small N.
    """
    eps = 1e-30
    gamma = 0.5772156649
    nu0 = np.sqrt((m2 + eps) / (m0 + eps)) / (2*np.pi)
    N   = np.maximum(nu0 * T, 1.0 + 1e-6)
    L   = np.log(N)
    base = np.sqrt(2*L)
    k = base + (gamma - np.log(L)) / np.maximum(base, 1e-9)
    return np.clip(k, 0.0, 4.0)



def ers_from_time(signal: np.ndarray, fs: float,
                f0: np.ndarray, damping: float) -> np.ndarray:
    """
    ERS from a finite time history: ERS(f0) = max_t |a_abs(t; f0)|
    Reuses your SRS engine and returns the maximax envelope.
    """
    # de-mean the base acceleration for stability (same as your usage)
    x = np.asarray(signal, float) - float(np.mean(signal))
    srs_pos, srs_neg = get_srs(x, float(fs), np.asarray(f0, float), float(damping))
    return np.maximum(srs_pos, np.abs(srs_neg))

def ers_from_psd(f: np.ndarray, G: np.ndarray,
                f0: np.ndarray, damping: float, T: float) -> np.ndarray:
    """
    ERS from one-sided input PSD G(f) by:
    1) building the absolute-acceleration response PSD via |H_a|^2
    2) computing spectral moments m0,m2,m4 of y(t)
    3) using a moments-based peak factor on the *acceleration* process
    4) ERS = k * sqrt(m0)

    Units: f[Hz], G[(m/s^2)^2/Hz], ERS[m/s^2].
    """
    f   = np.asarray(f,   float).ravel()
    G   = np.asarray(G,   float).ravel()
    f0  = np.asarray(f0,  float).ravel()
    zeta = float(damping); T = float(T)
    if f.size != G.size: raise ValueError("f and G must have the same length")
    if np.any(f0 <= 0) or T <= 0: raise ValueError("f0 must be > 0 and T>0")

    Ha2 = _Ha2_accel(f0, f, zeta)                   # (M,N)
    Syy = Ha2 * G[None, :]                          # (M,N) acceleration PSD

    # Moments of acceleration response process
    m0 = np.trapz(Syy,                      f[None, :], axis=1)                      # var(y)
    m2 = np.trapz(( _TWO_PI * f[None, :])**2 * Syy, f[None, :], axis=1)
    m4 = np.trapz(( _TWO_PI * f[None, :])**4 * Syy, f[None, :], axis=1)

    sigma_y = np.sqrt(np.maximum(m0, 0.0))          # RMS of acceleration response
    k = _peak_factor_vanmarcke(m0, m2, m4, T)       # dimensionless

    ERS = k * sigma_y
    ERS = np.where(sigma_y < 1e-20, 0.0, ERS)       # clean tiny bins
    return ERS


def get_ers(signal_or_psd,
            fs_or_freqs,
            freq_range_or_T,
            damping: float,
            from_psd: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified ERS front-end (FatigueDS-free), API-compatible with your old vak.get_ers:

    - If from_psd=False:
        signal_or_psd : time signal (np.ndarray)
        fs_or_freqs   : sampling rate fs [Hz]
        freq_range_or_T: (fmin, fmax, df)
        returns (f0, ERS_time)

    - If from_psd=True:
        signal_or_psd : PSD array G(f) [(m/s^2)^2/Hz], one-sided
        fs_or_freqs   : frequency vector f [Hz] aligned with PSD
        freq_range_or_T: duration T [s]
        returns (f0, ERS_psd)   with f0 == fs_or_freqs (convention preserved)
    """
    if not from_psd:
        x  = np.asarray(signal_or_psd, float)
        fs = float(fs_or_freqs)
        fmin, fmax, df = map(float, freq_range_or_T)
        f0 = np.arange(fmin, fmax + df/2.0, df, dtype=float)
        ers = ers_from_time(x, fs, f0, float(damping))
        return f0, ers
    else:
        G = np.asarray(signal_or_psd, float).ravel()
        f = np.asarray(fs_or_freqs,   float).ravel()
        T = float(freq_range_or_T)
        f0 = f.copy()  # evaluate ERS on the same grid by convention
        ers = ers_from_psd(f, G, f0, float(damping), T)
        return f0, ers
