# ers.py
import numpy as np
from typing import Tuple
from .srs import get_srs  # reuse your time-domain SRS engine

_TWO_PI = 2.0 * np.pi
_EPS    = 1e-30

def _hz_Hz2_displacement(f0: np.ndarray, f: np.ndarray, zeta: float) -> np.ndarray:
    """
    |H_z(j2πf)|^2 for base-acceleration -> relative displacement of SDOF.
    H_z(ω) = 1 / ( (ω0^2 - ω^2) + j*2ζω0ω )
    => |H_z|^2 = 1 / [ (ω0^2 - ω^2)^2 + (2ζω0ω)^2 ]
    Shapes:
    f0: (M,), f: (N,) -> return (M, N)
    """
    f0 = np.asarray(f0, float).reshape(-1, 1)      # (M,1)
    f  = np.asarray(f,  float).reshape(1, -1)      # (1,N)
    w0 = _TWO_PI * f0
    w  = _TWO_PI * f
    denom = (w0**2 - w**2)**2 + (2.0 * zeta * w0 * w)**2
    return 1.0 / np.maximum(denom, _EPS)

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
    ERS for stationary Gaussian random vibration (one-sided PSD G over f in Hz):
    z_rms^2   = ∫ |H_z|^2 G df
    zdot_rms^2= ∫ (2πf)^2 |H_z|^2 G df
      n0        = (1/π) * (zdot_rms / z_rms)
      ERS       = (2π f0)^2 * z_rms * sqrt( 2 ln( max(n0*T, 1+) ) )

    Units:
    f [Hz], G [(m/s^2)^2/Hz], ERS [m/s^2], f0 [Hz].
    """
    f   = np.asarray(f,   float).ravel()
    G   = np.asarray(G,   float).ravel()
    f0  = np.asarray(f0,  float).ravel()
    zeta = float(damping)
    T    = float(T)

    if f.size != G.size:
        raise ValueError("f and G must have the same length")
    if np.any(f0 <= 0) or T <= 0:
        raise ValueError("f0 must be > 0 and T>0")

    # |H_z|^2 on the (M,N) grid
    Hz2 = _hz_Hz2_displacement(f0, f, zeta)  # (M,N)

    # One-sided integrations over frequency in Hz
    z_rms2    = np.trapz(Hz2 * G[None, :], f[None, :], axis=1)              # (M,)
    zdot_rms2 = np.trapz((_TWO_PI * f[None, :])**2 * Hz2 * G[None, :],
                        f[None, :], axis=1)

    z_rms    = np.sqrt(np.maximum(z_rms2,    0.0))
    zdot_rms = np.sqrt(np.maximum(zdot_rms2, 0.0))

    # Rice zero-crossing rate (narrow-band approximation)
    n0 = (1.0 / np.pi) * (zdot_rms / np.maximum(z_rms, _EPS))

    # Guard the logarithm argument; if n0*T <= 1, the expected max ~ sigma, so clamp at >1
    arg = np.maximum(n0 * T, 1.0000001)
    extreme_factor = np.sqrt(2.0 * np.log(arg))

    ERS = (_TWO_PI * f0)**2 * z_rms * extreme_factor
    # Clean any numerical junk if z_rms≈0
    ERS = np.where(z_rms < 1e-20, 0.0, ERS)
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
