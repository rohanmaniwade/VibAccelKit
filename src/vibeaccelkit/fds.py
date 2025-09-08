from typing import Tuple
import numpy as np
from scipy.signal import welch

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
    """RMS from one-sided PSD using Parseval: σ^2 = ∫ G df."""
    return float(np.sqrt(np.trapezoid(np.asarray(G, float), np.asarray(f, float))))

def get_fds(signal: np.ndarray, fs: float, freq_range: tuple, damping: float,
            b: float = 7.0, C: float = 1.0, K: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convolution-based FDS via FatigueDS. Returns (frequencies, fds).
    """
    from FatigueDS import SpecificationDevelopment
    sd = SpecificationDevelopment(freq_data=freq_range, damp=damping)
    sd.set_random_load((signal, 1.0/fs), unit='ms2', method='convolution')
    sd.get_fds(b=b, C=C, K=K)
    return sd.f0_range, sd.fds

def get_ers(signal_or_psd, fs_or_freqs, freq_range_or_T, damping: float,
            from_psd: bool = False):
    """
    ERS via FatigueDS.
    - If from_psd=False: signal_or_psd is time signal, fs_or_freqs is fs (Hz), freq_range_or_T is freq_range (tuple)
    - If from_psd=True:  signal_or_psd is PSD array, fs_or_freqs is freq vector, freq_range_or_T is duration T (s)
    Returns (frequencies, ers).
    """
    from FatigueDS import SpecificationDevelopment
    if not from_psd:
        x, fs, freq_range = signal_or_psd, fs_or_freqs, freq_range_or_T
        sd = SpecificationDevelopment(freq_data=freq_range, damp=damping)
        sd.set_random_load((x, 1.0/fs), unit='ms2', method='timehistory')
    else:
        psd, freqs, T = signal_or_psd, fs_or_freqs, float(freq_range_or_T)
        freq_range = (float(freqs[0]), float(freqs[-1]), float(freqs[1]-freqs[0]))
        sd = SpecificationDevelopment(freq_data=freq_range, damp=damping)
        sd.set_random_load((psd, freqs), unit='ms2', T=T)
    sd.get_ers()
    return sd.f0_range, sd.ers

import numpy as np

def fds_from_psd(
    f: np.ndarray,
    G: np.ndarray,
    T: float,
    f0: np.ndarray,
    damp: float = 0.05,
    b: float = 8.0,
    C: float = 1.0,
    K: float = 1.0,
) -> np.ndarray:
    """
    Compute the Fatigue Damage Spectrum (FDS) from a ONE-SIDED acceleration PSD.

    Parameters
    ----------
    f : (N,) array
        Frequency axis [Hz] for G (one-sided, 0..Nyquist).
    G : (N,) array
        PSD [(m/s^2)^2/Hz], one-sided, aligned to `f`.
    T : float
        Duration [s] represented by the PSD.
    f0 : (M,) array
        Natural frequencies [Hz] at which to evaluate the FDS.
    damp : float
        Modal damping ratio ζ (e.g., 0.05 for 5%).
    b : float
        S-N slope exponent for damage (Lalanne b).
    C, K : float
        Material constants. Keep C=K=1.0 unless you deliberately use S-N data.

    Returns
    -------
    D : (M,) array
        FDS values (damage units) at each f0.
    """
    f = np.asarray(f, float)
    G = np.asarray(G, float)
    f0 = np.asarray(f0, float)

    if f.ndim != 1 or G.ndim != 1 or f.shape[0] != G.shape[0]:
        raise ValueError("f and G must be 1D arrays of equal length")
    if np.any(f0 <= 0):
        raise ValueError("All f0 must be > 0")

    zeta = float(damp)

    # Broadcasting: r has shape (M, N)
    r = f0[:, None]
    x = f[None, :] / r  # (f/f0)

    # |H(jω)|^2 for SDOF acceleration response
    H2 = 1.0 / ((1.0 - x**2)**2 + (2.0 * zeta * x)**2)

    # Response variance σ^2(f0) = ∫ |H|^2 G df  (one-sided)
    sigma2 = np.trapz(H2 * G[None, :], f[None, :], axis=1)
    sigma = np.sqrt(np.maximum(sigma2, 0.0))

    # Damage: D(f0) = K * T * sigma^b / C
    D = K * T * (sigma**b) / C
    return D
