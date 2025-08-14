from typing import Tuple
import numpy as np

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
