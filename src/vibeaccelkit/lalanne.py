import numpy as np
from scipy.special import gamma

def fds_to_psd(freqs, fds_comp, T_eq, b, Q):
    """
    Lalanne inversion:
    PSD_eq(f) = (8πf/Q) * [ FDS_comp(f) / ( f * T_eq * Γ(1 + b/2) ) ]^(1/b)
    """
    sigma_pv_sq = (fds_comp / (freqs * T_eq * gamma(1.0 + b/2.0))) ** (1.0 / b)
    return (sigma_pv_sq * 8.0 * np.pi * freqs) / Q

def accelerate_psd(psd_eq, T_eq, T_acc, b):
    """
    PSD_acc = PSD_eq * (T_eq / T_acc)^(2/b)
    """
    return psd_eq * (T_eq / float(T_acc)) ** (2.0 / b)
