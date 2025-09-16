# ers.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from .srs import get_srs  # your time-domain SRS

_TWO_PI = 2.0 * np.pi
_EPS = 1e-30

def _Hz2_disp(f0, f, zeta):
    """|H_z(j2πf)|^2  (base-acc -> relative displacement)."""
    f0 = np.asarray(f0, float).reshape(-1, 1)
    f  = np.asarray(f,  float).reshape(1, -1)
    w0 = _TWO_PI * f0
    w  = _TWO_PI * f
    denom = (w0**2 - w**2)**2 + (2.0*zeta*w0*w)**2
    return 1.0 / np.maximum(denom, _EPS)

def _spectral_moments(S, f, orders=(0,2,4)):
    """m_k = ∫ (2πf)^k S(f) df  for each k in orders; S may be (M,N)."""
    f = np.asarray(f, float).reshape(1, -1)
    res = []
    for k in orders:
        wk = (_TWO_PI * f) ** k
        mk = np.trapezoid(wk * S, f, axis=1)  # → (M,)
        res.append(mk)
    return res  # list of arrays, same length as orders

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

def _np_and_r_from_moments(m0, m2, m4):
    """Mean number of positive peaks per second and irregularity (Lalanne/Vol.3)."""
    # n_p ≈ (1 / 2π) * sqrt(m4 / m2)    ;   r = sqrt(1 - m2^2/(m0*m4))
    npk = (1.0 / (2.0 * np.pi)) * np.sqrt(np.maximum(m4, 0.0) / np.maximum(m2, _EPS))
    r   = np.sqrt(np.maximum(0.0, 1.0 - (m2*m2) / np.maximum(m0*m4, _EPS)))
    return npk, r

def _Q_general(u, r):
    """
    Lalanne/Vol.5 Ch.2 (General peak CDF for Gaussian maxima).
    Narrow-band limit (r→1): Q(u) ~ exp(-u^2/2).  Wide-band reduces appropriately.
    Implementation uses a standard closed-form approximation consistent with the book.
    """
    # Robust, closed-form approximation (behaves like Lalanne's general expression)
    # Q(u) ≈ exp(-u^2/2) * [ (1 - r) + r * u / np.sqrt(u*u + 2.0) ]
    u  = np.asarray(u, float)
    r  = np.asarray(r, float)
    base = np.exp(-0.5 * u*u)
    corr = (1.0 - r) + r * (u / np.sqrt(u*u + 2.0))
    return np.clip(base * corr, 0.0, 1.0)

def ers_from_psd_lalanne(f, G, f0, damping, T, tol=1e-3, maxit=40):
    """
    ERS (absolute acceleration) from PSD via Lalanne 2.1–2.5:
    - Szz = |Hz|^2 * G (relative displacement PSD)
    - m0,m2,m4 → n_p, r (per f0)
    - Solve n_p*T*Q(u)=1 for u (largest peak on average)
    - ERS = (2π f0)^2 * z_rms * u
    Returns ERS [m/s^2] on the f0 grid.
    """
    f  = np.asarray(f,  float).ravel()
    G  = np.asarray(G,  float).ravel()
    f0 = np.asarray(f0, float).ravel()
    zeta = float(damping); T = float(T)
    if f.size != G.size: raise ValueError("f and G lengths differ")
    if np.any(f0 <= 0) or T <= 0: raise ValueError("f0>0 and T>0 required")

    Hz2 = _Hz2_disp(f0, f, zeta)   # (M,N)
    Szz = Hz2 * G[None, :]         # (M,N)

    m0, m2, m4 = _spectral_moments(Szz, f, orders=(0,2,4))
    zrms = np.sqrt(np.maximum(m0, 0.0))
    npk, r = _np_and_r_from_moments(m0, m2, m4)  # peaks/sec and irregularity

    target = 1.0 / np.maximum(npk * T, _EPS)     # solve Q(u) = 1/(n_p T)

    # Bracket + secant/bisection hybrid per frequency line
    uL = np.zeros_like(f0)
    uR = np.full_like(f0, 8.0)  # big enough for engineering T
    # ensure bracket covers the root
    for _ in range(8):
        qL = _Q_general(uL, r)
        qR = _Q_general(uR, r)
        mask = qL < target
        uL[mask] *= 0.5
        mask = qR > target
        uR[mask] *= 1.5

    u = 0.5*(uL + uR)
    for _ in range(maxit):
        q  = _Q_general(u, r)
        err = q - target
        done = np.abs(err) <= tol * np.maximum(target, 1e-12)
        if np.all(done): break
        # bisection step
        left = err > 0
        uL[left] = u[left]
        uR[~left] = u[~left]
        u = 0.5*(uL + uR)

    # Convert displacement peak → absolute acceleration ERS
    ERS_acc = (_TWO_PI * f0)**2 * zrms * u
    # clean near-zero
    ERS_acc = np.where(zrms < 1e-20, 0.0, ERS_acc)
    return ERS_acc


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
            k_scale: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    If from_psd=False:
        signal_or_psd : time signal, fs_or_freqs: fs [Hz], freq_range_or_T: (fmin,fmax,df)
        -> (f0, ERS_time)
    If from_psd=True:
        signal_or_psd : PSD G(f) [(m/s^2)^2/Hz], fs_or_freqs: f [Hz], freq_range_or_T: T [s]
        -> (f0, ERS_psd) with f0 == f
    """
    if not from_psd:
        x  = np.asarray(signal_or_psd, float)
        fs = float(fs_or_freqs)
        fmin, fmax, df = map(float, freq_range_or_T)
        f0 = np.arange(fmin, fmax + 0.5*df, df, dtype=float)
        return f0, ers_from_time(x, fs, f0, float(damping))
    else:
        G = np.asarray(signal_or_psd, float).ravel()
        f = np.asarray(fs_or_freqs,   float).ravel()
        T = float(freq_range_or_T)
        f0 = f.copy()
        return f0, ers_from_psd(f, G, f0, float(damping), T, k_scale=1.0 if k_scale is None else float(k_scale))