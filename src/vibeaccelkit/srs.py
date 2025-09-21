from typing import Tuple, Union, Optional
import numpy as np
from .stats import make_freq_grid

def get_srs(signal: np.ndarray, fs: float, freqs: Union[np.ndarray, Tuple[float, float, float]], damping: float,
            bins: str = "log", points_per_decade: int = 24, n_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Acceleration SRS (peak absolute *absolute-acceleration* of unit-mass SDOF)
    for a base-acceleration input using a ramp-invariant (Tuma-style) update.

    Returns:
        (SRS_pos, SRS_neg)  -- both in [m/s^2], on the same grid as `freqs`.
        SRS_pos is the peak positive |a_abs|, SRS_neg is the peak negative |a_abs|.
        (Both are returned as absolute magnitudes for convenience.)
    """
    # Allow passing a freq_range tuple (fmin, fmax, df) to build a grid
    if not isinstance(freqs, np.ndarray):
        f0 = make_freq_grid(tuple(freqs), bins=bins, points_per_decade=points_per_decade, n_points=n_points)
    else:
        f0 = np.asarray(freqs, float)

    omega = 2 * np.pi * f0
    dt = 1.0 / fs
    srs_pos = np.zeros_like(omega)
    srs_neg = np.zeros_like(omega)

    for i, wn in enumerate(omega):
        zeta = damping
        k = wn**2
        a = np.exp(-zeta * wn * dt)
        b = wn * np.sqrt(max(0.0, 1.0 - zeta**2))

        if b < 1e-20:  # critically damped limit safeguard
            A = a
            B = a * dt
            C = -wn * a * dt
            D = a
        else:
            sin_bdt = np.sin(b * dt)
            cos_bdt = np.cos(b * dt)
            A = a * cos_bdt
            B = a * sin_bdt / b
            C = -wn * a * sin_bdt
            D = a * cos_bdt - 2 * zeta * wn * a * sin_bdt / b

        x = 0.0
        v = 0.0
        peak_pos = 0.0
        peak_neg = 0.0

        for a_base in signal:  # base acceleration sample
            # ramp-invariant state update for base-accel forcing
            x_new = A * x + B * v + (1.0 - A) * (a_base / k)
            v_new = C * x + D * v + (B * k) * (a_base / k)  # simplified; jerk term neglected
            x, v = x_new, v_new

            # absolute acceleration of the mass:
            #   a_abs = -2ζω v - ω^2 x
            a_abs = -2.0 * zeta * wn * v - k * x

            if a_abs > peak_pos: peak_pos = a_abs
            if a_abs < peak_neg: peak_neg = a_abs

        srs_pos[i] = abs(peak_pos)
        srs_neg[i] = abs(peak_neg)

    return srs_pos, srs_neg

