from typing import Tuple
import numpy as np

def get_srs(signal: np.ndarray, fs: float, freqs: np.ndarray, damping: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ramp-invariant SRS (Tuma-style digital SDOF). Returns (SRS_pos, SRS_neg).
    """
    omega = 2 * np.pi * freqs
    dt = 1.0 / fs
    srs_pos = np.zeros_like(freqs)
    srs_neg = np.zeros_like(freqs)

    for i, wn in enumerate(omega):
        zeta = damping
        k = wn**2
        a = np.exp(-zeta * wn * dt)
        b = wn * np.sqrt(max(0.0, 1.0 - zeta**2))
        if b < 1e-20:
            # critically damped limit
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

        for f in signal:
            # ramp-invariant update for base acceleration input
            x_new = A * x + B * v + (1.0 - A) * f / k
            v_new = C * x + D * v + (B * k) * f / k - (D * 0.0)  # simplified, zero jerk term
            x, v = x_new, v_new
            if x > peak_pos: peak_pos = x
            if x < peak_neg: peak_neg = x

        srs_pos[i] = abs(peak_pos)
        srs_neg[i] = abs(peak_neg)

    return srs_pos, srs_neg

def get_srs_abs_accel(signal: np.ndarray, fs: float, freqs: np.ndarray, damping: float) -> np.ndarray:
    """
    Absolute-acceleration SRS (peak |a_abs|) for a unit-mass SDOF
    under base acceleration input, using the same ramp-invariant
    update style as get_srs().
    """
    omega = 2 * np.pi * freqs
    dt = 1.0 / fs
    srs_abs = np.zeros_like(freqs, dtype=float)

    for i, wn in enumerate(omega):
        zeta = damping
        k = wn**2
        a = np.exp(-zeta * wn * dt)
        b = wn * np.sqrt(max(0.0, 1.0 - zeta**2))
        if b < 1e-20:
            A = a; B = a * dt; C = -wn * a * dt; D = a
        else:
            sin_bdt = np.sin(b * dt); cos_bdt = np.cos(b * dt)
            A = a * cos_bdt
            B = a * sin_bdt / b
            C = -wn * a * sin_bdt
            D = a * cos_bdt - 2 * zeta * wn * a * sin_bdt / b

        x = 0.0; v = 0.0; peak = 0.0
        for a_base in signal:
            # state update (base-accel forcing)
            x_new = A * x + B * v + (1.0 - A) * a_base / k
            v_new = C * x + D * v + (B * k) * a_base / k
            x, v = x_new, v_new

            # absolute acceleration of mass: a_abs = -2ζω v - ω² x
            a_abs = -2.0 * zeta * wn * v - k * x
            if abs(a_abs) > peak:
                peak = abs(a_abs)

        srs_abs[i] = peak

    return srs_abs

