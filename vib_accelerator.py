import os
import numpy as np
from scipy.io import loadmat
from scipy.special import gamma
from scipy.integrate import trapezoid
import plotly.graph_objs as go
import FatigueDS

# ===========================
# USER SETTINGS
# ===========================
DATA_DIR = "data"
freq_range = (10, 2000, 10)
damping = 0.05
b_default = 7
C_default = 1
K_default = 1
Q_default = 1 / (2 * damping)
T_acc_list = [1*3600, 10*3600, 30*3600]  # in seconds

# ===========================
# HELPERS
# ===========================
def read_mat_timesignal(filepath):
    """Load .mat file with [time, signal] columns."""
    mat = loadmat(filepath)
    main_key = [k for k in mat.keys() if not k.startswith("__")][0]
    data = mat[main_key]
    time = data[:, 0]
    signal = data[:, 1]
    fs = 1.0 / (time[1] - time[0])
    return time, signal, fs

def rms_from_psd(psd, freqs):
    return np.sqrt(trapezoid(psd, freqs))

def generate_accelerated_psd(freqs, fds_list, durations, T_acc, b=7, Q=10):
    fds_composite = np.zeros_like(freqs)
    for fds, duration in zip(fds_list, durations):
        fds_composite += fds * duration
    T_eq = sum(durations)
    sigma_pv_sq = (fds_composite / (freqs * T_eq * gamma(1 + b / 2))) ** (1 / b)
    psd_eq = (sigma_pv_sq * 8 * np.pi * freqs) / Q
    psd_acc = psd_eq * (T_eq / T_acc) ** (2 / b)
    return psd_acc, psd_eq, fds_composite, T_eq

# ===========================
# SRS FUNCTION (Tuma method)
# ===========================
def get_srs(signal, fs, freqs, damping):
    omega = 2 * np.pi * freqs
    dt = 1 / fs
    SRS_pos = np.zeros_like(freqs)
    SRS_neg = np.zeros_like(freqs)

    for i, wn in enumerate(omega):
        zeta = damping
        k = wn ** 2
        a = np.exp(-zeta * wn * dt)
        b = wn * np.sqrt(1 - zeta ** 2)
        sin_bdt = np.sin(b * dt)
        cos_bdt = np.cos(b * dt)

        A = a * cos_bdt
        B = a * sin_bdt / b
        C = -wn * a * sin_bdt
        D = a * cos_bdt - 2 * zeta * wn * a * sin_bdt / b

        x, v = 0.0, 0.0
        max_pos, max_neg = 0.0, 0.0

        for f in signal:
            x_new = A * x + B * v + (1 - A) * f / k
            v_new = C * x + D * v + (B * k - D / dt) * f / k
            x, v = x_new, v_new
            max_pos = max(max_pos, x)
            max_neg = min(max_neg, x)

        SRS_pos[i] = abs(max_pos)
        SRS_neg[i] = abs(max_neg)

    return SRS_pos, SRS_neg

# ===========================
# ERS FROM PSD
# ===========================
def get_ers_from_psd(psd, freqs, T_acc, damp=damping):
    sd = FatigueDS.SpecificationDevelopment(freq_data=(freqs[0], freqs[-1], freqs[1]-freqs[0]), damp=damp)
    sd.set_random_load((psd, freqs), unit='ms2', T=T_acc)
    sd.get_ers()
    return sd.ers

# ===========================
# VALIDATION FUNCTION
# ===========================
def validate_srs_ers(srs_pos, ers, factor=2.0):
    violations = ers > (factor * srs_pos)
    valid = not np.any(violations)
    return valid, violations

# ===========================
# PLOTTING HELPERS
# ===========================
def plot_fds(fds_dict, freqs):
    fig = go.Figure()
    for label, fds in fds_dict.items():
        fig.add_trace(go.Scatter(x=freqs, y=fds, mode='lines', name=label))
    fig.update_layout(title='FDS', xaxis_title='Frequency [Hz]', yaxis_title='FDS', yaxis_type='log')
    fig.show()

def plot_ers(ers_dict, freqs):
    fig = go.Figure()
    for label, ers in ers_dict.items():
        fig.add_trace(go.Scatter(x=freqs, y=ers, mode='lines', name=label))
    fig.update_layout(title='ERS', xaxis_title='Frequency [Hz]', yaxis_title='ERS', yaxis_type='log')
    fig.show()

def plot_srs(srs_dict, freqs):
    fig = go.Figure()
    for label, (srs_p, srs_n) in srs_dict.items():
        fig.add_trace(go.Scatter(x=freqs, y=srs_p, mode='lines', name=f"{label} SRS+"))
        fig.add_trace(go.Scatter(x=freqs, y=srs_n, mode='lines', name=f"{label} SRS-"))
    fig.update_layout(title='SRS', xaxis_title='Frequency [Hz]', yaxis_title='SRS Amplitude', yaxis_type='log')
    fig.show()

def plot_psd(psd_dict, freqs):
    fig = go.Figure()
    for label, psd in psd_dict.items():
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name=label))
    fig.update_layout(title='PSD', xaxis_title='Frequency [Hz]', yaxis_title='PSD [$(m/s^2)^2$/Hz]', yaxis_type='log')
    fig.show()

def plot_srs_ers_comparison(freqs, srs_pos, ers_acc, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=srs_pos, mode='lines', name='Original SRS+'))
    fig.add_trace(go.Scatter(x=freqs, y=2.0 * srs_pos, mode='lines', name='2×SRS+', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=freqs, y=ers_acc, mode='lines', name='ERS (Accelerated PSD)'))
    fig.update_layout(title=title, xaxis_title='Frequency [Hz]', yaxis_title='Amplitude', yaxis_type='log')
    fig.show()

# ===========================
# MAIN PROCESS
# ===========================
def main():
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]

    fds_dict = {}
    ers_dict = {}
    srs_dict = {}
    durations = []
    fds_list = []
    names = []

    for fname in mat_files:
        time, sig, fs = read_mat_timesignal(os.path.join(DATA_DIR, fname))
        duration = len(sig) / fs
        durations.append(duration)
        label = os.path.splitext(fname)[0]
        names.append(label)

        # FDS
        sd = FatigueDS.SpecificationDevelopment(freq_data=freq_range, damp=damping)
        sd.set_random_load((sig, 1 / fs), unit='ms2', method='convolution')
        sd.get_fds(b=b_default, C=C_default, K=K_default)
        fds_list.append(sd.fds)
        fds_dict[label] = sd.fds

        # ERS
        sd.get_ers()
        ers_dict[label] = sd.ers

        # SRS
        srs_p, srs_n = get_srs(sig, fs, sd.f0_range, damping)
        srs_dict[label] = (srs_p, srs_n)

    freqs = sd.f0_range

    # Composite FDS
    fds_composite = sum(fds * dur for fds, dur in zip(fds_list, durations))
    fds_dict["Composite FDS"] = fds_composite

    # Accelerated PSDs + Validation
    psd_dict = {}
    for T_acc in T_acc_list:
        psd_acc, _, _, _ = generate_accelerated_psd(freqs, fds_list, durations, T_acc, b=b_default, Q=Q_default)
        psd_dict[f"Accelerated PSD {int(T_acc/3600)}h"] = psd_acc

        ers_acc = get_ers_from_psd(psd_acc, freqs, T_acc)
        srs_pos_comp, _ = srs_dict[names[0]]  # Could also use a merged/composite SRS
        valid, _ = validate_srs_ers(srs_pos_comp, ers_acc, factor=2.0)
        print(f"[Validation] Accelerated PSD {int(T_acc/3600)}h: {'PASS' if valid else 'FAIL'}")

        plot_srs_ers_comparison(freqs, srs_pos_comp, ers_acc,
                                title=f"SRS vs ERS Validation - {int(T_acc/3600)}h")

    # Example manual plots:
    # plot_fds(fds_dict, freqs)
    # plot_ers(ers_dict, freqs)
    # plot_srs(srs_dict, freqs)
    # plot_psd(psd_dict, freqs)

if __name__ == "__main__":
    main()
