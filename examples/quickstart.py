import numpy as np
import vibeaccelkit as vak

# Load one mission (add more and combine as needed)
t, x, fs = vak.load_mat_timesignal("../data/srb_iea_th.mat")

freq_range = (10, 2000, 10)
damp = 0.05
b = 7.0
Q = 1/(2*damp)

# Per-signal spectra
f0, fds = vak.get_fds(x, fs, freq_range, damp, b=b)
_, ers_time = vak.get_ers(x, fs, freq_range, damp, from_psd=False)
srs_pos, srs_neg = vak.get_srs(x, fs, f0, damp)

# Composite FDS (example with two events)
fds_comp = vak.combine_fds(f0, [fds, fds], [300, 300])
T_eq = 600.0

# Lalanne inversion + acceleration
psd_eq = vak.fds_to_psd(f0, fds_comp, T_eq=T_eq, b=b, Q=Q)
psd_acc = vak.accelerate_psd(psd_eq, T_eq=T_eq, T_acc=3600, b=b)

# ERS from accelerated PSD
_, ers_acc = vak.get_ers(psd_acc, f0, 3600, damp, from_psd=True)

# Validation: ERS vs 2×SRS+
ok, mask = vak.validate_srs_ers(srs_pos, ers_acc, factor=2.0)
print("ERS ≤ 2×SRS ? ", "PASS" if ok else "FAIL (see frequencies with mask=True)")

# Optional: synthesize time history from accelerated PSD and play with it
x_acc = vak.synthesize_time_from_psd(f0, psd_acc, fs=4096, duration=3600, seed=1)

# Plot
vak.plot_fds(f0, {"FDS": fds, "Composite": fds_comp}).show()
vak.plot_ers(f0, {"ERS (original time)": ers_time, "ERS (acc PSD)": ers_acc}).show()
vak.plot_srs(f0, {"Original": (srs_pos, srs_neg)}).show()
vak.plot_psd(f0, {"PSD_eq": psd_eq, "PSD_acc_1h": psd_acc}).show()
