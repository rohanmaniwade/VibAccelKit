# examples/process_folder.py
import os
from pathlib import Path
import numpy as np
import plotly.io as pio
import vibeaccelkit as vak

# Render figures in a browser window
pio.renderers.default = "browser"

# ---------- CONFIG ----------
DATA_DIR = Path("data")        # put your .mat / .csv / .txt files here
OUT_DIR  = Path("results")     # outputs (CSVs + HTML plots)
OUT_DIR.mkdir(parents=True, exist_ok=True)

damp = 0.05                    # damping (e.g., 5% => Q = 1/(2ζ))
b = 7.0                        # Lalanne exponent
Q = 1.0 / (2.0 * damp)
T_acc = 3600.0                 # accelerated test duration (seconds) — change as needed

# One common linear grid for everything (good for FatigueDS freq_range)
fmin, fmax, df = 10.0, 2000.0, 10.0
f0 = np.arange(fmin, fmax + 0.5 * df, df)
freq_range = (float(f0[0]), float(f0[-1]), float(f0[1] - f0[0]))

# ---------- HELPERS ----------
def rms_from_psd(psd, f):
    from scipy.integrate import trapezoid
    return float(np.sqrt(trapezoid(psd, f)))

# ---------- FIND FILES ----------
files = []
files += list(DATA_DIR.glob("*.mat"))
files += list(DATA_DIR.glob("*.csv"))
files += list(DATA_DIR.glob("*.txt"))

if not files:
    raise SystemExit(f"No .mat/.csv/.txt files found in {DATA_DIR.resolve()}")

# ---------- PER-FILE PROCESS ----------
durations = []
fds_list = []
srs_abs_list = []
names = []

for path in files:
    # Load
    if path.suffix.lower() == ".mat":
        t, x, fs = vak.load_mat_timesignal(str(path))
    else:
        # .csv or .txt (ASCII)
        t, x, fs = vak.load_ascii_timesignal(str(path))  # auto units detection; override if needed

    duration = len(x) / fs

    # Per-signal SRS (absolute accel) on common grid
    srs_abs = vak.get_srs_abs_accel(x, fs, f0, damp)

    # Per-signal FDS via FatigueDS
    f_fds, fds = vak.get_fds(x, fs, freq_range, damp, b=b)
    if f_fds.shape != f0.shape or not np.allclose(f_fds, f0):
        # Safety: align to our common f0 if needed
        fds = np.interp(f0, f_fds, fds)

    # Collect
    durations.append(duration)
    fds_list.append(fds)
    srs_abs_list.append(srs_abs)
    names.append(path.stem)

    # Optional: per-file plots
    vak.plot_srs(f0, {path.stem: (srs_abs, srs_abs*0)}).write_html(OUT_DIR / f"srs_abs_{path.stem}.html")
    vak.plot_fds(f0, {path.stem: fds}).write_html(OUT_DIR / f"fds_{path.stem}.html")

# ---------- COMPOSITE / LALANNE / ERS ----------
T_eq = float(np.sum(durations))
fds_comp = vak.combine_fds(f0, fds_list, durations)

psd_eq = vak.fds_to_psd(f0, fds_comp, T_eq=T_eq, b=b, Q=Q)
psd_acc = vak.accelerate_psd(psd_eq, T_eq=T_eq, T_acc=T_acc, b=b)

# ERS of accelerated PSD over T_acc
_, ers_acc = vak.get_ers(psd_acc, f0, T_acc, damp, from_psd=True)

# Reference SRS curve: max across missions (absolute-accel)
srs_abs_ref = np.max(np.vstack(srs_abs_list), axis=0)

# ---------- VALIDATION ----------
factor = 2.0
ok, mask = vak.validate_srs_ers(srs_abs_ref, ers_acc, factor=factor)
print(f"[Validation] ERS ≤ {factor}×SRS_abs_ref ?  {'PASS' if ok else 'FAIL'}")
if not ok:
    ratio = ers_acc / (factor * srs_abs_ref + 1e-30)
    bad = np.where(ratio > 1.0)[0]
    worst = bad[np.argsort(ratio[bad])][-5:] if bad.size else []
    if bad.size:
        print("Top violations (freq Hz, ERS/(factor*SRS_abs_ref)):")
        for i in worst[::-1]:
            print(f"{f0[i]:8.2f} Hz -> {ratio[i]:.2f}x")

# ---------- SCALING INFO ----------
rms_eq  = rms_from_psd(psd_eq,  f0)
rms_acc = rms_from_psd(psd_acc, f0)
accel_factor = T_eq / T_acc
print(f"T_eq={T_eq:.1f}s, T_acc={T_acc:.1f}s, accel x={accel_factor:.3g}")
print(f"PSD scale theory: {accel_factor ** (2.0/b):.3g}")
print(f"RMS scale theory: {accel_factor ** (1.0/b):.3g} | measured: {rms_acc / max(rms_eq,1e-30):.3g}")

# ---------- SAVE & PLOT ----------
np.savetxt(OUT_DIR / "f0.csv", np.c_[f0], delimiter=",", header="Hz", comments="")
np.savetxt(OUT_DIR / "fds_composite.csv", np.c_[f0, fds_comp], delimiter=",", header="Hz,FDS", comments="")
np.savetxt(OUT_DIR / "psd_eq_acc.csv", np.c_[f0, psd_eq, psd_acc], delimiter=",", header="Hz,PSD_eq,PSD_acc", comments="")
np.savetxt(OUT_DIR / "srs_abs_ref_and_ers.csv", np.c_[f0, srs_abs_ref, ers_acc], delimiter=",", header="Hz,SRS_abs_ref,ERS_acc", comments="")

vak.plot_fds(f0, {"Composite FDS": fds_comp}).write_html(OUT_DIR / "composite_fds.html")
vak.plot_psd(f0, {"PSD_eq": psd_eq, "PSD_acc": psd_acc}).write_html(OUT_DIR / "psd_eq_acc.html")
vak.plot_ers(f0, {"ERS (acc PSD)": ers_acc}).write_html(OUT_DIR / "ers_acc.html")
vak.plot_srs(f0, {"SRS_abs_ref": (srs_abs_ref, srs_abs_ref*0)}).write_html(OUT_DIR / "srs_abs_ref.html")

print(f"Done. Results saved in: {OUT_DIR.resolve()}")
