import numpy as np
import vibeaccelkit as vak

def main():
    # --- 0) Config ---
    fs = 2048.0
    duration = 20.0
    damp = 0.05
    b = 7.0
    Q = 1.0 / (2.0 * damp)

    # --- 1) Define a simple one-sided PSD (no files needed) ---
    fgrid = np.geomspace(5.0, 500.0, 400)
    psd = 1e-3 * (fgrid / 50.0) ** (-2.0)          # ~1/f^2 inside band
    psd[(fgrid < 8.0) | (fgrid > 400.0)] = 1e-12   # taper outside

    # --- 2) Synthesize a time signal from PSD ---
    x = vak.synthesize_time_from_psd(fgrid, psd, fs=fs, duration=duration, seed=7)

    # --- 3) SRS (always available) ---
    fn = np.geomspace(8.0, 400.0, 120)
    srs_pos, srs_neg = vak.get_srs(x, fs, fn, damp)

    # --- 4) Optional parts (need FatigueDS) ---
    have_fatigue = True
    try:
        import FatigueDS  # noqa: F401
    except Exception:
        have_fatigue = False
        print("[Note] FatigueDS not installed; skipping FDS/ERS/Lalanne demo. "
              "Install with: pip install -e .[fatigue]")

    if have_fatigue:
        # FDS from time
        f0, fds = vak.get_fds(x, fs, (float(fn[0]), float(fn[-1]), float(np.diff(fn).mean())), damp, b=b)

        # Fake a second mission (milder) and make a composite FDS (no files)
        fds_comp = vak.combine_fds(f0, [fds, 0.7 * fds], [10.0, 10.0])
        T_eq = 20.0

        # Lalanne FDS→PSD (equivalent) then accelerate to shorter duration
        psd_eq = vak.fds_to_psd(f0, fds_comp, T_eq=T_eq, b=b, Q=Q)
        psd_acc = vak.accelerate_psd(psd_eq, T_eq=T_eq, T_acc=5.0, b=b)

        # ERS from accelerated PSD
        _, ers_acc = vak.get_ers(psd_acc, f0, 5.0, damp, from_psd=True)

        # Validation: ERS ≤ 2×SRS+
        ok, mask = vak.validate_srs_ers(srs_pos, ers_acc, factor=2.0)
        print(f"[Validation] ERS ≤ 2×SRS+ ?  {'PASS' if ok else 'FAIL'}")
    else:
        f0 = fn
        fds = fds_comp = psd_eq = psd_acc = ers_acc = None

    # --- 5) Plots (they return figures; you decide when to show) ---
    vak.plot_psd(fgrid, {"PSD (design)": psd}).show()
    vak.plot_srs(fn, {"Synth signal": (srs_pos, srs_neg)}).show()
    if fds is not None:
        vak.plot_fds(f0, {"FDS (from time)": fds, "Composite FDS": fds_comp}).show()
        vak.plot_psd(f0, {"PSD_eq (from FDS)": psd_eq, "PSD_acc (5 s)": psd_acc}).show()
        vak.plot_ers(f0, {"ERS (from PSD_acc)": ers_acc}).show()

if __name__ == "__main__":
    main()
