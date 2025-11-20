import numpy as np
import vibeaccelkit as vak
import plotly.io as pio

# Show plots in your browser (avoids nbformat errors)
pio.renderers.default = "browser"

def rms_from_psd(psd, f):
    from scipy.integrate import trapezoid
    return float(np.sqrt(trapezoid(psd, f)))

def main():
    # --- 0) Config ---
    fs = 2048.0
    duration = 20.0
    damp = 0.05
    b = 7.0
    Q = 1.0 / (2.0 * damp)

    # knobs you can tweak
    T_acc = 20.0          # accelerated test duration (seconds)
    srs_ers_factor = 2.0 # ERS must be <= factor * SRS_acc

    # --- 1) ONE frequency grid for EVERYTHING (linear for FatigueDS) ---
    fmin, fmax, df = 8.0, 400.0, 4.0
    f0 = np.arange(fmin, fmax + 0.5 * df, df)  # inclusive end

    # --- 2) PSD ON THE SAME GRID (no files) ---
    psd = 1e-3 * (np.maximum(f0, 1.0) / 50.0) ** (-2.0)  # ~1/f^2 in-band
    psd = np.maximum(psd, 1e-12)  # tiny floor

    # --- 3) Synthesize a time signal from PSD ---
    x = vak.synthesize_time_from_psd(f0, psd, fs=fs, duration=duration, seed=7)

    # --- 4) SRS (displacement) on the SAME GRID, then convert to acceleration SRS ---
    srs_pos, srs_neg = vak.get_srs(x, fs, f0, damp)     # displacement peaks
    omega = 2.0 * np.pi * f0
    srs_acc_pos = (omega ** 2) * srs_pos                # acceleration peaks (relative)

    # --- 5) Optional parts (need FatigueDS) ---
    try:
        import FatigueDS  # noqa: F401

        # FDS from time
        freq_range = (float(f0[0]), float(f0[-1]), float(f0[1] - f0[0]))
        f_fds, fds = vak.get_fds(x, fs, freq_range, damp, b=b)

        # Align to f0 if the backend quantized differently
        if f_fds.shape != f0.shape or not np.allclose(f_fds, f0):
            fds = np.interp(f0, f_fds, fds)

        # Composite FDS (fake second mission)
        fds_comp = vak.combine_fds(f0, [fds, 0.7 * fds], [10.0, 10.0])
        T_eq = 20.0

        # Lalanne FDS→PSD (equivalent) and acceleration to T_acc
        psd_eq = vak.fds_to_psd(f0, fds_comp, T_eq=T_eq, b=b, Q=Q)
        psd_acc = vak.accelerate_psd(psd_eq, T_eq=T_eq, T_acc=T_acc, b=b)

        # Print acceleration info
        accel_factor = T_eq / T_acc
        psd_scale_theory = accel_factor ** (2.0 / b)
        rms_eq = rms_from_psd(psd_eq, f0)
        rms_acc = rms_from_psd(psd_acc, f0)
        print(f"T_eq={T_eq:.3g}s, T_acc={T_acc:.3g}s, acceleration x={accel_factor:.3g}")
        print(f"PSD scale theory: {psd_scale_theory:.3g}")
        print(f"RMS scale theory: {accel_factor ** (1.0 / b):.3g} | measured: {rms_acc / max(rms_eq, 1e-30):.3g}")

        # ERS from accelerated PSD (same grid)
        _, ers_acc = vak.get_ers(psd_acc, f0, T_acc, damp, from_psd=True)

        # --- Validation: ERS ≤ factor × SRS_acc (same grid) ---
        ok, mask = vak.validate_srs_ers(srs_acc_pos, ers_acc, factor=srs_ers_factor)
        print(f"[Validation] ERS ≤ {srs_ers_factor}×SRS_acc ?  {'PASS' if ok else 'FAIL'}")

        # (Optional) print worst offenders if it fails
        if not ok:
            ratio = ers_acc / (srs_ers_factor * srs_acc_pos + 1e-30)
            bad = np.where(ratio > 1.0)[0]
            worst = bad[np.argsort(ratio[bad])][-5:] if bad.size else []
            if bad.size:
                print("Top violations (freq Hz, ERS/(factor*SRS_acc)):")
                for i in worst[::-1]:
                    print(f"{f0[i]:8.2f} Hz -> {ratio[i]:.2f}x")

    except Exception:
        print("[Note] FatigueDS not installed; skipping FDS/ERS/Lalanne demo. "
              "Install with: pip install -e .[fatigue]")
        fds = fds_comp = psd_eq = psd_acc = ers_acc = None

    # --- 6) Plots ---
    rms_design = vak.psd_rms(f0, psd)
    vak.plot_psd(f0, {"PSD (design)": (psd, rms_design)}).show()
    vak.plot_srs(f0, {"Synth signal": (srs_pos, srs_neg)}).show()
    try:
        if fds is not None:
            vak.plot_fds(f0, {"FDS (from time)": fds, "Composite FDS": fds_comp}).show()
            vak.plot_psd(f0, {
                "PSD_eq (from FDS)": (psd_eq, rms_eq),
                "PSD_acc": (psd_acc, rms_acc)
            }).show()
            vak.plot_ers(f0, {"ERS (from PSD_acc)": ers_acc}).show()
    except NameError:
        pass

if __name__ == "__main__":
    main()
