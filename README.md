# VibeAccelKit

VibeAccelKit is a Python toolkit for vibration test tailoring, mission synthesis, and test acceleration.  
It computes Fatigue Damage Spectra (FDS), Energy Response Spectra (ERS), Shock Response Spectra (SRS),  
composite spectra, and generates accelerated PSD profiles, with optional PSD-to-time synthesis.

## Features
- Read multiple `.mat` time history files
- Compute FDS (convolution), ERS, and SRS per signal
- Combine into composite FDS
- Convert FDS → PSD using Lalanne’s method
- Accelerate PSD to target durations
- Optional PSD → time synthesis
- Interactive Plotly plots + optional SVG export

## Installation
```bash
pip install -e .
# optional FatigueDS-backed features:
pip install -e .[fatigue]

## Quickstart

```python
import numpy as np
import vibeaccelkit as vak

fs = 2048.0
duration = 20.0
damp = 0.05
b = 7.0
Q = 1.0 / (2.0 * damp)

# PSD (no data files)
fgrid = np.geomspace(5.0, 500.0, 400)
psd = 1e-3 * (fgrid / 50.0) ** -2
psd[(fgrid < 8.0) | (fgrid > 400.0)] = 1e-12

# Synthesize + SRS
x = vak.synthesize_time_from_psd(fgrid, psd, fs=fs, duration=duration, seed=7)
fn = np.geomspace(8.0, 400.0, 120)
srs_pos, srs_neg = vak.get_srs(x, fs, fn, damp)

# Optional (requires FatigueDS):
# f0, fds = vak.get_fds(x, fs, (float(fn[0]), float(fn[-1]), float(np.diff(fn).mean())), damp, b=b)
