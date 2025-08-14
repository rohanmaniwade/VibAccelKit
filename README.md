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

