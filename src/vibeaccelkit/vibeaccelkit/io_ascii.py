# src/vibeaccelkit/io_ascii.py
import re
from io import StringIO
import numpy as np

G0 = 9.80665  # m/s^2 per g

def _is_comment(line: str) -> bool:
    s = line.lstrip()
    return s.startswith(("#", "%", ";", "!", "//"))

def load_ascii_timesignal(
    path: str,
    time_col: int = 0,
    accel_col: int = 1,
    units: str = "auto",
    tol_rel_dt: float = 1e-3,
):
    """
    Load ASCII text with time and acceleration columns.
    - Handles commas, semicolons, tabs, or whitespace as delimiters.
    - Skips comment/header lines (starting with #, %, ;, !, //).
    - Units:
        units='auto' tries to detect 'mg' or 'g' in headers; otherwise assumes m/s^2.
        units='g'   -> multiply by 9.80665
        units='mg'  -> multiply by 9.80665/1000
        units='m/s^2' or 'ms2' -> no change
    - Ensures (or resamples to) uniform sampling if time spacing is slightly non-uniform.

    Returns: (time, accel_mps2, fs)
    """
    header_lines = []
    data_lines = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                header_lines.append(line)
                continue
            if _is_comment(line):
                header_lines.append(line)
                continue
            # normalize delimiters to spaces
            s = line.replace(",", " ").replace(";", " ").replace("\t", " ")
            s = re.sub(r"\s+", " ", s).strip()
            if not s:
                header_lines.append(line)
                continue
            toks = s.split(" ")
            # must have at least two numeric columns
            try:
                float(toks[0]); float(toks[1])
                data_lines.append(s)
            except Exception:
                header_lines.append(line)

    if not data_lines:
        raise ValueError(f"No numeric data lines found in {path}")

    arr = np.loadtxt(StringIO("\n".join(data_lines)))
    if arr.ndim == 1:  # single row?
        arr = arr.reshape(1, -1)
    if arr.shape[1] <= max(time_col, accel_col):
        raise ValueError(f"Not enough columns in {path}; got {arr.shape[1]} columns")

    t = arr[:, time_col].astype(float)
    a = arr[:, accel_col].astype(float)

    # Unit handling
    header_text = " ".join(header_lines).lower()
    if units == "auto":
        if re.search(r"\bmg\b", header_text):
            a = a * (G0 / 1000.0)
        elif re.search(r"\bg\b", header_text):
            a = a * G0
        # elif header mentions m/s^2, do nothing
    elif units == "g":
        a = a * G0
    elif units == "mg":
        a = a * (G0 / 1000.0)
    # else assume already m/s^2

    # Ensure uniform sampling (or resample if slightly non-uniform)
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0:
        raise ValueError("Non-increasing or invalid time column.")
