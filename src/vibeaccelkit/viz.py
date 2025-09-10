import numpy as np
import plotly.graph_objs as go
from scipy.integrate import trapezoid
import math

def _rms_from_psd(psd, f): return float(np.sqrt(trapezoid(psd, f)))


def _ticks_powers_with_end(freqs, include_end=True):
    """Make log-x ticks at decades (10^n) and optionally append the exact fmax."""
    f = np.asarray(freqs, float)
    f = f[f > 0]
    if f.size == 0:
        return None, None
    fmin, fmax = float(f.min()), float(f.max())

    n0 = int(math.ceil(math.log10(fmin)))
    n1 = int(math.floor(math.log10(fmax)))
    decade_vals = [10.0 ** k for k in range(n0, n1 + 1)]
    # label decades as 10^n using HTML superscripts
    decade_text = [f"10<sup>{k}</sup>" for k in range(n0, n1 + 1)]

    tickvals = decade_vals[:]
    ticktext = decade_text[:]

    if include_end:
        # append the exact max frequency if it's not already a decade tick
        if not np.isclose(fmax, decade_vals[-1] if decade_vals else -1.0):
            tickvals.append(fmax)
            # show it as a plain number (e.g., 500, 2000)
            ticktext.append(f"{int(round(fmax))}" if fmax >= 10 else f"{fmax:g}")

    return tickvals, ticktext

def _apply_log_axes(fig, logx, logy, freqs, include_end_tick=True):
    fig.update_layout(
        xaxis_type="log" if logx else "linear",
        yaxis_type="log" if logy else "linear",
    )
    if logx:
        tv, tt = _ticks_powers_with_end(freqs, include_end=include_end_tick)
        if tv:
            fig.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)

def plot_fds(freqs, curves, title="FDS", logx=True, logy=True, include_end_tick=True):
    fig = go.Figure()
    for label, arr in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=arr, mode="lines", name=label))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="FDS",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs, include_end_tick)
    return fig

def plot_ers(freqs, curves, title="ERS", logx=True, logy=True, include_end_tick=True):
    fig = go.Figure()
    for label, arr in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=arr, mode="lines", name=label))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="ERS [m/s²]",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs, include_end_tick)
    return fig

def plot_srs(freqs, curves, title="Shock Response Spectrum", logx=True, logy=True, include_end_tick=True):
    fig = go.Figure()
    for label, (pos, neg) in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=pos, mode="lines", name=f"{label} SRS+"))
        fig.add_trace(go.Scatter(x=freqs, y=neg, mode="lines", name=f"{label} SRS-"))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="SRS [m/s²]",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs, include_end_tick)
    return fig

def plot_psd(freqs, curves, title="PSD", logx=True, logy=True, include_end_tick=True):
    from scipy.integrate import trapezoid
    fig = go.Figure()
    for label, psd in curves.items():
        rms = float(np.sqrt(trapezoid(np.maximum(psd, 0.0), freqs)))
        # PSD curve
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name=f"{label} (RMS={rms:.2f})"))
        # RMS horizontal line
        fig.add_trace(go.Scatter(
            x=[freqs[0], freqs[-1]],
            y=[rms, rms],
            mode="lines",
            line=dict(dash="dot"),
            name=f"{label} RMS line",
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="Acceleration PSD (m/s²)²/Hz",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs, include_end_tick)
    return fig

def plot_srs_vs_ers(f, srs_plus, ers, factor=2.0, title="SRS⁺ vs ERS (Validation)"):
    f = np.asarray(f, float)
    srs_plus = np.asarray(srs_plus, float)
    ers = np.asarray(ers, float)

    limit = factor * srs_plus

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=srs_plus, mode="lines", name="SRS⁺"))
    fig.add_trace(go.Scatter(x=f, y=limit, mode="lines", name=f"{factor:g}×SRS⁺ limit", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=f, y=ers, mode="lines", name="ERS"))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="Acceleration [m/s²]",
        template="plotly_white"
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    return fig
