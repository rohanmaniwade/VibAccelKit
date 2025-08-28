import numpy as np
import plotly.graph_objs as go
from scipy.integrate import trapezoid

def _rms_from_psd(psd, f): return float(np.sqrt(trapezoid(psd, f)))

def _decade_ticks(freqs):
    """Return (tickvals, ticktext) at powers of 10 within freqs range."""
    f = np.asarray(freqs, float)
    f = f[f > 0]
    if f.size == 0:
        return None, None
    fmin, fmax = np.min(f), np.max(f)
    n0 = int(np.ceil(np.log10(fmin)))
    n1 = int(np.floor(np.log10(fmax)))
    vals = [10.0 ** k for k in range(n0, n1 + 1)]
    labels = [f"{v:g}" for v in vals]
    return vals, labels

def _apply_log_axes(fig, logx, logy, freqs):
    fig.update_layout(
        xaxis_type="log" if logx else "linear",
        yaxis_type="log" if logy else "linear",
    )
    if logx:
        tickvals, ticktext = _decade_ticks(freqs)
        if tickvals:
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

def plot_fds(freqs, curves, title="FDS", logx=True, logy=True):
    fig = go.Figure()
    for label, arr in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=arr, mode="lines", name=label))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="FDS",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs)
    return fig

def plot_ers(freqs, curves, title="ERS", logx=True, logy=True):
    fig = go.Figure()
    for label, arr in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=arr, mode="lines", name=label))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="ERS",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs)
    return fig

def plot_srs(freqs, curves, title="SRS", logx=True, logy=True):
    fig = go.Figure()
    for label, (pos, neg) in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=pos, mode="lines", name=f"{label} SRS+"))
        fig.add_trace(go.Scatter(x=freqs, y=neg, mode="lines", name=f"{label} SRS-"))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="SRS amplitude",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs)
    return fig

def plot_psd(freqs, curves, title="PSD", logx=True, logy=True):
    fig = go.Figure()
    for label, psd in curves.items():
        rms = _rms_from_psd(psd, freqs)
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name=f"{label} (RMS={rms:.2f})"))

    fig.update_layout(
        title=title,
        xaxis_title="Frequency [Hz]",
        yaxis_title="PSD [(m/s^2)^2/Hz]",
        template="plotly_white")
    _apply_log_axes(fig, logx, logy, freqs)
    return fig
