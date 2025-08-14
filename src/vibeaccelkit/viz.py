import numpy as np
import plotly.graph_objs as go
from scipy.integrate import trapezoid

def _rms_from_psd(psd, f): return float(np.sqrt(trapezoid(psd, f)))

def plot_fds(freqs, curves, title="FDS"):
    fig = go.Figure()
    for label, arr in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=arr, mode="lines", name=label))
    fig.update_layout(title=title, xaxis_title="Frequency [Hz]", yaxis_title="FDS", yaxis_type="log",
                      template="plotly_white")
    return fig

def plot_ers(freqs, curves, title="ERS"):
    fig = go.Figure()
    for label, arr in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=arr, mode="lines", name=label))
    fig.update_layout(title=title, xaxis_title="Frequency [Hz]", yaxis_title="ERS", yaxis_type="log",
                      template="plotly_white")
    return fig

def plot_srs(freqs, curves, title="SRS"):
    fig = go.Figure()
    for label, (pos, neg) in curves.items():
        fig.add_trace(go.Scatter(x=freqs, y=pos, mode="lines", name=f"{label} SRS+"))
        fig.add_trace(go.Scatter(x=freqs, y=neg, mode="lines", name=f"{label} SRS-"))
    fig.update_layout(title=title, xaxis_title="Frequency [Hz]", yaxis_title="SRS amplitude", yaxis_type="log",
                      template="plotly_white")
    return fig

def plot_psd(freqs, curves, title="PSD"):
    fig = go.Figure()
    for label, psd in curves.items():
        rms = _rms_from_psd(psd, freqs)
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name=f"{label} (RMS={rms:.2f})"))
    fig.update_layout(title=title, xaxis_title="Frequency [Hz]", yaxis_title="PSD [(m/s^2)^2/Hz]",
                      yaxis_type="log", template="plotly_white")
    return fig
