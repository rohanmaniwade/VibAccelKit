from .io_mat import load_mat_timesignal
from .fds import get_fds_time, get_fds_psd, get_fds, fds_from_psd, rms_from_psd, time_to_psd
from .ers import get_ers
from .srs import get_srs
from .composite import combine_fds, fds_envelope
from .lalanne import fds_to_psd, accelerate_psd
from .synth import synthesize_time_from_psd
from .validate import validate_srs_ers, validate_fds_meets_target
from .viz import plot_fds, plot_ers, plot_srs, plot_psd, plot_srs_vs_ers
from .stats import time_rms, psd_rms, welch_psd, make_freq_grid


__version__ = "0.1.0"

# Public API
__all__ = [
    "load_mat_timesignal",
    "get_fds",
    "get_fds_time",
    "get_fds_psd",
    "get_ers",
    "get_srs",
    "time_to_psd",
    "fds_from_psd",
    "rms_from_psd",
    "combine_fds",
    "fds_envelope",
    "fds_to_psd",
    "accelerate_psd",
    "synthesize_time_from_psd",
    "validate_srs_ers",
    "validate_fds_meets_target",
    "plot_fds",
    "plot_ers",
    "plot_srs",
    "plot_psd",
    "plot_srs_vs_ers",
    "time_rms",
    "psd_rms",
    "welch_psd",
    "make_freq_grid",
]
