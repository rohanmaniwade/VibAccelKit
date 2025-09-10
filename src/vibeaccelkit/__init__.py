from .io_mat import load_mat_timesignal
from .fds import get_fds, get_ers, fds_from_psd, rms_from_psd, time_to_psd
from .srs import get_srs
from .composite import combine_fds, fds_envelope
from .lalanne import fds_to_psd, accelerate_psd
from .synth import synthesize_time_from_psd
from .validate import validate_srs_ers, validate_fds_meets_target
from .viz import plot_fds, plot_ers, plot_srs, plot_psd, plot_srs_vs_ers
from .stats import time_rms, psd_rms, welch_psd


__version__ = "0.1.0"

__all__ = [
    "load_mat_timesignal",
    "get_fds", "get_ers", "time_to_psd"
    "get_srs",
    "combine_fds", "fds_envelope",
    "fds_to_psd", "accelerate_psd",
    "synthesize_time_from_psd",
    "validate_srs_ers", "validate_fds_meets_target",
    "plot_fds", "plot_ers", "fds_from_psd", "rms_from_psd", "plot_srs", "plot_psd",
    "plot_srs_vs_ers",
    "time_rms", "psd_rms", "welch_psd",

    
]
