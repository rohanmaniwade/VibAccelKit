import numpy as np

def synthesize_time_from_psd(freqs, psd, fs, duration, seed=0):
    """
    Random-phase IFFT synthesis of a zero-mean Gaussian time history
    whose one-sided PSD approximately matches (freqs, psd).
    - freqs: 1D array (Hz), monotonically increasing, starts near 0
    - psd:   1D array, same length as freqs, units of (m/s^2)^2/Hz
    - fs:    sample rate (Hz)
    - duration: seconds
    Returns: time_signal (np.ndarray, length N=int(duration*fs))
    """
    rng = np.random.default_rng(seed)
    N = int(np.round(duration * fs))
    if N < 2:
        raise ValueError("duration*fs too small")

    df = fs / N
    kmax = N // 2
    f_bins = np.arange(0, kmax + 1) * df

    # Interpolate one-sided PSD to FFT bin centers
    psd_bins = np.interp(f_bins, freqs, psd, left=0.0, right=0.0)

    # Amplitude per bin for real IFFT (one-sided)
    amp = np.sqrt(psd_bins * df)

    # Random phases
    phi = rng.uniform(0, 2*np.pi, size=amp.shape)

    X_half = amp * np.exp(1j * phi)
    X_half[0] = 0.0  # no DC
    if N % 2 == 0:
        X_half[-1] = 0.0  # Nyquist bin real-only, set to 0 for safety

    # Build full spectrum with Hermitian symmetry
    X = np.zeros(N, dtype=complex)
    X[:kmax+1] = X_half
    X[kmax+1:] = np.conj(X_half[1:kmax][::-1])

    x = np.fft.ifft(X).real * N
    return x
