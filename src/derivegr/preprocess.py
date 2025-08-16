import cupy as cp
import cupyx.scipy.signal as cpx_signal
from cupy import fft as cpfft
import numpy as np

from .utils_gpu import as_cupy


def whiten_and_bandpass(x_np, fs: float, flow: float, fhigh: float, nperseg: float = 4.0, tukey_alpha: float = 0.1):
    """Whiten by dividing by sqrt(PSD) in freq domain and apply an FFT bandpass mask (GPU-only).

    Parameters
    - x_np: numpy array (time series)
    - fs: sampling rate (Hz)
    - flow, fhigh: bandpass (Hz)
    - nperseg: seconds for Welch segments
    - tukey_alpha: tapering parameter (not yet used; placeholder for future windowing)
    """
    x = as_cupy(x_np)
    n = x.shape[0]

    # Welch PSD on GPU
    seg = int(max(256, int(nperseg * fs)))
    f_psd, Pxx = cpx_signal.welch(x, fs=fs, window=('hann'), nperseg=seg, noverlap=seg // 2, detrend=False, return_onesided=True, scaling='density')

    # FFT of signal
    X = cpfft.rfft(x)
    freqs = cpfft.rfftfreq(n, d=1.0 / fs)

    # Interpolate PSD to FFT bins
    Pxx_interp = cp.interp(freqs, f_psd, Pxx)
    eps = 1e-18
    W = 1.0 / cp.sqrt(cp.maximum(Pxx_interp, eps))

    # Bandpass mask
    mask = (freqs >= flow) & (freqs <= fhigh)

    Xw = X * W * mask
    xw = cpfft.irfft(Xw, n=n)
    return xw

