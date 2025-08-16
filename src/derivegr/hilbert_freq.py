import cupy as cp
import cupyx.scipy.signal as cpx_signal
import numpy as np


def instantaneous_frequency_and_dfdt(xw: cp.ndarray, fs: float, smooth_sec: float = 0.05):
    """Compute instantaneous frequency and its time derivative on GPU.

    - xw: whitened, bandpassed cp.ndarray
    - fs: sampling rate (Hz)
    - smooth_sec: smoothing window in seconds for f_inst
    Returns: (t_np, f_inst_np, dfdt_np) as numpy arrays for downstream IO/plotting
    """
    n = xw.shape[0]
    # Analytic signal via Hilbert transform
    z = cpx_signal.hilbert(xw)
    phase = cp.unwrap(cp.angle(z))
    finst = (fs / (2.0 * np.pi)) * cp.gradient(phase)

    # Smooth instantaneous frequency
    w = max(3, int(smooth_sec * fs) | 1)  # odd window length
    if w > 3:
        kernel = cp.ones(w, dtype=cp.float64) / w
        finst = cp.convolve(finst, kernel, mode='same')

    # df/dt
    dfdt = cp.gradient(finst, 1.0 / fs)

    t = cp.arange(n, dtype=cp.float64) / fs
    return cp.asnumpy(t), cp.asnumpy(finst), cp.asnumpy(dfdt)

