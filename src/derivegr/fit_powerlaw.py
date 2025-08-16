import cupy as cp
import numpy as np
from typing import Dict


def fit_powerlaw(finst_np: np.ndarray, dfdt_np: np.ndarray, flow: float = 40.0, fhigh: float = 180.0) -> Dict:
    """Fit df/dt ~ K * f^alpha in log–log space on GPU (CuPy).

    Returns dict with alpha, K, intercept, and simple diagnostics.
    """
    f = cp.asarray(finst_np)
    fd = cp.asarray(dfdt_np)
    mask = cp.isfinite(f) & cp.isfinite(fd) & (f > 0) & (fd > 0) & (f >= flow) & (f <= fhigh)
    f = f[mask]
    fd = fd[mask]

    x = cp.log(f)
    y = cp.log(fd)
    n = x.size
    if n < 10:
        raise RuntimeError("Insufficient valid samples after masking; adjust flow/fhigh or smoothing.")

    # Unweighted least squares: y = a*x + b
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = cp.sum((x - x_mean) ** 2)
    Sxy = cp.sum((x - x_mean) * (y - y_mean))
    alpha = Sxy / Sxx
    b = y_mean - alpha * x_mean
    K = cp.exp(b)

    # Diagnostics
    yhat = alpha * x + b
    rss = cp.sum((y - yhat) ** 2)
    tss = cp.sum((y - y_mean) ** 2)
    r2 = 1.0 - (rss / tss)

    # Standard errors (approx):
    sigma2 = rss / (n - 2)
    alpha_stderr = cp.sqrt(sigma2 / Sxx)
    K_stderr = K * cp.sqrt(sigma2 * (1.0 / n + x_mean**2 / Sxx))

    return {
        "alpha": float(alpha.get()),
        "K": float(K.get()),
        "intercept": float(b.get()),
        "R2": float(r2.get()),
        "RSS": float(rss.get()),
        "alpha_stderr": float(alpha_stderr.get()),
        "K_stderr": float(K_stderr.get()),
        "n": int(n),
    }

