from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_finst(t: np.ndarray, finst: np.ndarray, flow: float, fhigh: float, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(t, finst, lw=1)
    plt.axhspan(flow, fhigh, color='orange', alpha=0.1, label='bandpass')
    plt.xlabel('Time (s)')
    plt.ylabel('Instantaneous frequency (Hz)')
    plt.title('Instantaneous frequency')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dfdt(t: np.ndarray, dfdt: np.ndarray, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(t, dfdt, lw=1)
    plt.xlabel('Time (s)')
    plt.ylabel('df/dt (Hz/s)')
    plt.title('Frequency derivative')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loglog_fit(finst: np.ndarray, dfdt: np.ndarray, fit: dict, cm: dict, out_path: Path):
    mask = np.isfinite(finst) & np.isfinite(dfdt) & (finst > 0) & (dfdt > 0)
    f = finst[mask]
    d = dfdt[mask]
    x = np.log(f)
    y = np.log(d)
    a = fit['alpha']
    b = fit['intercept']

    xx = np.linspace(x.min(), x.max(), 200)
    yy = a * xx + b

    plt.figure(figsize=(5.5, 5))
    plt.scatter(x, y, s=6, alpha=0.5, label='data')
    plt.plot(xx, yy, 'r-', lw=2, label=f'fit: alpha={a:.4f}')
    plt.xlabel('log f')
    plt.ylabel('log df/dt')
    plt.title(f"Power-law fit (Mc ≈ {cm['Mc_solar']:.1f} M_sun)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

