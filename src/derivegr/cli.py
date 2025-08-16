import os
import json
import math
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from . import utils_gpu as gpu
from . import data_gwosc
from . import preprocess
from . import hilbert_freq
from . import fit_powerlaw
from . import chirp_mass
from . import plotting
from . import io as io_utils

app = typer.Typer(no_args_is_help=True, help="Derive GR inspiral law and chirp mass from GWOSC data (GPU-only, CuPy)")
console = Console()


def _default_out_dir(event: str) -> Path:
    return Path("outputs") / event.lower()


@app.command()
def prefetch(
    event: str = typer.Option("GW150914", help="Event name, e.g., GW150914"),
    ifo: List[str] = typer.Option(["H1", "L1"], help="Interferometers to fetch (repeatable)"),
    tpad: float = typer.Option(16.0, help="Seconds of padding on either side of the event"),
    cache: Optional[Path] = typer.Option(None, help="Override cache directory (defaults to %LOCALAPPDATA%/DeriveGR/gwosc)"),
):
    """Download and cache GWOSC data for an event."""
    cache_dir = cache or data_gwosc.get_default_cache_dir()
    console.log(f"Cache directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for d in ifo:
        path = data_gwosc.ensure_cached(event, d, tpad, cache_dir)
        console.log(f"Cached {d}: {path}")


@app.command()
def discover_inspiral(
    event: str = typer.Option("GW150914", help="Event name, e.g., GW150914"),
    ifo: List[str] = typer.Option(["H1", "L1"], help="Interferometers to use (repeatable)"),
    tpad: float = typer.Option(16.0, help="Seconds of padding on either side of the event"),
    flow: float = typer.Option(30.0, help="Low cutoff for bandpass (Hz)"),
    fhigh: float = typer.Option(350.0, help="High cutoff for bandpass (Hz)"),
    smooth_sec: float = typer.Option(0.05, help="Smoothing window in seconds for instantaneous frequency"),
    out: Optional[Path] = typer.Option(None, help="Output directory (default: outputs/<event>)"),
    gpu_device: int = typer.Option(0, help="CUDA device index"),
    save_plots: bool = typer.Option(True, help="Save PNG plots"),
    save_csv: bool = typer.Option(True, help="Save CSV of finst and dfdt and fit results"),
    symbolic: bool = typer.Option(False, help="Try optional symbolic discovery (CPU/Julia)"),
):
    """End-to-end pipeline: fetch, preprocess, discover df/dt ~ K f^alpha, recover chirp mass."""
    # GPU setup
    dev = gpu.get_device(gpu_device)
    info = gpu.gpu_info()
    console.log(f"Using GPU: {info['name']} (CC {info['cc']}, {info['total_mem_gb']:.1f} GiB)")

    # Fetch data
    cache_dir = data_gwosc.get_default_cache_dir()
    series = []
    meta = None
    for d in ifo:
        path = data_gwosc.ensure_cached(event, d, tpad, cache_dir)
        s, m = data_gwosc.read_ifo_timeseries(path)
        series.append(s)
        meta = m
    fs = meta["fs"]

    # Align and average
    min_len = min(len(s) for s in series)
    series = [s[:min_len] for s in series]
    import numpy as np
    h = np.mean(np.stack(series, axis=0), axis=0)

    # GPU preprocess
    xw = preprocess.whiten_and_bandpass(h, fs=fs, flow=flow, fhigh=fhigh)

    # Instantaneous frequency and df/dt
    t, finst, dfdt = hilbert_freq.instantaneous_frequency_and_dfdt(xw, fs=fs, smooth_sec=smooth_sec)

    # Fit power law in a conservative band
    fit = fit_powerlaw.fit_powerlaw(finst, dfdt, flow=40.0, fhigh=180.0)

    # Chirp mass
    cm = chirp_mass.from_K(fit["K"])  # returns dict with kg and solar

    # Outputs
    out_dir = out or _default_out_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    if save_csv:
        io_utils.write_csv(out_dir / "finst.csv", ["t", "f_inst"], t, finst)
        io_utils.write_csv(out_dir / "dfdt.csv", ["t", "dfdt"], t, dfdt)

    # Fit results JSON
    results = {
        "event": event,
        "fs": fs,
        "flow": flow,
        "fhigh": fhigh,
        "alpha": float(fit["alpha"]),
        "K": float(fit["K"]),
        "alpha_stderr": float(fit.get("alpha_stderr", math.nan)),
        "K_stderr": float(fit.get("K_stderr", math.nan)),
        "Mc_kg": float(cm["Mc_kg"]),
        "Mc_solar": float(cm["Mc_solar"]),
        "gpu": info,
    }
    (out_dir / "fit_results.json").write_text(json.dumps(results, indent=2))

    # Plots
    if save_plots:
        plotting.plot_finst(t, finst, flow, fhigh, out_dir / "finst.png")
        plotting.plot_dfdt(t, dfdt, out_dir / "dfdt.png")
        plotting.plot_loglog_fit(finst, dfdt, fit, cm, out_dir / "loglog_fit.png")

    # Console table
    tbl = Table(title=f"{event} discovery results")
    tbl.add_column("Metric")
    tbl.add_column("Value")
    tbl.add_row("alpha", f"{fit['alpha']:.4f} (GR ~ 3.6667)")
    tbl.add_row("K (SI)", f"{fit['K']:.6e}")
    tbl.add_row("Mc (M_sun)", f"{cm['Mc_solar']:.2f}")
    console.print(tbl)


@app.command()
def chirp_mass(
    from_json: Optional[Path] = typer.Option(None, help="Path to fit_results.json"),
    K: Optional[float] = typer.Option(None, help="Coefficient K from df/dt = K f^alpha"),
    alpha: Optional[float] = typer.Option(None, help="Fitted alpha (informational)"),
):
    """Compute chirp mass from K (and optionally alpha)."""
    if from_json:
        data = json.loads(Path(from_json).read_text())
        K = data["K"]
    if K is None:
        raise typer.BadParameter("Provide either --from-json or --K")
    cm = chirp_mass.from_K(K)
    print({"Mc_kg": cm["Mc_kg"], "Mc_solar": cm["Mc_solar"]})


if __name__ == "__main__":
    app()

