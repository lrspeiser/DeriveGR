import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import h5py
from gwosc.datasets import event_gps
from gwosc.locate import get_urls
from urllib.request import urlretrieve


def get_default_cache_dir() -> Path:
    # Windows default: %LOCALAPPDATA%\DeriveGR\gwosc, else XDG cache
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "DeriveGR" / "gwosc"


def resolve_event_gps(event: str) -> float:
    return float(event_gps(event))


def ensure_cached(event: str, ifo: str, tpad: float, cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gps = resolve_event_gps(event)
    start = int(gps) - int(tpad)
    end = int(gps) + int(tpad)
    urls = get_urls(
        instrument=ifo,
        start=start,
        end=end,
        tag="LOSC",
    )
    if not urls:
        raise RuntimeError(f"No GWOSC URLs found for {ifo} {event} [{start}, {end}]")
    # Pick the first URL
    url = urls[0]
    fname = f"{event}_{ifo}_{start}_{end}.hdf5"
    out = cache_dir / fname
    if not out.exists():
        urlretrieve(url, out)
    return out


def read_ifo_timeseries(hdf5_path: Path) -> Tuple[np.ndarray, Dict]:
    """Read LOSC HDF5 file; return strain time series and metadata (fs).

    The LOSC files contain datasets under e.g., strain/Strain and metadata describing dt or fs.
    """
    with h5py.File(hdf5_path, "r") as f:
        # Common LOSC structure: f["strain"]["Strain"] and f["strain"].attrs["Xspacing"] (seconds)
        if "strain" in f and "Strain" in f["strain"]:
            dset = f["strain"]["Strain"]
            data = dset[...].astype(np.float64)
            # sampling from attribute
            dt = None
            if "Xspacing" in f["strain"].attrs:
                dt = float(f["strain"].attrs["Xspacing"])  # seconds
            elif "sampling_rate" in f["strain"].attrs:
                fs = float(f["strain"].attrs["sampling_rate"])  # Hz
                dt = 1.0 / fs
            else:
                # Fallback: infer from length vs. timeseries length if time vector exists
                dt = None
            if dt is None:
                # Try to infer from top-level meta
                fs = 4096.0
            else:
                fs = 1.0 / dt
        else:
            raise RuntimeError("Unexpected HDF5 structure; expected 'strain/Strain'.")
    meta = {"fs": fs}
    return data, meta

