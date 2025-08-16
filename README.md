# DeriveGR

Discover the leading-order GR inspiral law directly from LIGO open data and recover the chirp mass — fast, reproducible, and GPU-accelerated.

DeriveGR is a Windows-first, CuPy-powered command-line tool that:
- Fetches LIGO GWOSC strain data for an event (e.g., GW150914)
- Whitens and band-limits the signal on GPU
- Extracts instantaneous frequency via a Hilbert transform
- Learns the power-law df/dt ≈ K · f^α from data (no GR templates)
- Recovers the chirp mass from the fitted coefficient K
- Saves plots (PNG) and data (CSV/JSON) in outputs/

Why it’s useful
- Data-driven rediscovery: α ≈ 11/3 emerges from the data without waveform templates
- Direct parameter recovery: chirp mass from K at leading (quadrupole) order
- Reproducible: simple CLI, deterministic defaults, and cached data

Status: v0.1.0 (CLI only)

Requirements
- Windows 11 with a recent NVIDIA GPU/driver (e.g., RTX 5090) and CUDA 12.x runtime
- Python 3.11 (64-bit)
- CuPy wheel matching CUDA 12.x: cupy-cuda12x

Install
Option A — pip (recommended)
- pip install -e .
- pip install cupy-cuda12x

Option B — conda (creates a base env; add CuPy via pip)
- conda env create -f env/environment.yml
- conda activate derivegr
- pip install -e .
- pip install cupy-cuda12x

Sanity check GPU
- python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.cuda.Device(0).name)"

First run (fetches and caches data automatically)
- derivegr discover-inspiral --event GW150914 --ifo H1 L1 --tpad 16 --flow 30 --fhigh 350 --out outputs/gw150914 --save-plots --save-csv --gpu-device 0

Outputs
- outputs/<event>/
  - finst.csv      (t, f_inst)
  - dfdt.csv       (t, df/dt)
  - fit_results.json  (alpha, K, Mc in SI and solar units, plus metadata)
  - finst.png, dfdt.png, loglog_fit.png

Follow-up: chirp mass from saved results
- derivegr chirp-mass --from-json outputs/gw150914/fit_results.json

Prefetch (optional)
- derivegr prefetch --event GW150914 --ifo H1 L1 --tpad 16

Data & caching
- Default cache directory on Windows: %LOCALAPPDATA%/DeriveGR/gwosc
- Override with environment variable DERIVEGR_CACHE
- Large artifacts (outputs/ etc.) are tracked by Git LFS if committed

Tips for Windows shells
- Use straight quotes ' and " (avoid “ ” and ’)
- Ensure no unmatched quotes or trailing backslashes in commands

Troubleshooting
- CuPy import error: install a wheel matching your CUDA runtime (cupy-cuda12x)
- No GPU device found: verify NVIDIA driver install and CUDA support
- Not enough samples after masking: try adjusting --flow/--fhigh or --smooth-sec

License
- MIT

Acknowledgments
- LIGO Open Science Center (GWOSC) open data
- CuPy for GPU-accelerated NumPy/SciPy operations

