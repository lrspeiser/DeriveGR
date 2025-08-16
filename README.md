# DeriveGR

Windows-first, CuPy-accelerated CLI that discovers the leading-order GR inspiral law (df/dt ≈ K f^α) from open LIGO data and recovers the chirp mass — no templates required. Optional module includes a Cassini Shapiro-delay (1+γ) skeleton.

Status: v0.1.0 (work-in-progress)

Highlights
- GPU-only signal pipeline (CuPy, CUDA 12.x) tuned for RTX 5090; no CPU fallback
- Fetches GWOSC data on first run and caches locally (configurable)
- Fits power-law in log–log space and recovers chirp mass from K
- Saves PNG plots and CSV/JSON artifacts under outputs/
- Optional: symbolic discovery (PySR) and Cassini skeleton (spiceypy)

Quick start (Windows 11 + CUDA 12.x)
1) Create env (pip example)
   - Install Python 3.11 64-bit
   - pip install -e .
   - pip install cupy-cuda12x
   - (Optional) pip install -e .[symbolic]  # for PySR
2) Sanity check GPU
   - python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.cuda.Device(0).name)"
3) Run discovery (first run fetches GW150914)
   - derivegr discover-inspiral --event GW150914 --ifo H1 L1 --tpad 16 --flow 30 --fhigh 350 --out outputs/gw150914 --save-plots --save-csv --gpu-device 0
4) Chirp mass from result
   - derivegr chirp-mass --from-json outputs/gw150914/fit_results.json

Install with conda (alternative)
- conda env create -f env/environment.yml
- conda activate derivegr
- pip install -e .
- pip install cupy-cuda12x

Data and caching
- Default cache dir: %LOCALAPPDATA%/DeriveGR/gwosc (override with DERIVEGR_CACHE)
- You can prefetch: derivegr prefetch --event GW150914 --ifo H1 L1 --tpad 16

Rust-first Windows tools (optional but recommended)
- fd, rg (ripgrep), bat — great for searching/previewing. Example:
  - fd -H -t f "cli.py" src
  - rg -n --hidden "df/dt" src

Troubleshooting
- Ensure straight quotes ' and " in your shell (avoid “ ” and ’)
- If commands appear to hang, check for unmatched quotes or trailing backslashes
- CuPy import errors: install cupy-cuda12x to match your CUDA runtime; verify driver version supports CUDA 12.x

License
- MIT

