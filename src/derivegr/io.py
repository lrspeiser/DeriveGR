from pathlib import Path
import csv
import numpy as np


def write_csv(path: Path, header: list, t: np.ndarray, y: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for ti, yi in zip(t, y):
            w.writerow([f"{ti:.9f}", f"{yi:.9e}"])

