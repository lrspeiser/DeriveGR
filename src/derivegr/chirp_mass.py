import numpy as np

G = 6.67430e-11     # m^3 kg^-1 s^-2
c = 299_792_458.0   # m/s
M_sun = 1.98847e30  # kg


def from_K(K: float):
    """Compute chirp mass from the coefficient K in df/dt = K f^alpha.

    For GR leading-order with alpha ≈ 11/3:
    K = (96/5) * pi^(8/3) * (G * Mc / c^3)^(5/3)
    => Mc = (c^3 / G) * [ (5/96) * K * pi^(-8/3) ]^(3/5)
    """
    pi = np.pi
    pref = (5.0 / 96.0) * (K) * (pi ** (-8.0 / 3.0))
    Mc_kg = (c ** 3 / G) * (pref ** (3.0 / 5.0))
    return {"Mc_kg": float(Mc_kg), "Mc_solar": float(Mc_kg / M_sun)}

