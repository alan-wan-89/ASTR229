#!/usr/bin/env python3
import argparse
import io
import os
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Physical constants (SI)
h = 6.62607015e-34       # Planck constant [J s]
c = 2.99792458e8         # Speed of light [m/s]
k_B = 1.380649e-23       # Boltzmann constant [J/K]
R_SUN = 6.957e8          # Solar radius [m]
L_SUN = 3.828e26         # Solar luminosity [W]


# -----------------------------
# Planck-law utilities
# -----------------------------

def intensity_planck_lambda(wavelength_m: np.ndarray, temperature_K: float) -> np.ndarray:
    """Planck spectral radiance B_λ(λ, T) in W m^-3 sr^-1.
    Args:
        wavelength_m: Wavelength(s) in meters (array-like).
        temperature_K: Temperature in Kelvin.
    Returns:
        B_lambda with shape like wavelength_m, units W / m^3 / sr.
    """
    lam = np.asarray(wavelength_m, dtype=float)
    T = float(temperature_K)
    x = (h * c) / (lam * k_B * T)
    numerator = 2.0 * h * c**2 / lam**5
    denom = np.expm1(x)
    return numerator / denom


def bolometric_flux(temperature_K: float, lam_min_m: float = 1e-9,
                    lam_max_m: float = 1e-2, n: int = 20000) -> float:
    """Integrate π B_λ dλ over wavelength to get F_bol [W m^-2]."""
    lam = np.logspace(np.log10(lam_min_m), np.log10(lam_max_m), n)
    B = intensity_planck_lambda(lam, temperature_K)  # W m^-3 sr^-1
    integral = np.trapz(B, lam)  # W m^-2 sr^-1
    return float(np.pi * integral)  # W m^-2


# -----------------------------
# HR computation and plotting
# -----------------------------

def compute_luminosity_from_TR(temperature_K: float, radius_Rsun: float,
                                n: int = 20000,
                                lam_min_m: float = 1e-9,
                                lam_max_m: float = 1e-2) -> float:
    """Compute bolometric luminosity via Planck integral and surface area [W]."""
    R_m = radius_Rsun * R_SUN
    F_bol = bolometric_flux(temperature_K, lam_min_m=lam_min_m, lam_max_m=lam_max_m, n=n)
    L = 4.0 * np.pi * (R_m ** 2) * F_bol
    return float(L)


def compute_luminosities(temps_K: np.ndarray, radii_Rsun: np.ndarray,
                         n: int = 20000) -> np.ndarray:
    L_list = []
    for T, R in zip(temps_K, radii_Rsun):
        L_list.append(compute_luminosity_from_TR(float(T), float(R), n=n))
    return np.asarray(L_list, dtype=float)


def plot_hr(temperatures_K: np.ndarray, luminosities_Lsun: np.ndarray,
            masses_Msun: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(temperatures_K, luminosities_Lsun,
                    c=masses_Msun, cmap="plasma", s=35, edgecolor="k", linewidth=0.3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()

    ax.set_xlabel("Effective temperature [K]")
    ax.set_ylabel("Luminosity [L$_\\odot$]")
    cbar = fig.colorbar(sc, ax=ax, label="Mass [M$_\\odot$]")

    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# -----------------------------
# Data loading
# -----------------------------
_DEFAULT_MS_DATA = """
Mass_Msun Radius_Rsun Temperature_K
-----------------------------------
30.300000 10.508744 54000.0
22.900000 8.627689 45000.0
21.700000 8.334676 43300.0
19.700000 7.827926 40600.0
17.600000 7.186731 37800.0
12.000000 5.525893 29200.0
8.240000 4.271461 23000.0
7.140000 3.852133 21000.0
5.480000 3.226629 17600.0
4.360000 2.736007 15200.0
3.980000 2.576031 14300.0
3.640000 2.418269 13500.0
3.160000 2.202130 12300.0
2.810000 2.018549 11400.0
2.170000 1.695595 9600.0
2.060000 1.623776 9330.0
1.970000 1.578927 9040.0
1.860000 1.507397 8750.0
1.780000 1.465080 8480.0
1.730000 1.447346 8310.0
1.610000 1.374802 7920.0
1.440000 1.278830 7350.0
1.350000 1.217676 7050.0
1.290000 1.188095 6850.0
1.250000 1.149765 6700.0
1.200000 1.125331 6550.0
1.160000 1.091263 6400.0
1.140000 1.094452 6300.0
1.070000 1.037801 6050.0
1.040000 1.037850 5930.0
1.000000 0.990369 5800.0
0.963000 0.964427 5660.0
0.908000 0.928346 5440.0
0.857000 0.891636 5240.0
0.824000 0.865347 5110.0
0.785000 0.834798 4960.0
0.746000 0.805103 4800.0
0.700000 0.771335 4600.0
0.660000 0.750109 4400.0
0.562000 0.658466 4000.0
0.513000 0.622322 3750.0
0.503000 0.615658 3700.0
0.482000 0.597372 3600.0
0.463000 0.583305 3500.0
0.442000 0.561807 3400.0
0.402000 0.524614 3200.0
0.385000 0.514211 3100.0
0.346000 0.468728 2900.0
0.311000 0.440724 2700.0
0.293000 0.423958 2600.0
0.227000 0.357676 2200.0
0.126000 0.238758 1500.0
0.114000 0.221626 1400.0
0.068000 0.152673 1000.0
0.048300 0.122083 800.0
""".strip()


def _clean_ascii_lines(lines: list[str]) -> str:
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if set(s) <= {"-"}:
            continue
        cleaned.append(s)
    return "\n".join(cleaned)


def load_ms_data(input_path: str | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if input_path and os.path.exists(input_path):
        with open(input_path, "r") as f:
            text = _clean_ascii_lines(f.readlines())
    else:
        text = _clean_ascii_lines(_DEFAULT_MS_DATA.splitlines())

    arr = np.genfromtxt(io.StringIO(text), names=True)
    masses = np.array(arr["Mass_Msun"], dtype=float)
    radii = np.array(arr["Radius_Rsun"], dtype=float)
    temps = np.array(arr["Temperature_K"], dtype=float)
    return masses, radii, temps


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HR diagram using bolometric flux integration.")
    parser.add_argument("--input", type=str, default=None, help="Path to MS table (whitespace columns).")
    parser.add_argument("--output", type=str, default="hr_diagram.png", help="Output image path.")
    parser.add_argument("--n", type=int, default=20000, help="Integration samples for wavelength grid.")
    args = parser.parse_args()

    masses, radii, temps = load_ms_data(args.input)
    L = compute_luminosities(temps, radii, n=args.n)  # W
    L_over_Lsun = L / L_SUN

    plot_hr(temps, L_over_Lsun, masses, args.output)

    print(f"Saved HR diagram to: {args.output}")


if __name__ == "__main__":
    main()
