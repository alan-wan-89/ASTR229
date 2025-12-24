import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

import astropy.units as u
import astropy.constants as const
from scipy.integrate import simpson


# ------------------------------
# Blackbody / Planck law helpers
# ------------------------------

def planck_lambda(wavelength: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """
    Planck blackbody spectral radiance B_lambda(T) as a function of wavelength.

    Parameters
    ----------
    wavelength : Quantity
        Wavelength array with length units (e.g., u.m, u.nm).
    temperature : Quantity
        Temperature with units of Kelvin.

    Returns
    -------
    Quantity
        Spectral radiance per unit wavelength with units of
        W / (m2 * sr * m). This is B_lambda.
    """
    # Ensure proper units
    wavelength = u.Quantity(wavelength).to(u.m)
    temperature = u.Quantity(temperature).to(u.K)

    h = const.h
    c = const.c
    k_B = const.k_B

    # Planck law in wavelength form
    # B_lambda = (2hc^2 / lambda^5) * 1 / (exp(hc/(lambda kT)) - 1)
    prefactor = 2 * h * c**2 / wavelength**5  # units: W / m^3
    exponent = (h * c) / (wavelength * k_B * temperature)
    # Add steradian^-1 explicitly for radiance
    intensity = (prefactor / np.expm1(exponent.value)) / u.sr  # W / (sr m^3)
    intensity = intensity.to(u.W / (u.sr * u.m**3))
    return intensity


def flux_surface_from_B_lambda(wavelength: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """
    Integrate pi * B_lambda over wavelength to get surface flux F (per unit area)
    emerging from the stellar surface (bolometric if integrated over all wavelengths).

    Parameters
    ----------
    wavelength : Quantity
        Wavelength array covering desired band, with length units.
    temperature : Quantity
        Temperature with units of Kelvin.

    Returns
    -------
    Quantity
        Flux per unit area with units of W / m^2 for the provided band.
    """
    lam = u.Quantity(wavelength).to(u.m)
    B_lam = planck_lambda(lam, temperature)  # W / (m^2 sr m)
    # Integrate over solid angle of a Lambertian emitter: F_lambda = pi * B_lambda
    integrand = (np.pi * u.sr) * B_lam  # W / (m^2 m)
    F = simpson(integrand.value, x=lam.to(u.m).value) * integrand.unit * u.m
    F = F.to(u.W / u.m**2)
    return F


# ----------------------------------------
# Luminosity and Stefan-Boltzmann relation
# ----------------------------------------

def luminosity_from_radius_temperature(radius: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """
    L = 4 * pi * R^2 * sigma_sb * T^4
    """
    R = u.Quantity(radius).to(u.m)
    T = u.Quantity(temperature).to(u.K)
    sigma = const.sigma_sb
    L = 4 * np.pi * R**2 * sigma * T**4
    return L.to(u.W)


def bolometric_flux_via_stefan_boltzmann(temperature: u.Quantity) -> u.Quantity:
    """
    F = sigma_sb * T^4 (surface flux integrated over all wavelengths)
    """
    T = u.Quantity(temperature).to(u.K)
    sigma = const.sigma_sb
    return (sigma * T**4).to(u.W / u.m**2)


# ------------------------------
# Solar sanity checks
# ------------------------------

def solar_luminosity_checks():
    T_sun = 5772 * u.K
    R_sun = const.R_sun

    # Numerical integration over wavelength (broad range) for F\_bol
    wavelengths = np.logspace(-9, -3, 10000) * u.m  # 1 nm to 1 mm
    F_num = flux_surface_from_B_lambda(wavelengths, T_sun)

    # Stefan-Boltzmann flux
    F_sb = bolometric_flux_via_stefan_boltzmann(T_sun)

    # Luminosities
    L_num = (4 * np.pi * R_sun**2 * F_num).to(u.W)
    L_sb = luminosity_from_radius_temperature(R_sun, T_sun)

    return {
        "F_num": F_num,
        "F_sb": F_sb,
        "L_num": L_num,
        "L_sb": L_sb,
        "L_sun_const": const.L_sun.to(u.W),
    }


# ------------------------------
# Mass–radius estimate for main sequence
# ------------------------------

@dataclass
class StellarEntry:
    mass_Msun: float
    temperature_K: float
    luminosity_Lsun: float


def estimate_radius_from_mass(mass: u.Quantity) -> u.Quantity:
    """
    Approximate main-sequence radius scaling: R ~ R_sun * (M / M_sun)^0.8
    """
    mass = u.Quantity(mass)
    M_ratio = (mass / const.M_sun).to(u.dimensionless_unscaled).value
    R = const.R_sun * (M_ratio ** 0.8)
    return R.to(u.m)


def load_ms_data_or_synthetic(path: str | None = None) -> pd.DataFrame:
    """
    Try to load ASTR229_data/ms_updated.dat. If not found, synthesize a plausible grid.
    Expected columns: mass(Msun), temperature(K), luminosity(Lsun)
    """
    if path is not None:
        try:
            df = pd.read_csv(path, delim_whitespace=True, comment="#",
                             names=["mass_Msun", "temperature_K", "luminosity_Lsun"])  # fallback names
            return df
        except Exception:
            pass

    # Synthetic fallback: sample masses, estimate T and L with crude scalings
    # L ~ M^3.5, T ~ 5800 * M^0.5 (very rough), just for demonstration
    masses = np.logspace(-1, 1.2, 60)  # 0.1 to ~16 Msun
    temperatures = 5800.0 * masses ** 0.5
    luminosities = masses ** 3.5
    df = pd.DataFrame({
        "mass_Msun": masses,
        "temperature_K": temperatures,
        "luminosity_Lsun": luminosities,
    })
    return df


# ------------------------------
# Build H-R diagram from model luminosities
# ------------------------------

def build_hr_diagram(df: pd.DataFrame, use_model_luminosity: bool = True) -> plt.Figure:
    """
    Create an H-R diagram. If use_model_luminosity=True, compute L via blackbody with
    radius estimate; otherwise use dataset luminosity if present.
    """
    temperature = df["temperature_K"].values * u.K
    mass = df["mass_Msun"].values * const.M_sun

    if use_model_luminosity:
        radii = [estimate_radius_from_mass(m) for m in mass]
        radii = u.Quantity(radii)
        L = luminosity_from_radius_temperature(radii, temperature)
        L_over_Lsun = (L / const.L_sun).to(u.dimensionless_unscaled).value
    else:
        if "luminosity_Lsun" in df.columns:
            L_over_Lsun = df["luminosity_Lsun"].values
        else:
            raise ValueError("Dataset lacks 'luminosity_Lsun' column and use_model_luminosity=False")

    T = temperature.to(u.K).value

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.scatter(T, L_over_Lsun, s=20, c=np.log10(L_over_Lsun), cmap="viridis", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Conventions: HR diagram has temperature decreasing to the right
    ax.invert_xaxis()

    ax.set_xlabel("Effective Temperature T (K)")
    ax.set_ylabel("Luminosity (L / L_sun)")
    ax.set_title("H-R Diagram (Main Sequence Approximation)")
    fig.tight_layout()
    return fig


# ------------------------------
# Main entry to run tasks
# ------------------------------

def main():
    checks = solar_luminosity_checks()

    print("Solar checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    # Load dataset if available
    import os
    candidates = [
        "ASTR229_data/ms_updated.dat",
        "ms_updated.dat",
        "./ASTR229_data/ms_updated.dat",
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break

    df = load_ms_data_or_synthetic(path)

    # Show a few radius estimates
    df["radius_Rsun_model"] = [
        (estimate_radius_from_mass(m * const.M_sun) / const.R_sun).to(u.dimensionless_unscaled).value
        for m in df["mass_Msun"].values
    ]

    # Build and save HR diagram using modeled luminosities
    fig = build_hr_diagram(df, use_model_luminosity=True)
    out_path = "hr_diagram.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved H-R diagram to {out_path}")


if __name__ == "__main__":
    main()
