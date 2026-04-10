"""Acoustic radiation force (ARF) computation from pressure field output.

NOTE: The current implementation uses a plane-wave approximation for intensity
(I = <p^2> / (rho*c)), which assumes a locally planar wavefront. This breaks
down for converging beams near the focal zone and produces only a scalar force
assumed to act along the propagation direction.

A more accurate approach would use the Poynting vector I_vec = <p * v_vec>,
which gives the true directional energy flux and a vector body force
f_vec = 2*alpha/c * I_vec in all three spatial directions. This requires
particle velocity, which can be obtained from the existing pressure output
via the linearized Euler equation:

    rho_0 * dv/dt = -grad(p)
    v(t+dt) = v(t) - (dt/rho_0) * grad(p(t))

The downsampled grid (~2.5 points/wavelength at mod=4, 1 MHz) is adequate for
computing spatial gradients via finite differences. Time integration of the
velocity uses the recorded time steps (dt_recorded = dt * sampling_modulus_time).
The Poynting vector is then time-averaged over the gated window to yield I_vec.
"""

import numpy as np
from numpy.typing import NDArray


def compute_radiation_force(
    pressure_4d: NDArray[np.floating],
    c_map: NDArray[np.floating],
    rho_map: NDArray[np.floating],
    alpha_coeff_map: NDArray[np.floating],
    f0: float,
    gate: slice | NDArray[np.integer] | None = None,
) -> NDArray[np.floating]:
    """Compute acoustic radiation force from pressure field.

    Implements: b = 2 * alpha_Np * Isppa / c

    Parameters
    ----------
    pressure_4d : ndarray (nx, ny, nz, nt)
        Pressure field time series on the downsampled grid.
    c_map : ndarray (nx, ny, nz)
        Sound speed map [m/s].
    rho_map : ndarray (nx, ny, nz)
        Density map [kg/m^3].
    alpha_coeff_map : ndarray (nx, ny, nz)
        Attenuation coefficient map [dB/cm/MHz].
    f0 : float
        Center frequency [Hz].
    gate : slice, index array, or None
        Time indices for SPPA computation. If None, auto-detects
        active region using 10% threshold on mean squared pressure.

    Returns
    -------
    b0 : ndarray (nx, ny, nz)
        Body force density [N/m^3].

    """
    # Auto-gate: find time indices where mean p^2 exceeds 10% of peak
    if gate is None:
        p_sq = pressure_4d**2
        env = p_sq.mean(axis=(0, 1, 2))  # (nt,)
        threshold = 0.1 * env.max()
        gate_mask = env >= threshold
        gate = slice(None) if not gate_mask.any() else np.nonzero(gate_mask)[0]

    # Isppa = mean(p^2) / Z  [W/m^2]
    z_map = rho_map * c_map  # acoustic impedance
    isppa = np.mean(pressure_4d[:, :, :, gate] ** 2, axis=3) / z_map

    # Attenuation: dB/cm/MHz -> Np/m
    f0_mhz = f0 / 1e6
    alpha_np_per_m = (alpha_coeff_map * f0_mhz) / 8.685889638 * 100.0

    # Radiation force: b = 2 * alpha * I / c
    return 2.0 * alpha_np_per_m * isppa / c_map


def compute_radiation_force_velocity(
    pressure_4d: NDArray[np.floating],
    velocity_u_4d: NDArray[np.floating],
    velocity_w_4d: NDArray[np.floating],
    velocity_v_4d: NDArray[np.floating],
    c_map: NDArray[np.floating],
    rho_map: NDArray[np.floating],
    alpha_coeff_map: NDArray[np.floating],
    f0: float,
    gate: slice | NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute vector acoustic radiation force using the Poynting vector.

    Uses I_vec = <p * v_vec> (time-averaged Poynting vector) to compute
    a 3-component body force: f_i = 2 * alpha_Np * I_i / c.

    This is more accurate than the plane-wave approximation for converging
    beams near the focal zone, where the wavefront is not locally planar.

    Parameters
    ----------
    pressure_4d : ndarray (nx, ny, nz, nt)
        Pressure field time series on the downsampled grid [Pa].
    velocity_u_4d : ndarray (nx, ny, nz, nt)
        Particle velocity in x/depth direction [m/s].
    velocity_w_4d : ndarray (nx, ny, nz, nt)
        Particle velocity in y/lateral direction [m/s].
    velocity_v_4d : ndarray (nx, ny, nz, nt)
        Particle velocity in z/elevational direction [m/s].
    c_map : ndarray (nx, ny, nz)
        Sound speed map [m/s].
    rho_map : ndarray (nx, ny, nz)
        Density map [kg/m^3].
    alpha_coeff_map : ndarray (nx, ny, nz)
        Attenuation coefficient map [dB/cm/MHz].
    f0 : float
        Center frequency [Hz].
    gate : slice, index array, or None
        Time indices for averaging. If None, auto-detects active region
        using 10% threshold on mean squared pressure.

    Returns
    -------
    bx : ndarray (nx, ny, nz)
        Body force density along depth [N/m^3].
    by : ndarray (nx, ny, nz)
        Body force density along lateral [N/m^3].
    bz : ndarray (nx, ny, nz)
        Body force density along elevational [N/m^3].

    """
    # Auto-gate: find time indices where mean p^2 exceeds 10% of peak
    if gate is None:
        p_sq = pressure_4d**2
        env = p_sq.mean(axis=(0, 1, 2))  # (nt,)
        threshold = 0.1 * env.max()
        gate_mask = env >= threshold
        gate = slice(None) if not gate_mask.any() else np.nonzero(gate_mask)[0]

    p_gated = pressure_4d[:, :, :, gate]

    # Poynting vector: I_i = mean(p * v_i) [W/m^2]
    ix = np.mean(p_gated * velocity_u_4d[:, :, :, gate], axis=3)
    iy = np.mean(p_gated * velocity_w_4d[:, :, :, gate], axis=3)
    iz = np.mean(p_gated * velocity_v_4d[:, :, :, gate], axis=3)

    # Attenuation: dB/cm/MHz -> Np/m
    f0_mhz = f0 / 1e6
    alpha_np_per_m = (alpha_coeff_map * f0_mhz) / 8.685889638 * 100.0

    # Vector radiation force: f_i = 2 * alpha * I_i / c
    coeff = 2.0 * alpha_np_per_m / c_map
    bx = coeff * ix
    by = coeff * iy
    bz = coeff * iz

    return bx, by, bz
