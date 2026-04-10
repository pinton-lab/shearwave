"""Acoustic strain and strain-gradient computation from pressure fields.

Computes the volumetric acoustic strain from the linearized equation of state:

    epsilon_vol = -p / K

where K = rho * c^2 is the bulk modulus. This is the *acoustic-timescale*
strain produced directly by the ultrasound wave, distinct from the slower
radiation-force-driven shear-wave strain computed by the shear FDTD solver.

The spatial gradient of the strain field (d(epsilon)/dx_i) is relevant for
flexoelectric coupling models (Felix et al. 2022), where acoustic strain
gradients drive membrane polarization and induced currents in neural tissue.

Notes
-----
Only volumetric (hydrostatic) strain is computed here.  The full deviatoric
strain tensor would require particle velocity output, which is not always
available.  For a longitudinal plane wave the volumetric strain is the
dominant component (deviatoric contributions are O(v/c) smaller).
"""

import numpy as np
from numpy.typing import NDArray

from .solver import gradient_center

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _bulk_modulus(
    c_map: NDArray[np.floating],
    rho_map: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute bulk modulus K = rho * c^2 [Pa]."""
    return rho_map * c_map**2


def _resolve_gate(
    pressure_4d: NDArray[np.floating],
    gate: slice | NDArray[np.integer] | None,
) -> slice | NDArray[np.integer]:
    """Auto-gate using 10% threshold on mean squared pressure.

    Parameters
    ----------
    pressure_4d : ndarray (nx, ny, nz, nt)
        Pressure field time series.
    gate : slice, index array, or None
        If not None, returned as-is.

    Returns
    -------
    gate : slice or index array
        Time indices to use.

    """
    if gate is not None:
        return gate
    p_sq = pressure_4d**2
    env = p_sq.mean(axis=(0, 1, 2))  # (nt,)
    threshold = 0.1 * env.max()
    gate_mask = env >= threshold
    if not gate_mask.any():
        return slice(None)
    return np.nonzero(gate_mask)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_acoustic_strain_volumetric(
    pressure_4d: NDArray[np.floating],
    c_map: NDArray[np.floating],
    rho_map: NDArray[np.floating],
    gate: slice | NDArray[np.integer] | None = None,
) -> NDArray[np.floating]:
    """Compute full 4-D volumetric acoustic strain from pressure.

    epsilon(x,y,z,t) = -p(x,y,z,t) / K(x,y,z)

    Parameters
    ----------
    pressure_4d : ndarray (nx, ny, nz, nt)
        Pressure field time series [Pa].
    c_map : ndarray (nx, ny, nz)
        Sound speed map [m/s].
    rho_map : ndarray (nx, ny, nz)
        Density map [kg/m^3].
    gate : slice, index array, or None
        Time indices. If None, auto-detects using 10% threshold on mean p^2.

    Returns
    -------
    strain_vol : ndarray (nx, ny, nz, nt_gated)
        Volumetric strain (dimensionless).

    """
    gate = _resolve_gate(pressure_4d, gate)
    K = _bulk_modulus(c_map, rho_map)  # (nx, ny, nz)
    return -pressure_4d[:, :, :, gate] / K[..., np.newaxis]


def compute_acoustic_strain_peak(
    pressure_4d: NDArray[np.floating],
    c_map: NDArray[np.floating],
    rho_map: NDArray[np.floating],
    gate: slice | NDArray[np.integer] | None = None,
) -> dict[str, NDArray[np.floating]]:
    """Memory-efficient peak/RMS/mean acoustic strain statistics.

    Computes statistics directly from pressure without materializing the
    full 4-D strain array.

    Parameters
    ----------
    pressure_4d : ndarray (nx, ny, nz, nt)
        Pressure field time series [Pa].
    c_map : ndarray (nx, ny, nz)
        Sound speed map [m/s].
    rho_map : ndarray (nx, ny, nz)
        Density map [kg/m^3].
    gate : slice, index array, or None
        Time indices. If None, auto-detects using 10% threshold on mean p^2.

    Returns
    -------
    result : dict
        ``"peak"`` : ndarray (nx, ny, nz) — max |epsilon| over time.
        ``"rms"``  : ndarray (nx, ny, nz) — sqrt(mean(epsilon^2)) over time.
        ``"mean"`` : ndarray (nx, ny, nz) — mean(epsilon) over time.

    """
    gate = _resolve_gate(pressure_4d, gate)
    K = _bulk_modulus(c_map, rho_map)
    p_gated = pressure_4d[:, :, :, gate]

    # peak |epsilon| = max_t |p| / K
    peak = np.max(np.abs(p_gated), axis=3) / K

    # rms epsilon = sqrt(mean(p^2)) / K
    rms = np.sqrt(np.mean(p_gated**2, axis=3)) / K

    # mean epsilon = -mean(p) / K
    mean = -np.mean(p_gated, axis=3) / K

    return {"peak": peak, "rms": rms, "mean": mean}


def compute_acoustic_strain_gradient(
    pressure_4d: NDArray[np.floating],
    c_map: NDArray[np.floating],
    rho_map: NDArray[np.floating],
    dX: float,
    dY: float,
    dZ: float,
    gate: slice | NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute spatial gradient of volumetric acoustic strain.

    Returns d(epsilon)/dx_i for each spatial direction.  The gradient is
    taken of the strain field (not just pressure) so that spatial variations
    of bulk modulus K(x) are captured.

    Parameters
    ----------
    pressure_4d : ndarray (nx, ny, nz, nt)
        Pressure field time series [Pa].
    c_map : ndarray (nx, ny, nz)
        Sound speed map [m/s].
    rho_map : ndarray (nx, ny, nz)
        Density map [kg/m^3].
    dX, dY, dZ : float
        Grid spacing in each direction [m].
    gate : slice, index array, or None
        Time indices. If None, auto-detects using 10% threshold on mean p^2.

    Returns
    -------
    grad_x, grad_y, grad_z : ndarray (nx, ny, nz, nt_gated)
        Strain gradient components [1/m].

    """
    gate = _resolve_gate(pressure_4d, gate)
    K = _bulk_modulus(c_map, rho_map)
    p_gated = pressure_4d[:, :, :, gate]
    nt = p_gated.shape[3]

    nx, ny, nz = K.shape
    grad_x = np.zeros((nx, ny, nz, nt), dtype=p_gated.dtype)
    grad_y = np.zeros_like(grad_x)
    grad_z = np.zeros_like(grad_x)

    for t in range(nt):
        strain_t = -p_gated[:, :, :, t] / K
        gx, gy, gz = gradient_center(strain_t, dX, dY, dZ)
        grad_x[:, :, :, t] = gx
        grad_y[:, :, :, t] = gy
        grad_z[:, :, :, t] = gz

    return grad_x, grad_y, grad_z


def compute_acoustic_strain_gradient_peak(
    pressure_4d: NDArray[np.floating],
    c_map: NDArray[np.floating],
    rho_map: NDArray[np.floating],
    dX: float,
    dY: float,
    dZ: float,
    gate: slice | NDArray[np.integer] | None = None,
) -> dict[str, NDArray[np.floating]]:
    """Memory-efficient peak strain-gradient statistics.

    Loops over time steps and tracks running max/sum to avoid materializing
    the full 4-D gradient arrays.

    Parameters
    ----------
    pressure_4d : ndarray (nx, ny, nz, nt)
        Pressure field time series [Pa].
    c_map : ndarray (nx, ny, nz)
        Sound speed map [m/s].
    rho_map : ndarray (nx, ny, nz)
        Density map [kg/m^3].
    dX, dY, dZ : float
        Grid spacing in each direction [m].
    gate : slice, index array, or None
        Time indices. If None, auto-detects using 10% threshold on mean p^2.

    Returns
    -------
    result : dict
        ``"peak_magnitude"`` : ndarray (nx, ny, nz) — max over time of
        |grad(epsilon)|.
        ``"peak_grad_x"`` : ndarray (nx, ny, nz) — max |d(epsilon)/dx|.
        ``"peak_grad_y"`` : ndarray (nx, ny, nz) — max |d(epsilon)/dy|.
        ``"peak_grad_z"`` : ndarray (nx, ny, nz) — max |d(epsilon)/dz|.
        ``"rms_magnitude"`` : ndarray (nx, ny, nz) — sqrt(mean(|grad|^2)).

    """
    gate = _resolve_gate(pressure_4d, gate)
    K = _bulk_modulus(c_map, rho_map)
    p_gated = pressure_4d[:, :, :, gate]
    nt = p_gated.shape[3]

    nx, ny, nz = K.shape
    peak_mag = np.zeros((nx, ny, nz), dtype=np.float64)
    peak_gx = np.zeros_like(peak_mag)
    peak_gy = np.zeros_like(peak_mag)
    peak_gz = np.zeros_like(peak_mag)
    sum_mag_sq = np.zeros_like(peak_mag)

    for t in range(nt):
        strain_t = -p_gated[:, :, :, t].astype(np.float64) / K
        gx, gy, gz = gradient_center(strain_t, dX, dY, dZ)

        mag = np.sqrt(gx**2 + gy**2 + gz**2)
        np.maximum(peak_mag, mag, out=peak_mag)
        np.maximum(peak_gx, np.abs(gx), out=peak_gx)
        np.maximum(peak_gy, np.abs(gy), out=peak_gy)
        np.maximum(peak_gz, np.abs(gz), out=peak_gz)
        sum_mag_sq += mag**2

    rms_mag = np.sqrt(sum_mag_sq / max(nt, 1))

    return {
        "peak_magnitude": peak_mag,
        "peak_grad_x": peak_gx,
        "peak_grad_y": peak_gy,
        "peak_grad_z": peak_gz,
        "rms_magnitude": rms_mag,
    }
