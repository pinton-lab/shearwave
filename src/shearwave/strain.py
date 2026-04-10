"""Strain tensor and invariant computation from displacement fields.

Converts 3D displacement fields (output of the shear FDTD solver) into
the symmetric strain tensor and scalar invariants useful for downstream
mechanics (e.g., membrane tension estimation for neuromodulation).

All quantities are in SI units (meters, dimensionless strain).
"""

import numpy as np
from numpy.typing import NDArray

from .solver import gradient_center


def compute_strain_tensor(
    ux: NDArray[np.floating],
    uy: NDArray[np.floating],
    uz: NDArray[np.floating],
    dX: float,
    dY: float,
    dZ: float,
) -> NDArray[np.floating]:
    """Compute the symmetric strain tensor from a displacement field.

    Uses central differences (via ``gradient_center``) to compute spatial
    derivatives of the displacement vector ``(ux, uy, uz)``.

    Parameters
    ----------
    ux, uy, uz : ndarray, shape (nx, ny, nz)
        Displacement components along depth, lateral, and elevational axes [m].
    dX, dY, dZ : float
        Grid spacing along each axis [m].

    Returns
    -------
    epsilon : ndarray, shape (nx, ny, nz, 6)
        Symmetric strain tensor in Voigt notation:
        ``[eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy]``.
        Shear components are engineering shear strain (2 * eps_ij).

    Raises
    ------
    ValueError
        If input arrays do not all share the same 3-D shape or if any
        grid spacing is not positive.

    """
    _validate_inputs(ux, uy, uz, dX, dY, dZ)

    nx, ny, nz = ux.shape
    dtype = np.result_type(ux.dtype, uy.dtype, uz.dtype)
    epsilon = np.zeros((nx, ny, nz, 6), dtype=dtype)

    # Diagonal: eps_ii = du_i / dx_i
    dux_dx, dux_dy, dux_dz = gradient_center(ux, dX, dY, dZ)
    duy_dx, duy_dy, duy_dz = gradient_center(uy, dX, dY, dZ)
    duz_dx, duz_dy, duz_dz = gradient_center(uz, dX, dY, dZ)

    epsilon[..., 0] = dux_dx  # eps_xx
    epsilon[..., 1] = duy_dy  # eps_yy
    epsilon[..., 2] = duz_dz  # eps_zz

    # Off-diagonal (engineering shear = du_i/dx_j + du_j/dx_i)
    epsilon[..., 3] = duy_dz + duz_dy  # gamma_yz
    epsilon[..., 4] = dux_dz + duz_dx  # gamma_xz
    epsilon[..., 5] = dux_dy + duy_dx  # gamma_xy

    return epsilon


def compute_strain_invariants(
    epsilon: NDArray[np.floating],
) -> dict[str, NDArray[np.floating]]:
    """Compute scalar strain invariants from a Voigt-notation strain tensor.

    Parameters
    ----------
    epsilon : ndarray, shape (..., 6)
        Strain tensor in Voigt notation (see ``compute_strain_tensor``).

    Returns
    -------
    dict
        ``'volumetric'`` : ndarray — trace / 3.
        ``'deviatoric'`` : ndarray — Frobenius norm of the deviatoric tensor.
        ``'max_shear'`` : ndarray — maximum shear strain (half-difference of
        extreme principal strains, approximated from invariants).
        ``'von_mises'`` : ndarray — von Mises equivalent strain.

    Raises
    ------
    ValueError
        If the last dimension of *epsilon* is not 6.

    """
    if epsilon.shape[-1] != 6:
        msg = f"Expected last dimension == 6 (Voigt notation), got {epsilon.shape[-1]}"
        raise ValueError(msg)

    exx = epsilon[..., 0]
    eyy = epsilon[..., 1]
    ezz = epsilon[..., 2]
    gyz = epsilon[..., 3]  # engineering shear
    gxz = epsilon[..., 4]
    gxy = epsilon[..., 5]

    # Volumetric strain (mean normal strain)
    vol = (exx + eyy + ezz) / 3.0

    # Deviatoric normal strains
    dxx = exx - vol
    dyy = eyy - vol
    dzz = ezz - vol

    # Deviatoric Frobenius norm  (shear components stored as engineering strain,
    # so tensor shear = gamma/2; Frobenius uses tensor components)
    dev = np.sqrt(dxx**2 + dyy**2 + dzz**2 + 0.5 * (gyz**2 + gxz**2 + gxy**2))

    # Von Mises equivalent strain
    # eps_vM = sqrt(2/3) * ||dev||_F  (using tensor shear components)
    von_mises = np.sqrt(2.0 / 3.0) * dev

    # Max shear from second invariant of deviatoric strain
    # J2 = 0.5 * ||dev||_F^2, max_shear = sqrt(J2)
    j2 = 0.5 * (dxx**2 + dyy**2 + dzz**2 + 0.5 * (gyz**2 + gxz**2 + gxy**2))
    max_shear = np.sqrt(np.maximum(j2, 0.0))

    return {
        "volumetric": vol,
        "deviatoric": dev,
        "max_shear": max_shear,
        "von_mises": von_mises,
    }


def _validate_inputs(
    ux: NDArray[np.floating],
    uy: NDArray[np.floating],
    uz: NDArray[np.floating],
    dX: float,
    dY: float,
    dZ: float,
) -> None:
    """Check shapes and grid spacing are valid."""
    if ux.ndim != 3:
        msg = f"Displacement arrays must be 3-D, got ndim={ux.ndim}"
        raise ValueError(msg)
    if ux.shape != uy.shape or ux.shape != uz.shape:
        msg = (
            f"All displacement components must share the same shape: "
            f"ux={ux.shape}, uy={uy.shape}, uz={uz.shape}"
        )
        raise ValueError(msg)
    for name, val in [("dX", dX), ("dY", dY), ("dZ", dZ)]:
        if val <= 0:
            msg = f"Grid spacing {name} must be positive, got {val}"
            raise ValueError(msg)
