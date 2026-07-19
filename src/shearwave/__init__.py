"""Shearwave: viscoelastic FDTD shear wave solver.

A staggered-grid finite-difference time-domain solver for shear wave
propagation in viscoelastic media with Kelvin-Voigt damping.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shearwave")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.1.0"

from .acoustic_strain import (
    compute_acoustic_strain_gradient,
    compute_acoustic_strain_gradient_peak,
    compute_acoustic_strain_peak,
    compute_acoustic_strain_volumetric,
)
from .radiation_force import compute_radiation_force, compute_radiation_force_velocity
from .solver import (
    divergence_center,
    elastic_solve_static,
    gradient_center,
    hann1d,
    laplacian_center,
    project_body_force_to_shear,
    project_vector_field,
    shear_fdtd_staggered,
    shear_fdtd_staggered_jax,
)
from .strain import compute_strain_invariants, compute_strain_tensor

__all__ = [
    "__version__",
    "compute_acoustic_strain_gradient",
    "compute_acoustic_strain_gradient_peak",
    "compute_acoustic_strain_peak",
    "compute_acoustic_strain_volumetric",
    "compute_radiation_force",
    "compute_radiation_force_velocity",
    "compute_strain_invariants",
    "compute_strain_tensor",
    "divergence_center",
    "elastic_solve_static",
    "gradient_center",
    "hann1d",
    "laplacian_center",
    "project_body_force_to_shear",
    "project_vector_field",
    "shear_fdtd_staggered",
    "shear_fdtd_staggered_jax",
]
