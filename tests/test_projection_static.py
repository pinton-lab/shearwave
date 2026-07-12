"""Regression tests for the Helmholtz-Hodge projection and static solver.

These guard two sign-convention bugs that previously slipped past the
NumPy/JAX parity suite (both backends were wrong in the same way):

* ``project_vector_field`` inverted the compact Laplacian while subtracting a
  gradient built from the 2h central operators, which *doubled* the divergence
  instead of removing it.  A correct projection must (a) leave a solenoidal
  field unchanged, (b) fully remove a pure-gradient field, and (c) return a
  discretely divergence-free field for arbitrary (boundary-tapered) input.
* ``elastic_solve_static`` used a negative-definite operator, so CG terminated
  on the first iteration and always returned all-zeros.
"""

import numpy as np
import pytest

from shearwave.solver import (
    divergence_center,
    elastic_solve_static,
    gradient_center,
    project_vector_field,
)


def _grid(n=32):
    d = 1.0 / n
    ax = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = ax[:, None, None]
    y = ax[None, :, None]
    z = ax[None, None, :]
    mask = np.zeros((n, n, n), dtype=np.float32)
    mask[1:-1, 1:-1, 1:-1] = 1.0
    return n, d, x, y, z, mask


def _max_div(fx, fy, fz, d, mask):
    return float(np.max(np.abs(divergence_center(fx, fy, fz, d, d, d) * mask)))


def _energy(fx, fy, fz, mask):
    return float(np.sum((fx**2 + fy**2 + fz**2) * mask))


def test_projection_removes_pure_gradient():
    """A curl-free field must be projected to (nearly) zero."""
    n, d, x, y, z, mask = _grid()
    psi = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)).astype(
        np.float32,
    ) * np.ones((n, n, n), np.float32)
    gx, gy, gz = (a.astype(np.float32) for a in gradient_center(psi, d, d, d))
    e_in = _energy(gx, gy, gz, mask)
    ox, oy, oz = project_vector_field(gx.copy(), gy.copy(), gz.copy(), d, d, d, 1e-9, 5000)
    assert _energy(ox, oy, oz, mask) / e_in < 1e-3


def test_projection_preserves_solenoidal_field():
    """A divergence-free field must pass through essentially unchanged."""
    n, d, x, y, z, mask = _grid()
    a_x = (np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)).astype(np.float32) * np.ones_like(x)
    a_z = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)).astype(np.float32) * np.ones_like(z)
    g_ax = gradient_center(a_x, d, d, d)
    g_az = gradient_center(a_z, d, d, d)
    # f = curl(a_x, 0, a_z)
    fx = g_az[1].astype(np.float32)
    fy = (g_ax[2] - g_az[0]).astype(np.float32)
    fz = (-g_ax[1]).astype(np.float32)
    e_in = _energy(fx, fy, fz, mask)
    ox, oy, oz = project_vector_field(fx.copy(), fy.copy(), fz.copy(), d, d, d, 1e-9, 5000)
    assert _energy(ox, oy, oz, mask) / e_in > 0.98


def test_projection_output_is_divergence_free():
    """A localized (boundary-tapered) push must project to ~zero divergence."""
    n, d, x, y, z, mask = _grid()
    blob = np.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / (2 * 0.08**2))).astype(
        np.float32,
    ) * np.ones((n, n, n), np.float32)
    zero = np.zeros_like(blob)
    div_before = _max_div(zero, zero, blob, d, mask)
    ox, oy, oz = project_vector_field(zero.copy(), zero.copy(), blob.copy(), d, d, d, 1e-9, 5000)
    div_after = _max_div(ox, oy, oz, d, mask)
    assert div_after < 1e-3 * div_before


def test_static_solver_recovers_manufactured_solution():
    """``elastic_solve_static`` must invert ``mu*lap(u) = -bz`` (not return 0)."""
    n, d, x, y, z, mask = _grid()
    mu = 1000.0
    u_true = (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)).astype(
        np.float32,
    ) * np.ones((n, n, n), np.float32)
    # bz = -mu * lap(u_true) so that the exact solution of mu*lap(u) = -bz is u_true.
    lap = np.zeros_like(u_true)
    c = u_true[1:-1, 1:-1, 1:-1]
    lap[1:-1, 1:-1, 1:-1] = (
        (u_true[2:, 1:-1, 1:-1] - 2 * c + u_true[:-2, 1:-1, 1:-1]) / d**2
        + (u_true[1:-1, 2:, 1:-1] - 2 * c + u_true[1:-1, :-2, 1:-1]) / d**2
        + (u_true[1:-1, 1:-1, 2:] - 2 * c + u_true[1:-1, 1:-1, :-2]) / d**2
    )
    bz = (-mu * lap).astype(np.float32)
    u = elastic_solve_static(bz, d, d, d, mu, tol=1e-8, max_iters=5000)
    assert np.max(np.abs(u)) > 0.0
    rel = np.max(np.abs((u - u_true) * mask)) / np.max(np.abs(u_true * mask))
    assert rel < 1e-3


@pytest.mark.parametrize("seed", [0, 1])
def test_projection_numpy_jax_parity_is_divergence_free(seed):
    """NumPy and JAX projections must agree *and* both be divergence-free."""
    pytest.importorskip("jax")
    from shearwave.solver import project_vector_field_jax

    n, d, x, y, z, mask = _grid(n=24)
    rng = np.random.default_rng(seed)
    # smooth, boundary-tapered random field
    taper = (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)).astype(np.float32)
    fx = (rng.standard_normal((n, n, n)).astype(np.float32) * taper) * np.ones((n, n, n), np.float32)
    fy = (rng.standard_normal((n, n, n)).astype(np.float32) * taper) * np.ones((n, n, n), np.float32)
    fz = (rng.standard_normal((n, n, n)).astype(np.float32) * taper) * np.ones((n, n, n), np.float32)
    nx, ny, nz = project_vector_field(fx.copy(), fy.copy(), fz.copy(), d, d, d, 1e-8, 5000)
    jx, jy, jz = (np.asarray(a) for a in project_vector_field_jax(fx, fy, fz, d, d, d, 1e-8, 5000))
    # backend agreement
    scale = np.max(np.abs(nx)) + 1e-20
    assert np.max(np.abs(nx - jx)) / scale < 1e-2
    # both divergence-free relative to the input divergence
    div_in = _max_div(fx, fy, fz, d, mask)
    assert _max_div(nx, ny, nz, d, mask) < 1e-2 * div_in
    assert _max_div(jx, jy, jz, d, mask) < 1e-2 * div_in
