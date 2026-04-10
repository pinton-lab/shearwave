"""Quasi-static validation for the shear FDTD solver.

Applies a constant body force with strong velocity damping so the FDTD
solution converges to the static elasticity solution mu*laplace(u) = -b.
Compares the converged FDTD displacement to a CG-based static solve.

Reference: Section 4.5 of the shear FDTD physics derivation document.
"""

import numpy as np

from shearwave import (
    elastic_solve_static,
    hann1d,
    laplacian_center,
    project_body_force_to_shear,
    shear_fdtd_staggered,
)


def main():
    # Grid and material setup
    dX = 2.0e-4
    dY = dX
    dZ = dX
    nX = 72
    nY = 48
    nZ = 72
    rho = 1000.0
    cs = 2.0
    mu = rho * cs**2

    CFL = 0.5
    dT = CFL * min(dX, dY, dZ) / (cs * np.sqrt(3.0))
    Tmax = 7e-4
    n_steps = int(np.ceil(Tmax / dT))

    # Body force: tapered 3D Gaussian
    xg, yg, zg = np.meshgrid(
        np.arange(nX) - nX // 2,
        np.arange(nY) - nY // 2,
        np.arange(nZ) - nZ // 2,
        indexing="ij",
    )
    sigma_xy = 8.0
    sigma_z = 6.0
    Gxyz = np.exp(
        -0.5 * ((xg / sigma_xy) ** 2 + (yg / sigma_xy) ** 2 + (zg / sigma_z) ** 2)
    ).astype(np.float32)
    Gxyz /= np.max(Gxyz)
    wx = hann1d(nX)
    wy = hann1d(nY)
    wz = hann1d(nZ)
    taper = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    Gxyz *= taper
    b_amp = 0.02
    b0 = b_amp * Gxyz

    print("Projecting body force to shear-only component...")
    bx_shear, by_shear, bz_shear = project_body_force_to_shear(
        b0, dX, dY, dZ, tol=5e-5, max_iters=400
    )

    bx = np.repeat(bx_shear[..., np.newaxis], n_steps, axis=3)
    by = np.repeat(by_shear[..., np.newaxis], n_steps, axis=3)
    bz = np.repeat(bz_shear[..., np.newaxis], n_steps, axis=3)

    opts = {
        "vel_damp": 0.05,
        "poisson_tol": 5e-5,
        "poisson_max_iters": 400,
        "verbose": True,
        "progress_every": max(1, n_steps // 5),
    }

    print("Running staggered-grid shear FDTD...")
    u_fdtd, v_fdtd = shear_fdtd_staggered(bx, by, bz, rho, mu, dX, dY, dZ, dT, n_steps, opts=opts)
    u_fdtd_z = u_fdtd[..., 2]
    v_norm = np.linalg.norm(v_fdtd.reshape(-1, 3))
    print(f"Final velocity L2 norm = {v_norm:.3e} m/s")

    print("Solving static elasticity for reference...")
    u_static_z = elastic_solve_static(
        bz_shear, dX, dY, dZ, mu, tol=1e-6, max_iters=1500, verbose=True
    )

    diff = u_fdtd_z - u_static_z
    rel_err = np.linalg.norm(diff.ravel()) / max(
        np.linalg.norm(u_static_z.ravel()), np.finfo(np.float32).eps
    )
    max_abs_diff = np.max(np.abs(diff))
    print(f"L2 relative error = {rel_err:.3e}")
    print(f"Max absolute difference = {max_abs_diff:.3e} m")

    lap_fdtd = laplacian_center(u_fdtd_z, dX, dY, dZ)
    lap_static = laplacian_center(u_static_z, dX, dY, dZ)
    res_fdtd = mu * lap_fdtd + b0
    res_static = mu * lap_static + b0
    norm_b = max(np.linalg.norm(b0.ravel()), np.finfo(np.float32).eps)
    rn_fdtd = np.linalg.norm(res_fdtd.ravel())
    rn_static = np.linalg.norm(res_static.ravel())
    print(f"Residual norms: FDTD {rn_fdtd:.3e}, Static {rn_static:.3e}")
    print(f"Residual rel norms: FDTD {rn_fdtd / norm_b:.3e}, Static {rn_static / norm_b:.3e}")

    if rel_err < 5e-2:
        print("Validation: PASS (FDTD converges to static solution within tolerance)")
    else:
        print("Validation: CHECK (difference exceeds tolerance)")


if __name__ == "__main__":
    main()
