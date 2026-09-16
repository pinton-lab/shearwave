"""3D collocated cell-centered shear FDTD solver with Kelvin-Voigt viscoelasticity.

Solves: rho * d^2u/dt^2 = mu * laplace(u) + eta * laplace(du/dt) + f_S

where f_S is the solenoidal (divergence-free) projection of the body force.
Uses central-difference time stepping and second-order spatial finite differences
with a Helmholtz-Hodge projection via conjugate-gradient Poisson solver.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.
"""

# ruff: noqa: ANN001, ANN201, ANN202, FBT002, FBT003, SIM108

import numpy as np

try:
    import jax.numpy as jnp
    from jax import jit, lax

    _HAS_JAX = True
except Exception:  # pragma: no cover - optional dependency
    jnp = None
    lax = None
    _HAS_JAX = False


def shear_fdtd(  # noqa: PLR0912
    bx,
    by,
    bz,
    rho,
    mu,
    dX,
    dY,
    dZ,
    dT,
    n_steps,
    opts=None,
    callback=None,
):
    """3D collocated cell-centered shear FDTD solver (CPU, numpy).

    Parameters
    ----------
    bx, by, bz : np.ndarray or None
        Body-force components with shape (nx, ny, nz) or (nx, ny, nz, n_steps).
        At least one component must be provided.
    rho, mu : float or np.ndarray
        Density and shear modulus. Scalars or arrays of shape (nx, ny, nz).
    dX, dY, dZ : float
        Grid spacing.
    dT : float
        Time step.
    n_steps : int
        Number of explicit steps.
    opts : dict, optional
        Solver options:
          - vel_damp (0.0)
          - eta (0.0) - Kelvin-Voigt viscosity [Pa*s]
          - env_t (None) - temporal envelope array of length n_steps. When
            provided, body forces are treated as 3D spatial patterns and the
            Helmholtz projection is pre-computed once before the time loop,
            then scaled by env_t[it] each step. This is exact for separable
            forces f(x,t) = b0(x)*g(t) and avoids per-step CG solves.
          - poisson_tol (1e-5)
          - poisson_max_iters (200)
          - reproject_every (0) - if > 0, re-apply the Helmholtz projection to
            the displacement history (u_curr and u_prev) every this many
            steps. The zero-Dirichlet box boundary is not solenoidal-
            preserving, so reflections from the domain edge inject
            divergence at a rate proportional to the wall-normal
            displacement; this option removes it at the cost of two CG
            solves per event. 0 (default) leaves the scheme unchanged.
          - verbose (False)
          - progress_every (max(1, n_steps//10))
    callback : callable(it, u_center), optional
        Invoked after displacement update each step; receives zero-based index.

    Returns
    -------
    u_center : np.ndarray (nx, ny, nz, 3)
        Displacement at cell centers.
    v_center : np.ndarray (nx, ny, nz, 3)
        Velocity at cell centers.

    """
    # Determine grid shape from any provided body-force component
    ref = bx if bx is not None else (by if by is not None else bz)
    if ref is None:
        msg = "At least one body-force component must be provided."
        raise ValueError(msg)

    opts = opts or {}
    vel_damp = opts.get("vel_damp", 0.0)
    poisson_tol = opts.get("poisson_tol", 1e-5)
    poisson_max_iters = opts.get("poisson_max_iters", 200)
    reproject_every = int(opts.get("reproject_every", 0) or 0)
    if reproject_every < 0:
        msg = "reproject_every must be a non-negative integer."
        raise ValueError(msg)
    verbose = opts.get("verbose", False)
    progress_every = opts.get("progress_every", max(1, n_steps // 10))

    # Shapes and basic fields
    nx, ny, nz = ref.shape[:3]
    env_t = opts.get("env_t")
    if env_t is not None:
        env_t = np.asarray(env_t, dtype=np.float32).ravel()
        if env_t.shape[0] != n_steps:
            msg = f"env_t length ({env_t.shape[0]}) must equal n_steps ({n_steps})."
            raise ValueError(msg)
        # Separable mode: body forces are 3D spatial patterns
        _z3d = np.zeros((nx, ny, nz), dtype=np.float32)
        bx_3d = _z3d if bx is None else np.asarray(bx, dtype=np.float32)
        by_3d = _z3d if by is None else np.asarray(by, dtype=np.float32)
        bz_3d = _z3d if bz is None else np.asarray(bz, dtype=np.float32)
        use_precomputed_projection = True
    else:
        bx = _prepare_body_component(bx, nx, ny, nz, n_steps)
        by = _prepare_body_component(by, nx, ny, nz, n_steps)
        bz = _prepare_body_component(bz, nx, ny, nz, n_steps)
        use_precomputed_projection = False

    rho_field = _prepare_material_field(rho, nx, ny, nz)
    mu_field = _prepare_material_field(mu, nx, ny, nz)
    mu_over_rho = mu_field / rho_field
    inv_rho = 1.0 / rho_field

    eta_field = opts.get("eta", 0.0)
    if np.isscalar(eta_field):
        eta_scalar = float(eta_field)
        if eta_scalar != 0.0:
            eta_field = _prepare_material_field(eta_scalar, nx, ny, nz)
        else:
            eta_field = None
    elif eta_field is None:
        pass
    else:
        eta_field = _prepare_material_field(eta_field, nx, ny, nz)
    if eta_field is not None:
        eta_over_rho = eta_field / rho_field
        has_viscosity = True
    else:
        eta_over_rho = None
        has_viscosity = False

    # Cell-centered displacement history (central difference scheme)
    u_prev = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    u_curr = np.zeros_like(u_prev)
    u_next = np.zeros_like(u_prev)

    mask_center = np.zeros((nx, ny, nz), dtype=np.float32)
    mask_center[1:-1, 1:-1, 1:-1] = 1.0

    dt2 = dT * dT

    initial_u = opts.get("initial_u")
    initial_v = opts.get("initial_v")
    if initial_u is not None:
        initial_u = np.asarray(initial_u, dtype=np.float32)
        if initial_u.shape != u_curr.shape:
            msg = "initial_u must match shape (nx,ny,nz,3)"
            raise ValueError(msg)
        u_curr = initial_u * mask_center[Ellipsis, None]
    if initial_v is not None:
        initial_v = np.asarray(initial_v, dtype=np.float32)
        if initial_v.shape != u_curr.shape:
            msg = "initial_v must match shape (nx,ny,nz,3)"
            raise ValueError(msg)
        u_prev = u_curr - dT * initial_v
        u_prev *= mask_center[Ellipsis, None]

    def project_force(fx, fy, fz):
        fx = fx.astype(np.float32, copy=True)
        fy = fy.astype(np.float32, copy=True)
        fz = fz.astype(np.float32, copy=True)
        return project_vector_field(
            fx,
            fy,
            fz,
            dX,
            dY,
            dZ,
            poisson_tol,
            poisson_max_iters,
        )

    # Pre-compute Helmholtz projection once for separable forces
    if use_precomputed_projection:
        if verbose:
            print("Pre-computing Helmholtz projection (separable force)...")
        fx0, fy0, fz0 = project_force(bx_3d, by_3d, bz_3d)

    for it in range(n_steps):
        if use_precomputed_projection:
            scale = env_t[it]
            fx, fy, fz = fx0 * scale, fy0 * scale, fz0 * scale
        else:
            fx, fy, fz = project_force(bx[..., it], by[..., it], bz[..., it])

        lap_u_x = laplacian_center(u_curr[..., 0], dX, dY, dZ)
        lap_u_y = laplacian_center(u_curr[..., 1], dX, dY, dZ)
        lap_u_z = laplacian_center(u_curr[..., 2], dX, dY, dZ)

        accel_x = mu_over_rho * lap_u_x + fx * inv_rho
        accel_y = mu_over_rho * lap_u_y + fy * inv_rho
        accel_z = mu_over_rho * lap_u_z + fz * inv_rho

        if has_viscosity:
            vel_curr = (u_curr - u_prev) / dT
            vel_curr *= mask_center[..., None]
            lap_v_x = laplacian_center(vel_curr[..., 0], dX, dY, dZ)
            lap_v_y = laplacian_center(vel_curr[..., 1], dX, dY, dZ)
            lap_v_z = laplacian_center(vel_curr[..., 2], dX, dY, dZ)
            accel_x += eta_over_rho * lap_v_x
            accel_y += eta_over_rho * lap_v_y
            accel_z += eta_over_rho * lap_v_z

        u_next[..., 0] = 2.0 * u_curr[..., 0] - u_prev[..., 0] + dt2 * accel_x
        u_next[..., 1] = 2.0 * u_curr[..., 1] - u_prev[..., 1] + dt2 * accel_y
        u_next[..., 2] = 2.0 * u_curr[..., 2] - u_prev[..., 2] + dt2 * accel_z

        if vel_damp > 0.0:
            u_next = u_curr + (1.0 - vel_damp) * (u_next - u_curr)

        u_next *= mask_center[Ellipsis, None]

        if callback is not None:
            callback(it, u_next)

        if verbose and (it + 1) % progress_every == 0:
            print(f"FDTD it={it + 1}/{n_steps}")

        u_prev, u_curr = u_curr, u_next.copy()

        if reproject_every and (it + 1) % reproject_every == 0:
            for arr in (u_curr, u_prev):
                px, py, pz = project_force(arr[..., 0], arr[..., 1], arr[..., 2])
                arr[..., 0], arr[..., 1], arr[..., 2] = px, py, pz

    u_center = u_curr.copy()
    v_center = (u_curr - u_prev) / dT
    return u_center, v_center


def shear_fdtd_jax(  # noqa: PLR0912
    bx,
    by,
    bz,
    rho,
    mu,
    dX,
    dY,
    dZ,
    dT,
    n_steps,
    opts=None,
    callback=None,
    trace_indices=None,
    return_traces=False,
    return_device=False,
    snapshot_stride=None,
):
    """3D collocated cell-centered shear FDTD solver using JAX.

    Returns either NumPy arrays (default) or device arrays if return_device=True.

    When ``opts["env_t"]`` is provided (a 1-D array of length *n_steps*), the
    body forces are treated as 3-D spatial patterns and the Helmholtz projection
    is pre-computed once, then scaled by ``env_t[it]`` each step.

    When ``snapshot_stride`` is provided (a positive int dividing ``n_steps``),
    the solver returns an additional array of shape
    ``(n_steps // snapshot_stride, nx, ny, nz, 3)`` holding ``u_curr`` at the
    end of every ``snapshot_stride`` steps. Snapshots stay on device when
    ``return_device=True`` and are converted to NumPy otherwise. The per-snap
    memory cost is ``nx * ny * nz * 3 * 4`` bytes (~17.7 MB at 145x101x101);
    callers control the time resolution via ``snapshot_stride``. Not supported
    together with ``return_traces=True`` in v1.
    """
    _require_jax()
    # Determine grid shape from any provided body-force component
    ref = bx if bx is not None else (by if by is not None else bz)
    if ref is None:
        msg = "At least one body-force component must be provided."
        raise ValueError(msg)
    if callback is not None:
        msg = "callback is not supported in shear_fdtd_jax."
        raise ValueError(msg)
    if return_traces and trace_indices is None:
        msg = "trace_indices must be provided when return_traces=True."
        raise ValueError(msg)
    if snapshot_stride is not None:
        if not isinstance(snapshot_stride, (int, np.integer)) or int(snapshot_stride) <= 0:
            msg = "snapshot_stride must be a positive integer."
            raise ValueError(msg)
        snapshot_stride = int(snapshot_stride)
        if n_steps % snapshot_stride != 0:
            msg = f"n_steps ({n_steps}) must be a multiple of snapshot_stride ({snapshot_stride})."
            raise ValueError(msg)
        if return_traces:
            msg = "return_traces and snapshot_stride are not both supported in v1."
            raise ValueError(msg)

    opts = opts or {}
    if opts.get("reproject_every"):
        msg = "reproject_every is only implemented in the NumPy solver (shear_fdtd)."
        raise NotImplementedError(msg)
    vel_damp = float(opts.get("vel_damp", 0.0))
    poisson_tol = float(opts.get("poisson_tol", 1e-5))
    poisson_max_iters = int(opts.get("poisson_max_iters", 200))
    use_jit = bool(opts.get("jit", True))

    nx, ny, nz = ref.shape[:3]
    env_t = opts.get("env_t")
    if env_t is not None:
        env_t_np = np.asarray(env_t, dtype=np.float32).ravel()
        if env_t_np.shape[0] != n_steps:
            msg = f"env_t length ({env_t_np.shape[0]}) must equal n_steps ({n_steps})."
            raise ValueError(msg)
        _z3d = np.zeros((nx, ny, nz), dtype=np.float32)
        bx_3d = _z3d if bx is None else np.asarray(bx, dtype=np.float32)
        by_3d = _z3d if by is None else np.asarray(by, dtype=np.float32)
        bz_3d = _z3d if bz is None else np.asarray(bz, dtype=np.float32)
        use_precomputed_projection = True
    else:
        bx = _prepare_body_component_jax(bx, nx, ny, nz, n_steps)
        by = _prepare_body_component_jax(by, nx, ny, nz, n_steps)
        bz = _prepare_body_component_jax(bz, nx, ny, nz, n_steps)
        use_precomputed_projection = False

    rho_field = _prepare_material_field_jax(rho, nx, ny, nz)
    mu_field = _prepare_material_field_jax(mu, nx, ny, nz)
    mu_over_rho = mu_field / rho_field
    inv_rho = 1.0 / rho_field

    eta_field = opts.get("eta", 0.0)
    if np.isscalar(eta_field):
        eta_scalar = float(eta_field)
        if eta_scalar != 0.0:
            eta_field = _prepare_material_field_jax(eta_scalar, nx, ny, nz)
        else:
            eta_field = None
    elif eta_field is None:
        pass
    else:
        eta_field = _prepare_material_field_jax(eta_field, nx, ny, nz)
    if eta_field is not None:
        eta_over_rho = eta_field / rho_field
        has_viscosity = True
    else:
        eta_over_rho = None
        has_viscosity = False

    u_prev = jnp.zeros((nx, ny, nz, 3), dtype=jnp.float32)
    u_curr = jnp.zeros_like(u_prev)
    mask_center = jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[1:-1, 1:-1, 1:-1].set(1.0)
    mask_center_4 = mask_center[..., None]
    dt2 = dT * dT

    initial_u = opts.get("initial_u")
    initial_v = opts.get("initial_v")
    if initial_u is not None:
        initial_u = np.asarray(initial_u, dtype=np.float32)
        if initial_u.shape != (nx, ny, nz, 3):
            msg = "initial_u must match shape (nx,ny,nz,3)"
            raise ValueError(msg)
        u_curr = jnp.asarray(initial_u) * mask_center_4
    if initial_v is not None:
        initial_v = np.asarray(initial_v, dtype=np.float32)
        if initial_v.shape != (nx, ny, nz, 3):
            msg = "initial_v must match shape (nx,ny,nz,3)"
            raise ValueError(msg)
        u_prev = (u_curr - dT * jnp.asarray(initial_v)) * mask_center_4

    # Pre-compute or transpose body forces for scan
    if use_precomputed_projection:
        fx0_np, fy0_np, fz0_np = project_vector_field(
            bx_3d,
            by_3d,
            bz_3d,
            dX,
            dY,
            dZ,
            poisson_tol,
            poisson_max_iters,
        )
        fx0 = jnp.asarray(fx0_np)
        fy0 = jnp.asarray(fy0_np)
        fz0 = jnp.asarray(fz0_np)
        env_t_jax = jnp.asarray(env_t_np)
    else:
        bx_t = jnp.transpose(bx, (3, 0, 1, 2))
        by_t = jnp.transpose(by, (3, 0, 1, 2))
        bz_t = jnp.transpose(bz, (3, 0, 1, 2))
    if trace_indices is not None:
        if not isinstance(trace_indices, (tuple, list)) or len(trace_indices) != 3:
            msg = "trace_indices must be a 3-tuple: (ix, iy, iz)."
            raise ValueError(msg)
        ix_np = np.asarray(trace_indices[0], dtype=np.int32)
        iy_np = np.asarray(trace_indices[1], dtype=np.int32)
        iz_np = np.asarray(trace_indices[2], dtype=np.int32)
        if not (ix_np.shape == iy_np.shape == iz_np.shape):
            msg = "trace index arrays must share the same shape."
            raise ValueError(msg)
        if ix_np.ndim != 1:
            msg = "trace index arrays must be 1D."
            raise ValueError(msg)
        ix = jnp.asarray(ix_np, dtype=jnp.int32)
        iy = jnp.asarray(iy_np, dtype=jnp.int32)
        iz = jnp.asarray(iz_np, dtype=jnp.int32)
    else:
        ix = iy = iz = None

    def _fdtd_core(u_prev_s, u_curr_s, fx, fy, fz):
        lap_u_x = laplacian_center_jax(u_curr_s[..., 0], dX, dY, dZ)
        lap_u_y = laplacian_center_jax(u_curr_s[..., 1], dX, dY, dZ)
        lap_u_z = laplacian_center_jax(u_curr_s[..., 2], dX, dY, dZ)

        accel_x = mu_over_rho * lap_u_x + fx * inv_rho
        accel_y = mu_over_rho * lap_u_y + fy * inv_rho
        accel_z = mu_over_rho * lap_u_z + fz * inv_rho

        if has_viscosity:
            vel_curr = ((u_curr_s - u_prev_s) / dT) * mask_center_4
            lap_v_x = laplacian_center_jax(vel_curr[..., 0], dX, dY, dZ)
            lap_v_y = laplacian_center_jax(vel_curr[..., 1], dX, dY, dZ)
            lap_v_z = laplacian_center_jax(vel_curr[..., 2], dX, dY, dZ)
            accel_x = accel_x + eta_over_rho * lap_v_x
            accel_y = accel_y + eta_over_rho * lap_v_y
            accel_z = accel_z + eta_over_rho * lap_v_z

        u_next = jnp.stack(
            (
                2.0 * u_curr_s[..., 0] - u_prev_s[..., 0] + dt2 * accel_x,
                2.0 * u_curr_s[..., 1] - u_prev_s[..., 1] + dt2 * accel_y,
                2.0 * u_curr_s[..., 2] - u_prev_s[..., 2] + dt2 * accel_z,
            ),
            axis=-1,
        )
        if vel_damp > 0.0:
            u_next = u_curr_s + (1.0 - vel_damp) * (u_next - u_curr_s)
        u_next = u_next * mask_center_4
        if return_traces:
            traces_t = u_next[ix, iy, iz, 2]
        else:
            traces_t = jnp.zeros((0,), dtype=jnp.float32)
        return u_next, traces_t

    def _force_at(idx):
        """Return (fx, fy, fz) at step ``idx``, projected and ready for _fdtd_core."""
        if use_precomputed_projection:
            scale = env_t_jax[idx]
            return fx0 * scale, fy0 * scale, fz0 * scale
        fx_in = lax.dynamic_index_in_dim(bx_t, idx, axis=0, keepdims=False)
        fy_in = lax.dynamic_index_in_dim(by_t, idx, axis=0, keepdims=False)
        fz_in = lax.dynamic_index_in_dim(bz_t, idx, axis=0, keepdims=False)
        return project_vector_field_jax(
            fx_in,
            fy_in,
            fz_in,
            dX,
            dY,
            dZ,
            poisson_tol,
            poisson_max_iters,
        )

    if snapshot_stride is not None:
        stride = snapshot_stride
        n_snap = n_steps // stride

        def inner_body(i_inner, carry):
            base, u_prev_s, u_curr_s = carry
            idx = base + i_inner
            fx, fy, fz = _force_at(idx)
            u_next, _ = _fdtd_core(u_prev_s, u_curr_s, fx, fy, fz)
            return (base, u_curr_s, u_next)

        def outer_step(carry, base):
            u_prev_s, u_curr_s = carry
            _, u_prev_new, u_curr_new = lax.fori_loop(
                0,
                stride,
                inner_body,
                (base, u_prev_s, u_curr_s),
            )
            return (u_prev_new, u_curr_new), u_curr_new

        def run_snapshot_scan(u_prev0, u_curr0):
            bases = jnp.arange(0, n_snap * stride, stride, dtype=jnp.int32)
            (u_prev_f, u_curr_f), snaps = lax.scan(
                outer_step,
                (u_prev0, u_curr0),
                bases,
            )
            return u_prev_f, u_curr_f, snaps

        if use_jit:
            run_snapshot_scan = jit(run_snapshot_scan)

        u_prev_f, u_curr_f, snapshots = run_snapshot_scan(u_prev, u_curr)
        u_center = u_curr_f
        v_center = (u_curr_f - u_prev_f) / dT
        if return_device:
            return u_center, v_center, snapshots
        return np.asarray(u_center), np.asarray(v_center), np.asarray(snapshots)

    if use_precomputed_projection:

        def step(carry, env_val):
            u_prev_s, u_curr_s = carry
            fx, fy, fz = fx0 * env_val, fy0 * env_val, fz0 * env_val
            u_next, traces_t = _fdtd_core(u_prev_s, u_curr_s, fx, fy, fz)
            return (u_curr_s, u_next), traces_t

        def run_scan(u_prev0, u_curr0):
            (u_prev_f, u_curr_f), traces = lax.scan(
                step,
                (u_prev0, u_curr0),
                env_t_jax,
            )
            return u_prev_f, u_curr_f, traces
    else:

        def step(carry, forces_t):
            u_prev_s, u_curr_s = carry
            fx_in, fy_in, fz_in = forces_t
            fx, fy, fz = project_vector_field_jax(
                fx_in,
                fy_in,
                fz_in,
                dX,
                dY,
                dZ,
                poisson_tol,
                poisson_max_iters,
            )
            u_next, traces_t = _fdtd_core(u_prev_s, u_curr_s, fx, fy, fz)
            return (u_curr_s, u_next), traces_t

        def run_scan(u_prev0, u_curr0):
            (u_prev_f, u_curr_f), traces = lax.scan(
                step,
                (u_prev0, u_curr0),
                (bx_t, by_t, bz_t),
            )
            return u_prev_f, u_curr_f, traces

    if use_jit:
        run_scan = jit(run_scan)

    u_prev_f, u_curr_f, traces = run_scan(u_prev, u_curr)
    u_center = u_curr_f
    v_center = (u_curr_f - u_prev_f) / dT
    if return_device:
        if return_traces:
            return u_center, v_center, jnp.transpose(traces, (1, 0))
        return u_center, v_center
    if return_traces:
        return (
            np.asarray(u_center),
            np.asarray(v_center),
            np.asarray(jnp.transpose(traces, (1, 0))),
        )
    return np.asarray(u_center), np.asarray(v_center)


def elastic_solve_static(
    bz,
    dX,
    dY,
    dZ,
    mu,
    tol=1e-6,
    max_iters=1500,
    *,
    verbose=False,
):
    """Solve mu * laplace(u) = -bz with zero Dirichlet boundary via CG.

    CG requires an SPD operator.  ``mu * laplace`` is negative-definite, so we
    solve the equivalent SPD system ``(-mu * laplace) u = bz`` (i.e. we negate
    both the operator and the right-hand side).  The previous formulation used
    ``+mu * laplace`` with rhs ``-bz``, which is negative-definite and caused
    the CG loop to terminate on the first iteration (``denom <= 0``), returning
    an all-zero displacement.
    """
    nx, ny, nz = bz.shape
    mu_field = _prepare_material_field(mu, nx, ny, nz)
    rhs = bz.astype(np.float32)
    u = np.zeros_like(rhs, dtype=np.float32)
    mask = np.zeros_like(rhs, dtype=np.float32)
    mask[1:-1, 1:-1, 1:-1] = 1.0

    def apply_a(x):
        return -mu_field * laplacian_center(x, dX, dY, dZ)

    r = rhs - apply_a(u)
    r *= mask
    p = r.copy()
    rs_old = np.sum(r * r)
    rs0 = max(rs_old, np.finfo(np.float32).eps)

    for it in range(max_iters):
        a_p = apply_a(p)
        denom = np.sum(p * a_p)
        if denom <= 0.0:
            break
        alpha = rs_old / denom
        u = (u + alpha * p) * mask
        r = (r - alpha * a_p) * mask
        rs_new = np.sum(r * r)
        rel = np.sqrt(rs_new / rs0)
        if verbose and (it == 0 or (it + 1) % 25 == 0 or rel < tol):
            print(
                f"CG (CPU): it={it + 1}/{max_iters}, |r|={np.sqrt(rs_new):.3e}, rel={rel:.3e}",
            )
        if rel < tol:
            break
        beta = rs_new / max(rs_old, np.finfo(np.float32).eps)
        p = (r + beta * p) * mask
        rs_old = rs_new

    return u


def project_body_force_to_shear(
    bz,
    dX,
    dY,
    dZ,
    tol=5e-5,
    max_iters=400,
):
    """Project body force (initially only z component) to be divergence-free."""
    bz = bz.astype(np.float32)
    nx, ny, nz = bz.shape
    fx = np.zeros_like(bz)
    fy = np.zeros_like(bz)
    return project_vector_field(fx, fy, bz.copy(), dX, dY, dZ, tol, max_iters)


def project_vector_field(
    fx,
    fy,
    fz,
    dX,
    dY,
    dZ,
    tol,
    max_iters,
):
    """Project a vector field to be divergence-free (Helmholtz decomposition).

    The field is restricted to the interior (zeroed on the one-voxel boundary
    layer) *before* its divergence is taken, so that the divergence the
    Poisson solve removes is the divergence of the field that is actually
    returned.  Taking the divergence first and masking afterwards leaves a
    residual ``f_boundary / (2h)`` in the layer adjacent to the domain edge
    whenever the input is non-zero on the boundary layer.
    """
    nx, ny, nz = fx.shape
    mask_center = np.zeros((nx, ny, nz), dtype=np.float32)
    mask_center[1:-1, 1:-1, 1:-1] = 1.0
    fx = fx.astype(np.float32, copy=False) * mask_center
    fy = fy.astype(np.float32, copy=False) * mask_center
    fz = fz.astype(np.float32, copy=False) * mask_center

    div_f = divergence_center(fx, fy, fz, dX, dY, dZ)
    div_f *= mask_center
    if np.max(np.abs(div_f)) < 1e-12:
        fx *= mask_center
        fy *= mask_center
        fz *= mask_center
        return fx, fy, fz

    phi = poisson_cg(-div_f, dX, dY, dZ, mask_center, tol=tol, max_iters=max_iters)
    gx, gy, gz = gradient_center(phi, dX, dY, dZ)
    fx = (fx - gx) * mask_center
    fy = (fy - gy) * mask_center
    fz = (fz - gz) * mask_center
    return fx, fy, fz


def laplacian_center(field, dX, dY, dZ):
    """Compute Laplacian at cell centers using second-order finite differences."""
    dtype = field.dtype
    lap = np.zeros_like(field, dtype=dtype)
    lap[1:-1, 1:-1, 1:-1] = (
        (field[2:, 1:-1, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / (dX**2)
        + (field[1:-1, 2:, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / (dY**2)
        + (field[1:-1, 1:-1, 2:] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / (dZ**2)
    )
    return lap


def divergence_faces_to_center(vx, vy, vz, dX, dY, dZ):
    """Compute divergence from face-centered velocities to cell centers."""
    dtype = np.result_type(vx.dtype, vy.dtype, vz.dtype)
    div = np.zeros((vx.shape[0] - 1, vy.shape[1] - 1, vz.shape[2] - 1), dtype=dtype)
    div += (vx[1:, :, :] - vx[:-1, :, :]) / dX
    div += (vy[:, 1:, :] - vy[:, :-1, :]) / dY
    div += (vz[:, :, 1:] - vz[:, :, :-1]) / dZ
    return div


def divergence_center(fx, fy, fz, dX, dY, dZ):
    """Compute divergence at cell centers using central differences."""
    dtype = np.result_type(fx.dtype, fy.dtype, fz.dtype)
    div = np.zeros_like(fx, dtype=dtype)
    div[1:-1, 1:-1, 1:-1] = (
        (fx[2:, 1:-1, 1:-1] - fx[:-2, 1:-1, 1:-1]) / (2.0 * dX)
        + (fy[1:-1, 2:, 1:-1] - fy[1:-1, :-2, 1:-1]) / (2.0 * dY)
        + (fz[1:-1, 1:-1, 2:] - fz[1:-1, 1:-1, :-2]) / (2.0 * dZ)
    )
    return div


def gradient_center(phi, dX, dY, dZ):
    """Compute gradient at cell centers using central differences."""
    dtype = phi.dtype
    gx = np.zeros_like(phi, dtype=dtype)
    gy = np.zeros_like(phi, dtype=dtype)
    gz = np.zeros_like(phi, dtype=dtype)
    gx[1:-1, 1:-1, 1:-1] = (phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / (2.0 * dX)
    gy[1:-1, 1:-1, 1:-1] = (phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) / (2.0 * dY)
    gz[1:-1, 1:-1, 1:-1] = (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, :-2]) / (2.0 * dZ)
    return gx, gy, gz


# --- JAX variants ---


def _require_jax():
    if not _HAS_JAX:
        msg = "JAX is required for *_jax functions. Install with `pip install jax`."
        raise ImportError(msg)


def _prepare_body_component_jax(comp, nx, ny, nz, n_steps):
    _require_jax()
    if comp is None:
        return jnp.zeros((nx, ny, nz, n_steps), dtype=jnp.float32)
    comp_np = np.asarray(comp, dtype=np.float32)
    if comp_np.ndim == 3:
        comp_np = np.repeat(comp_np[..., np.newaxis], n_steps, axis=3)
    if comp_np.ndim != 4 or comp_np.shape != (nx, ny, nz, n_steps):
        msg = "Body-force component has incompatible shape."
        raise ValueError(msg)
    return jnp.asarray(comp_np)


def _prepare_material_field_jax(field, nx, ny, nz):
    _require_jax()
    if np.isscalar(field):
        return jnp.full((nx, ny, nz), float(field), dtype=jnp.float32)
    field_np = np.asarray(field, dtype=np.float32)
    if field_np.shape != (nx, ny, nz):
        msg = "Material field has incompatible shape."
        raise ValueError(msg)
    return jnp.asarray(field_np)


def laplacian_center_jax(field, dX, dY, dZ):
    """Compute Laplacian at cell centers (JAX)."""
    _require_jax()
    field = jnp.asarray(field)
    lap_inner = (
        (field[2:, 1:-1, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / (dX**2)
        + (field[1:-1, 2:, 1:-1] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / (dY**2)
        + (field[1:-1, 1:-1, 2:] - 2.0 * field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / (dZ**2)
    )
    return jnp.zeros_like(field).at[1:-1, 1:-1, 1:-1].set(lap_inner)


def divergence_center_jax(fx, fy, fz, dX, dY, dZ):
    """Compute divergence at cell centers (JAX)."""
    _require_jax()
    fx = jnp.asarray(fx)
    fy = jnp.asarray(fy)
    fz = jnp.asarray(fz)
    div_inner = (
        (fx[2:, 1:-1, 1:-1] - fx[:-2, 1:-1, 1:-1]) / (2.0 * dX)
        + (fy[1:-1, 2:, 1:-1] - fy[1:-1, :-2, 1:-1]) / (2.0 * dY)
        + (fz[1:-1, 1:-1, 2:] - fz[1:-1, 1:-1, :-2]) / (2.0 * dZ)
    )
    return jnp.zeros_like(fx).at[1:-1, 1:-1, 1:-1].set(div_inner)


def gradient_center_jax(phi, dX, dY, dZ):
    """Compute gradient at cell centers (JAX)."""
    _require_jax()
    phi = jnp.asarray(phi)
    gx = jnp.zeros_like(phi)
    gy = jnp.zeros_like(phi)
    gz = jnp.zeros_like(phi)
    gx = gx.at[1:-1, 1:-1, 1:-1].set(
        (phi[2:, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1]) / (2.0 * dX),
    )
    gy = gy.at[1:-1, 1:-1, 1:-1].set(
        (phi[1:-1, 2:, 1:-1] - phi[1:-1, :-2, 1:-1]) / (2.0 * dY),
    )
    gz = gz.at[1:-1, 1:-1, 1:-1].set(
        (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, :-2]) / (2.0 * dZ),
    )
    return gx, gy, gz


def poisson_cg_jax(rhs, dX, dY, dZ, mask_center, tol=1e-4, max_iters=200):
    """Solve ``div(grad(phi)) = -rhs`` via CG (JAX; see :func:`poisson_cg`)."""
    _require_jax()
    rhs = jnp.asarray(rhs)
    dtype = rhs.dtype
    mask = jnp.asarray(mask_center, dtype=dtype)
    phi0 = jnp.zeros_like(rhs)

    def apply_a(x):
        gx, gy, gz = gradient_center_jax(x, dX, dY, dZ)
        return -divergence_center_jax(gx, gy, gz, dX, dY, dZ)

    r0 = (rhs - apply_a(phi0)) * mask
    p0 = r0
    rs_old0 = jnp.sum(r0 * r0)
    eps = jnp.finfo(dtype).eps
    rs0 = jnp.maximum(rs_old0, eps)

    def body_fun(_, state):
        phi, r, p, rs_old, active = state
        a_p = apply_a(p) * mask
        denom = jnp.sum(p * a_p)
        valid_denom = jnp.isfinite(denom) & (denom > 0.0)
        safe_denom = jnp.where(valid_denom, denom, jnp.array(1.0, dtype=dtype))

        alpha = rs_old / safe_denom
        phi_new = (phi + alpha * p) * mask
        r_new = (r - alpha * a_p) * mask
        rs_new = jnp.sum(r_new * r_new)
        finite_rs = jnp.isfinite(rs_new)
        rel = jnp.sqrt(rs_new / rs0)
        can_update = valid_denom & finite_rs
        keep_iterating = can_update & (rel >= tol)

        beta = rs_new / jnp.maximum(rs_old, eps)
        p_new = (r_new + beta * p) * mask

        do_update = active & can_update
        phi_out = jnp.where(do_update, phi_new, phi)
        r_out = jnp.where(do_update, r_new, r)
        p_out = jnp.where(do_update, p_new, p)
        rs_old_out = jnp.where(do_update, rs_new, rs_old)
        active_out = active & keep_iterating
        return phi_out, r_out, p_out, rs_old_out, active_out

    phi, _, _, _, _ = lax.fori_loop(
        0,
        int(max_iters),
        body_fun,
        (phi0, r0, p0, rs_old0, jnp.array(True)),
    )
    return phi.astype(dtype)


def project_vector_field_jax(
    fx,
    fy,
    fz,
    dX,
    dY,
    dZ,
    tol,
    max_iters,
):
    """Project a vector field to be divergence-free (JAX).

    See :func:`project_vector_field`: the input is restricted to the interior
    before its divergence is taken.
    """
    _require_jax()
    nx, ny, nz = fx.shape
    mask_center = jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[1:-1, 1:-1, 1:-1].set(1.0)
    fx = jnp.asarray(fx, dtype=jnp.float32) * mask_center
    fy = jnp.asarray(fy, dtype=jnp.float32) * mask_center
    fz = jnp.asarray(fz, dtype=jnp.float32) * mask_center

    div_f = divergence_center_jax(fx, fy, fz, dX, dY, dZ) * mask_center

    def no_projection(_):
        return fx * mask_center, fy * mask_center, fz * mask_center

    def do_projection(_):
        phi = poisson_cg_jax(-div_f, dX, dY, dZ, mask_center, tol=tol, max_iters=max_iters)
        gx, gy, gz = gradient_center_jax(phi, dX, dY, dZ)
        return (fx - gx) * mask_center, (fy - gy) * mask_center, (fz - gz) * mask_center

    return lax.cond(
        jnp.max(jnp.abs(div_f)) < 1e-12,
        no_projection,
        do_projection,
        operand=None,
    )


def project_body_force_to_shear_jax(
    bz,
    dX,
    dY,
    dZ,
    tol=5e-5,
    max_iters=400,
):
    """Project body force to be divergence-free (JAX)."""
    _require_jax()
    bz = jnp.asarray(bz, dtype=jnp.float32)
    fx = jnp.zeros_like(bz)
    fy = jnp.zeros_like(bz)
    return project_vector_field_jax(fx, fy, bz, dX, dY, dZ, tol, max_iters)


def average_to_face_x(center):
    """Average cell-center values to x-faces."""
    nx, ny, nz = center.shape
    face = np.zeros((nx + 1, ny, nz), dtype=center.dtype)
    face[1:nx, :, :] = 0.5 * (center[1:, :, :] + center[:-1, :, :])
    return face


def average_to_face_y(center):
    """Average cell-center values to y-faces."""
    nx, ny, nz = center.shape
    face = np.zeros((nx, ny + 1, nz), dtype=center.dtype)
    face[:, 1:ny, :] = 0.5 * (center[:, 1:, :] + center[:, :-1, :])
    return face


def average_to_face_z(center):
    """Average cell-center values to z-faces."""
    nx, ny, nz = center.shape
    face = np.zeros((nx, ny, nz + 1), dtype=center.dtype)
    face[:, :, 1:nz] = 0.5 * (center[:, :, 1:] + center[:, :, :-1])
    return face


def face_mu_over_rho(mu_over_rho, axis):
    """Average mu/rho from cell centers to faces along given axis."""
    nx, ny, nz = mu_over_rho.shape
    if axis == "x":
        acc = np.zeros((nx + 1, ny, nz), dtype=mu_over_rho.dtype)
        acc[1:nx, :, :] = 0.5 * (mu_over_rho[1:, :, :] + mu_over_rho[:-1, :, :])
    elif axis == "y":
        acc = np.zeros((nx, ny + 1, nz), dtype=mu_over_rho.dtype)
        acc[:, 1:ny, :] = 0.5 * (mu_over_rho[:, 1:, :] + mu_over_rho[:, :-1, :])
    elif axis == "z":
        acc = np.zeros((nx, ny, nz + 1), dtype=mu_over_rho.dtype)
        acc[:, :, 1:nz] = 0.5 * (mu_over_rho[:, :, 1:] + mu_over_rho[:, :, :-1])
    else:
        msg = "axis must be 'x', 'y', or 'z'"
        raise ValueError(msg)
    return acc


def project_face_forces(
    bx_face,
    by_face,
    bz_face,
    dX,
    dY,
    dZ,
    mask_x,
    mask_y,
    mask_z,
    mask_center,
    tol,
    max_iters,
):
    """Project face-centered forces to be divergence-free.

    NOTE: currently unused (no call site).  It also predates the fix to
    :func:`poisson_cg`, which now inverts the collocated ``div_center(grad_center)``
    operator rather than the face-based Laplacian implied by these staggered
    differences.  If this staggered projection path is revived, ``poisson_cg``
    must be given a face-consistent operator (and the RHS sign checked); as
    written the two operators are mismatched.
    """
    div_q = (
        (bx_face[1:, :, :] - bx_face[:-1, :, :]) / dX
        + (by_face[:, 1:, :] - by_face[:, :-1, :]) / dY
        + (bz_face[:, :, 1:] - bz_face[:, :, :-1]) / dZ
    )
    div_q *= mask_center

    if np.max(np.abs(div_q)) < 1e-12:
        return bx_face * mask_x, by_face * mask_y, bz_face * mask_z

    phi = poisson_cg(div_q, dX, dY, dZ, mask_center, tol=tol, max_iters=max_iters)
    grad_phi_x = (phi[1:, :, :] - phi[:-1, :, :]) / dX
    grad_phi_y = (phi[:, 1:, :] - phi[:, :-1, :]) / dY
    grad_phi_z = (phi[:, :, 1:] - phi[:, :, :-1]) / dZ

    bx_face[1:-1, :, :] -= grad_phi_x
    by_face[:, 1:-1, :] -= grad_phi_y
    bz_face[:, :, 1:-1] -= grad_phi_z

    bx_face *= mask_x
    by_face *= mask_y
    bz_face *= mask_z
    return bx_face, by_face, bz_face


def poisson_cg(rhs, dX, dY, dZ, mask_center, tol=1e-4, max_iters=200):
    """Solve the discrete Poisson equation ``div(grad(phi)) = -rhs`` via CG.

    The operator is the composition ``div_center(grad_center(.))`` (not the
    compact 7-point ``laplacian_center``).  Using the same first-difference
    div/grad operators that the Helmholtz projection applies is what makes
    ``f - grad(phi)`` discretely divergence-free; inverting the compact
    Laplacian instead leaves a residual divergence.  Callers pass ``-div_f`` so
    that ``phi`` satisfies ``div(grad(phi)) = div_f``.
    """
    rhs64 = rhs.astype(np.float64, copy=False)
    mask64 = mask_center.astype(np.float64, copy=False)
    phi = np.zeros_like(rhs64)

    def apply_a(x):
        gx, gy, gz = gradient_center(x, dX, dY, dZ)
        return -divergence_center(gx, gy, gz, dX, dY, dZ)

    r = (rhs64 - apply_a(phi)) * mask64
    p = r.copy()
    rs_old = np.sum(r * r)
    rs0 = max(rs_old, np.finfo(np.float64).eps)

    for _ in range(max_iters):
        a_p = apply_a(p) * mask64
        denom = np.sum(p * a_p)
        if not np.isfinite(denom) or denom <= 0.0:
            break
        alpha = rs_old / denom
        phi = (phi + alpha * p) * mask64
        r = (r - alpha * a_p) * mask64
        rs_new = np.sum(r * r)
        if not np.isfinite(rs_new):
            break
        if np.sqrt(rs_new / rs0) < tol:
            break
        beta = rs_new / max(rs_old, np.finfo(np.float64).eps)
        p = (r + beta * p) * mask64
        rs_old = rs_new

    return phi.astype(rhs.dtype, copy=False)


def hann1d(n):
    """Return a 1D Hann window of length n."""
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    return np.hanning(n).astype(np.float32)


def _prepare_body_component(comp, nx, ny, nz, n_steps):
    if comp is None:
        return np.zeros((nx, ny, nz, n_steps), dtype=np.float32)
    comp = np.asarray(comp, dtype=np.float32)
    if comp.ndim == 3:
        comp = np.repeat(comp[..., np.newaxis], n_steps, axis=3)
    if comp.ndim != 4 or comp.shape != (nx, ny, nz, n_steps):
        msg = "Body-force component has incompatible shape."
        raise ValueError(msg)
    return comp


def _prepare_material_field(field, nx, ny, nz):
    if np.isscalar(field):
        return np.full((nx, ny, nz), float(field), dtype=np.float32)
    field = np.asarray(field, dtype=np.float32)
    if field.shape != (nx, ny, nz):
        msg = "Material field has incompatible shape."
        raise ValueError(msg)
    return field
