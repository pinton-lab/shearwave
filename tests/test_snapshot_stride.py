"""Tests for ``snapshot_stride`` in-loop snapshot collection in the JAX solver.

Cross-checks numpy callback snapshots against JAX ``snapshot_stride`` output in
both 4D and separable (``env_t``) modes, and exercises the documented error
paths. The body forcing is a chirp-modulated Gaussian whose amplitude varies
~30% RMS frame-to-frame, so any off-by-one in the nested-loop force indexing
would dominate the snapshot diff rather than being hidden by zero forcing.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from shearwave.solver import shear_fdtd_staggered, shear_fdtd_staggered_jax


def _build_problem(nx=16, ny=12, nz=16, n_steps=24):
    rho, mu = 1000.0, 4000.0
    dX = dY = dZ = 2.0e-4
    cs = float(np.sqrt(mu / rho))
    dT = 0.15 * dX / (cs * np.sqrt(3.0))

    xg, yg, zg = np.meshgrid(
        np.arange(nx) - nx // 2,
        np.arange(ny) - ny // 2,
        np.arange(nz) - nz // 2,
        indexing="ij",
    )
    sigma = 3.0
    b3 = np.exp(
        -0.5 * ((xg / sigma) ** 2 + (yg / sigma) ** 2 + (zg / sigma) ** 2),
    ).astype(np.float32)
    b3 /= np.max(b3)
    b3 *= np.float32(0.03)

    t = np.arange(n_steps, dtype=np.float32)
    env = np.sin(2.0 * np.pi * 0.07 * t) * (1.0 + 0.5 * np.cos(2.0 * np.pi * 0.19 * t))
    env = env.astype(np.float32)

    bz_4d = (b3[..., None] * env[None, None, None, :]).astype(np.float32)
    opts = {"vel_damp": 0.0, "poisson_tol": 5e-5, "poisson_max_iters": 120, "verbose": False}
    grid = {
        "rho": rho,
        "mu": mu,
        "dX": dX,
        "dY": dY,
        "dZ": dZ,
        "dT": dT,
        "n_steps": n_steps,
    }
    return b3, env, bz_4d, opts, grid


def _numpy_snapshots(bz_4d, opts, grid, stride):
    snaps = []

    def cb(it, u_center):
        if (it + 1) % stride == 0:
            snaps.append(np.array(u_center, copy=True))

    u_np, _v = shear_fdtd_staggered(
        None,
        None,
        bz_4d,
        opts=opts,
        callback=cb,
        **grid,
    )
    return u_np, np.stack(snaps, axis=0)


@pytest.mark.parametrize("stride", [4, 6])
def test_snapshot_stride_matches_numpy_4d(stride):
    b3, env, bz_4d, opts, grid = _build_problem(n_steps=24)
    u_np, np_snaps = _numpy_snapshots(bz_4d, opts, grid, stride)

    u_jax, _v, jax_snaps = shear_fdtd_staggered_jax(
        None,
        None,
        bz_4d,
        opts={**opts, "jit": True},
        snapshot_stride=stride,
        **grid,
    )

    assert jax_snaps.shape == (grid["n_steps"] // stride, *u_np.shape)
    # Float32 reference magnitudes here are tiny; allow a few * |u| as tolerance.
    u_scale = float(np.max(np.abs(u_np))) + 1e-30
    assert np.max(np.abs(np_snaps - jax_snaps)) < 1e-3 * u_scale
    assert np.max(np.abs(u_np - u_jax)) < 1e-3 * u_scale


def test_snapshot_stride_matches_numpy_separable():
    b3, env, bz_4d, opts, grid = _build_problem(n_steps=24)
    stride = 6
    u_np, np_snaps = _numpy_snapshots(bz_4d, opts, grid, stride)

    u_jax, _v, jax_snaps = shear_fdtd_staggered_jax(
        None,
        None,
        b3,
        opts={**opts, "jit": True, "env_t": env},
        snapshot_stride=stride,
        **grid,
    )

    assert jax_snaps.shape == (grid["n_steps"] // stride, *u_np.shape)
    u_scale = float(np.max(np.abs(u_np))) + 1e-30
    assert np.max(np.abs(np_snaps - jax_snaps)) < 1e-3 * u_scale
    assert np.max(np.abs(u_np - u_jax)) < 1e-3 * u_scale


def test_snapshot_stride_separable_matches_4d_jax():
    """Cross-check: 4D and separable JAX paths should agree to machine precision."""
    b3, env, bz_4d, opts, grid = _build_problem(n_steps=24)
    stride = 6

    _u4, _v4, snaps_4d = shear_fdtd_staggered_jax(
        None,
        None,
        bz_4d,
        opts={**opts, "jit": True},
        snapshot_stride=stride,
        **grid,
    )
    _ue, _ve, snaps_sep = shear_fdtd_staggered_jax(
        None,
        None,
        b3,
        opts={**opts, "jit": True, "env_t": env},
        snapshot_stride=stride,
        **grid,
    )
    assert snaps_4d.shape == snaps_sep.shape
    scale = float(np.max(np.abs(snaps_4d))) + 1e-30
    assert np.max(np.abs(snaps_4d - snaps_sep)) < 1e-5 * scale


def test_snapshot_stride_returns_device_arrays_when_requested():
    b3, env, _bz4, opts, grid = _build_problem(n_steps=24)
    stride = 6

    u, v, snaps = shear_fdtd_staggered_jax(
        None,
        None,
        b3,
        opts={**opts, "jit": True, "env_t": env},
        snapshot_stride=stride,
        return_device=True,
        **grid,
    )
    # JAX arrays expose a `.device()` method or live on a JAX device.
    assert "jax" in type(u).__module__.lower()
    assert "jax" in type(v).__module__.lower()
    assert "jax" in type(snaps).__module__.lower()


def test_snapshot_stride_rejects_non_divisible():
    b3, env, _bz4, opts, grid = _build_problem(n_steps=24)
    with pytest.raises(ValueError, match=r"multiple of snapshot_stride"):
        shear_fdtd_staggered_jax(
            None,
            None,
            b3,
            opts={**opts, "jit": True, "env_t": env},
            snapshot_stride=5,
            **grid,
        )


def test_snapshot_stride_rejects_non_positive():
    b3, env, _bz4, opts, grid = _build_problem(n_steps=24)
    with pytest.raises(ValueError, match=r"positive integer"):
        shear_fdtd_staggered_jax(
            None,
            None,
            b3,
            opts={**opts, "jit": True, "env_t": env},
            snapshot_stride=0,
            **grid,
        )


def test_snapshot_stride_conflicts_with_return_traces():
    b3, env, _bz4, opts, grid = _build_problem(n_steps=24)
    with pytest.raises(ValueError, match=r"not both supported"):
        shear_fdtd_staggered_jax(
            None,
            None,
            b3,
            opts={**opts, "jit": True, "env_t": env},
            snapshot_stride=6,
            return_traces=True,
            trace_indices=([0], [0], [0]),
            **grid,
        )


def test_default_path_unchanged_without_snapshot_stride():
    """Sanity: omitting snapshot_stride must keep the 2-tuple return contract."""
    b3, env, _bz4, opts, grid = _build_problem(n_steps=12)
    out = shear_fdtd_staggered_jax(
        None,
        None,
        b3,
        opts={**opts, "jit": True, "env_t": env},
        **grid,
    )
    assert len(out) == 2
