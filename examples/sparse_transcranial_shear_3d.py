"""End-to-end sparse array transcranial shear wave demo.

Full pipeline:
  1. Build medium from microCT skull NRRD
  2. Sparse array focused acoustic transmit (FDTD on GPU)
  3. Compute acoustic radiation force (ARF)
  4. Shear wave FDTD for viscoelastic tissue displacement
  5. Generate publication-quality figures for physics derivation appendix

Figures produced (7 categories):
  (a) Array surface footprint
  (b) Acoustical medium property maps (sound speed, density, attenuation)
  (c) Pressure propagation snapshots through skull
  (d) Radiation force slices
  (e) Displacement magnitude snapshots at multiple shear wave times
  (f) 3D displacement rendering with isosurface
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fullwave
from fullwave.medium_builder import MediumBuilder
from fullwave.medium_builder.presets import BackgroundDomain, MicroCTSkullDomain
from shearwave import (
    compute_radiation_force,
    compute_radiation_force_velocity,
    shear_fdtd,
)
from fullwave.utils import plot_utils
from fullwave.utils.nrrd_reader import download_halle_skull
from fullwave.utils.slab_extraction import Placement

logger = logging.getLogger(__name__)

# Path to the sparse-array connector file. Override with --connector-path or
# the SHEARWAVE_SPARSE_MAT environment variable; defaults to the file name in
# the current working directory.
DEFAULT_SPARSE_MAT = os.environ.get("SHEARWAVE_SPARSE_MAT", "Sparse_TransConnector.mat")

# Optional directory to also copy figures into (e.g. a manuscript figures dir).
# Disabled unless SHEARWAVE_DOCS_FIGURES_DIR is set.
DOCS_FIGURES_DIR = (
    Path(os.environ["SHEARWAVE_DOCS_FIGURES_DIR"])
    if os.environ.get("SHEARWAVE_DOCS_FIGURES_DIR")
    else None
)


# ---------------------------------------------------------------------------
# Sparse connector helpers (reused from examples/wave_3d/sparse_transcranial_3d.py)
# ---------------------------------------------------------------------------


def load_sparse_connector(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load sparse connector element positions and IDs from .mat file."""
    with h5py.File(mat_path, "r") as f:
        positions = f["Position"][()].T  # (3, 1024) -> (1024, 3)
        element_ids = f["SparseConnector"][()].ravel().astype(int)
    return positions, element_ids


def build_sparse_element_map(
    positions: np.ndarray,
    element_ids: np.ndarray,
    element_size_px: int,
    dx: float,
    grid_ny: int,
    grid_nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map sparse element physical positions to grid coordinates."""
    n_elements = positions.shape[0]
    half_ptch = element_size_px // 2

    iy_raw = np.round(positions[:, 0] / dx).astype(int)
    iz_raw = np.round(positions[:, 1] / dx).astype(int)

    iy_centers = np.round(iy_raw - np.mean(iy_raw) + grid_ny / 2).astype(int)
    iz_centers = np.round(iz_raw - np.mean(iz_raw) + grid_nz / 2).astype(int)

    yz_element_ids = np.zeros((grid_ny, grid_nz), dtype=np.int32)
    for m in range(n_elements):
        for di in range(element_size_px):
            for dj in range(element_size_px):
                iy = iy_centers[m] + di - half_ptch
                iz = iz_centers[m] + dj - half_ptch
                if 0 <= iy < grid_ny and 0 <= iz < grid_nz:
                    yz_element_ids[iy, iz] = element_ids[m]

    yz_active = yz_element_ids > 0
    n_active = int(yz_active.sum())
    print(f"Sparse array: {n_elements} elements, {n_active} active pixels in y-z plane")
    pct = 100 * n_active / (grid_ny * grid_nz)
    print(f"  Element footprint: {element_size_px}x{element_size_px} pixels")
    print(f"  Grid coverage: {n_active}/{grid_ny * grid_nz} ({pct:.1f}%)")

    return yz_active, yz_element_ids, iy_centers, iz_centers


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def save_fig(fig, out_dir, name, *, docs_dir=None, dpi=200):
    """Save figure to output directory and optionally copy to docs/figures."""
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    if docs_dir is not None:
        docs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, docs_dir / name)


def plot_array_surface(
    yz_element_ids, iy_centers, iz_centers, element_ids, out_dir, *, docs_dir=None
):
    """Plot sparse array element footprint in the y-z transducer plane."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.scatter(
        iy_centers,
        iz_centers,
        c="k",
        s=8,
        alpha=0.8,
    )
    ax.set_xlabel("lateral (y) [grid]")
    ax.set_ylabel("elevational (z) [grid]")
    ax.set_title("(a) Sparse Array Surface Layout", fontsize=14)
    ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, out_dir, "fig_a_array_surface.png", docs_dir=docs_dir)


def plot_acoustical_maps(
    medium,
    grid,
    out_dir,
    *,
    docs_dir=None,
):
    """(b) Acoustical medium property maps (sound speed, density, attenuation)."""
    nz_mid = grid.nz // 2
    ny_mid = grid.ny // 2
    dx_mm = grid.dx * 1e3
    dy_mm = grid.dy * 1e3
    dz_mm = grid.dz * 1e3

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: x-y slices (depth-lateral at center elevation)
    for col, (field, label, units) in enumerate(
        [
            (medium.sound_speed[:, :, nz_mid], "Sound Speed", "m/s"),
            (medium.density[:, :, nz_mid], "Density", "kg/m³"),
            (medium.alpha_coeff[:, :, nz_mid], "Attenuation α₀", "dB/cm/MHz"),
        ]
    ):
        ax = axes[0, col]
        nx_f, ny_f = field.shape
        extent = [0, ny_f * dy_mm, nx_f * dx_mm, 0]
        im = ax.imshow(field, origin="upper", cmap="turbo", aspect="equal", extent=extent)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        ax.set_title(f"{label} — XY slice")
        fig.colorbar(im, ax=ax, label=units)

    # Row 1: x-z slices (depth-elevational at center lateral)
    for col, (field, label, units) in enumerate(
        [
            (medium.sound_speed[:, ny_mid, :], "Sound Speed", "m/s"),
            (medium.density[:, ny_mid, :], "Density", "kg/m³"),
            (medium.alpha_coeff[:, ny_mid, :], "Attenuation α₀", "dB/cm/MHz"),
        ]
    ):
        ax = axes[1, col]
        nx_f, nz_f = field.shape
        extent = [0, nz_f * dz_mm, nx_f * dx_mm, 0]
        im = ax.imshow(field, origin="upper", cmap="turbo", aspect="equal", extent=extent)
        ax.set_xlabel("Elevational (mm)")
        ax.set_ylabel("Depth (mm)")
        ax.set_title(f"{label} — XZ slice")
        fig.colorbar(im, ax=ax, label=units)

    fig.suptitle("(b) Acoustical Medium Properties (Skull)", fontsize=14)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_b_acoustical_maps.png", docs_dir=docs_dir)


def plot_pressure_snapshots(
    propagation_map,
    c_ds,
    nz_mid_ds,
    out_dir,
    *,
    dx_ds=None,
    dy_ds=None,
    dt_snapshot=None,
    docs_dir=None,
    n_snapshots=5,
):
    """(c) Pressure propagation snapshots through skull."""
    nt = propagation_map.shape[0]
    # Select time indices spaced across the propagation
    t_indices = np.linspace(nt // 8, nt - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 5))
    if n_snapshots == 1:
        axes = [axes]

    # Sign-preserving power compression: sign(p) * |p|^(1/3)
    power = 1.0 / 3.0
    all_compressed = []
    for t in t_indices:
        p = propagation_map[t, :, :, nz_mid_ds]
        all_compressed.append(np.sign(p) * np.abs(p) ** power)
    vmax = max(np.abs(f).max() for f in all_compressed)
    if vmax == 0:
        vmax = 1.0

    skull_mask = c_ds[:, :, nz_mid_ds] > 1600  # for overlay contour

    # Metric extent
    nx_p, ny_p = all_compressed[0].shape
    if dx_ds is not None and dy_ds is not None:
        dx_mm = dx_ds * 1e3
        dy_mm = dy_ds * 1e3
        extent = [0, ny_p * dy_mm, nx_p * dx_mm, 0]
        xlabel, ylabel = "Lateral (mm)", "Depth (mm)"
    else:
        extent = None
        xlabel, ylabel = "lateral (y)", "depth (x)"

    for i, (t_idx, compressed) in enumerate(zip(t_indices, all_compressed)):
        ax = axes[i]
        im = ax.imshow(
            compressed,
            origin="upper",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
        )
        # Skull contour overlay
        if skull_mask.any():
            if extent is not None:
                y_coords = np.linspace(extent[0], extent[1], ny_p)
                x_coords = np.linspace(extent[3], extent[2], nx_p)
                ax.contour(y_coords, x_coords, skull_mask, levels=[0.5], colors="k", linewidths=0.8)
            else:
                ax.contour(skull_mask, levels=[0.5], colors="k", linewidths=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if dt_snapshot is not None:
            t_us = t_idx * dt_snapshot * 1e6
            ax.set_title(f"t = {t_us:.0f} μs")
        else:
            ax.set_title(f"t = {t_idx}")
        ax.tick_params(labelsize=8)

    # Colorbar outside the panels
    fig.colorbar(
        im,
        ax=list(axes),
        shrink=0.8,
        pad=0.02,
        label="sgn(p) |p|$^{1/3}$ [Pa$^{1/3}$]",
    )

    fig.suptitle("(c) Pressure Propagation — XY Slices Through Skull", fontsize=14)
    save_fig(fig, out_dir, "fig_c_pressure_snapshots.png", docs_dir=docs_dir)


def plot_acoustical_maps_and_pressure(
    medium,
    grid,
    propagation_map,
    c_ds,
    nz_mid_ds,
    out_dir,
    *,
    mod_x=4,
    mod_y=4,
    sampling_modulus_time=7,
    t_indices=None,
    docs_dir=None,
):
    """Combined figure: acoustical maps (rows 0-1) + pressure snapshots (row 2).

    Parameters
    ----------
    t_indices : array-like of int, optional
        Time-step indices into propagation_map for the pressure snapshots.
        Defaults to 3 evenly spaced indices.
    """
    from matplotlib import gridspec

    nz_mid = grid.nz // 2
    ny_mid = grid.ny // 2
    dx_mm = grid.dx * 1e3
    dy_mm = grid.dy * 1e3
    dz_mm = grid.dz * 1e3

    # Downsampled grid spacing
    dx_ds_mm = dx_mm * mod_x
    dy_ds_mm = dy_mm * mod_y

    # Time per snapshot step
    dt_snapshot = grid.dt * sampling_modulus_time

    nt = propagation_map.shape[0]
    if t_indices is None:
        t_indices = np.linspace(nt // 8, nt - 1, 3, dtype=int)
    n_snap = len(t_indices)

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Rows 0-1: Acoustical maps ---
    map_fields = [
        [
            (medium.sound_speed[:, :, nz_mid], "Sound Speed", "m/s"),
            (medium.density[:, :, nz_mid], "Density", "kg/m³"),
            (medium.alpha_coeff[:, :, nz_mid], "Attenuation α₀", "dB/cm/MHz"),
        ],
        [
            (medium.sound_speed[:, ny_mid, :], "Sound Speed", "m/s"),
            (medium.density[:, ny_mid, :], "Density", "kg/m³"),
            (medium.alpha_coeff[:, ny_mid, :], "Attenuation α₀", "dB/cm/MHz"),
        ],
    ]
    slice_labels = [
        ("Lateral (mm)", "Depth (mm)", dy_mm, dx_mm, "XY"),
        ("Elevational (mm)", "Depth (mm)", dz_mm, dx_mm, "XZ"),
    ]
    for row in range(2):
        xlabel, ylabel, d_horiz, d_vert, plane = slice_labels[row]
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            field, label, units = map_fields[row][col]
            n_vert, n_horiz = field.shape
            extent = [0, n_horiz * d_horiz, n_vert * d_vert, 0]
            im = ax.imshow(field, origin="upper", cmap="turbo", aspect="equal", extent=extent)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label} — {plane} slice")
            fig.colorbar(im, ax=ax, label=units)

    # --- Row 2: Pressure snapshots ---
    # Sign-preserving power compression
    power = 1.0 / 3.0
    all_compressed = []
    for t in t_indices:
        p = propagation_map[t, :, :, nz_mid_ds]
        all_compressed.append(np.sign(p) * np.abs(p) ** power)
    vmax = max(np.abs(f).max() for f in all_compressed)
    if vmax == 0:
        vmax = 1.0

    skull_mask = c_ds[:, :, nz_mid_ds] > 1600
    nx_p, ny_p = all_compressed[0].shape
    p_extent = [0, ny_p * dy_ds_mm, nx_p * dx_ds_mm, 0]
    y_coords = np.linspace(p_extent[0], p_extent[1], ny_p)
    x_coords = np.linspace(p_extent[3], p_extent[2], nx_p)

    pressure_axes = []
    for i, (t_idx, compressed) in enumerate(zip(t_indices, all_compressed)):
        ax = fig.add_subplot(gs[2, i])
        pressure_axes.append(ax)
        im_p = ax.imshow(
            compressed,
            origin="upper",
            cmap="seismic",
            aspect="equal",
            vmin=-vmax,
            vmax=vmax,
            extent=p_extent,
        )
        if skull_mask.any():
            ax.contour(y_coords, x_coords, skull_mask, levels=[0.5], colors="k", linewidths=0.8)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        t_us = t_idx * dt_snapshot * 1e6
        ax.set_title(f"Pressure — t = {t_us:.0f} μs")

    fig.colorbar(
        im_p,
        ax=pressure_axes,
        shrink=0.8,
        pad=0.02,
        label="sgn(p) |p|$^{1/3}$ [Pa$^{1/3}$]",
    )

    save_fig(fig, out_dir, "fig_bc_acoustical_and_pressure.png", docs_dir=docs_dir)


def plot_radiation_force(
    b0,
    nz_mid_ds,
    ny_mid_ds,
    out_dir,
    *,
    docs_dir=None,
):
    """(d) Radiation force slices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    b_max = b0.max()
    if b_max == 0:
        b_max = 1.0

    # XY slice
    ax = axes[0]
    im = ax.imshow(
        b0[:, :, nz_mid_ds], origin="upper", cmap="hot", aspect="equal", vmin=0, vmax=b_max
    )
    ax.set_xlabel("lateral (y)")
    ax.set_ylabel("depth (x)")
    ax.set_title("ARF — XY slice")
    fig.colorbar(im, ax=ax, label="N/m³")

    # XZ slice
    ax = axes[1]
    im = ax.imshow(
        b0[:, ny_mid_ds, :], origin="upper", cmap="hot", aspect="equal", vmin=0, vmax=b_max
    )
    ax.set_xlabel("elevational (z)")
    ax.set_ylabel("depth (x)")
    ax.set_title("ARF — XZ slice")
    fig.colorbar(im, ax=ax, label="N/m³")

    fig.suptitle("(d) Acoustic Radiation Force", fontsize=14)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_d_radiation_force.png", docs_dir=docs_dir)


def plot_displacement_snapshots(
    snapshots_xy,
    snapshots_xz,
    c_ds,
    nz_mid_ds,
    ny_mid_ds,
    dt_shear,
    save_every,
    out_dir,
    *,
    docs_dir=None,
    n_snapshots=5,
):
    """(e) Displacement magnitude snapshots at multiple shear wave times."""
    n_frames = len(snapshots_xy)
    n_snapshots = min(n_snapshots, n_frames)
    frame_indices = np.linspace(n_frames // 4, n_frames - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, n_snapshots, figsize=(4 * n_snapshots, 8))
    if n_snapshots == 1:
        axes = axes.reshape(2, 1)

    # Compute |u| for all recorded snapshots at selected frames
    vmax = 0.0
    for idx in frame_indices:
        vmax = max(vmax, np.abs(snapshots_xy[idx]).max())
    if vmax == 0:
        vmax = 1.0

    skull_xy = c_ds[:, :, nz_mid_ds] > 1600
    skull_xz = c_ds[:, ny_mid_ds, :] > 1600

    for i, idx in enumerate(frame_indices):
        t_ms = idx * save_every * dt_shear * 1e3

        # XY slice
        ax = axes[0, i]
        im = ax.imshow(
            np.abs(snapshots_xy[idx]),
            origin="upper",
            cmap="hot",
            aspect="equal",
            vmin=0,
            vmax=vmax,
        )
        if skull_xy.any():
            ax.contour(skull_xy, levels=[0.5], colors="w", linewidths=0.5)
        ax.set_title(f"t = {t_ms:.1f} ms")
        ax.set_xlabel("lateral (y)")
        ax.set_ylabel("depth (x)")
        ax.tick_params(labelsize=8)

        # XZ slice
        ax = axes[1, i]
        im = ax.imshow(
            np.abs(snapshots_xz[idx]),
            origin="upper",
            cmap="hot",
            aspect="equal",
            vmin=0,
            vmax=vmax,
        )
        if skull_xz.any():
            ax.contour(skull_xz, levels=[0.5], colors="w", linewidths=0.5)
        ax.set_title(f"t = {t_ms:.1f} ms")
        ax.set_xlabel("elevational (z)")
        ax.set_ylabel("depth (x)")
        ax.tick_params(labelsize=8)

    # Shared colorbar for displacement magnitude
    fig.colorbar(im, ax=axes.ravel().tolist(), label="|u| [m]", shrink=0.6, pad=0.02)

    fig.suptitle("(e) Shear Wave |u| Snapshots — XY (top) / XZ (bottom)", fontsize=14)
    save_fig(fig, out_dir, "fig_e_displacement_snapshots.png", docs_dir=docs_dir)


def plot_displacement_3d(
    u,
    dx2,
    dy2,
    dz2,
    out_dir,
    *,
    c_ds=None,
    docs_dir=None,
    iso_fraction=0.5,
):
    """(f) 3D displacement rendering with translucent isosurface."""
    try:
        from skimage import measure
    except ImportError:
        print("  Skipping 3D isosurface (scikit-image not available)")
        _plot_displacement_3d_scatter(
            u,
            dx2,
            dy2,
            dz2,
            out_dir,
            c_ds=c_ds,
            docs_dir=docs_dir,
        )
        return

    u_mag = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2 + u[..., 2] ** 2)
    level = iso_fraction * u_mag.max()
    if level <= 0:
        print("  Skipping 3D isosurface (zero displacement)")
        return

    try:
        verts, faces, _, _ = measure.marching_cubes(
            u_mag,
            level=level,
            spacing=(dx2, dy2, dz2),
        )
    except (RuntimeError, ValueError):
        print("  Skipping 3D isosurface (marching_cubes failed)")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    nx, ny, nz = u_mag.shape

    # Skull surface in background
    if c_ds is not None:
        skull_vol = c_ds > 1600
        if skull_vol.any():
            try:
                skull_verts, skull_faces, _, _ = measure.marching_cubes(
                    skull_vol.astype(float),
                    level=0.5,
                    spacing=(dx2, dy2, dz2),
                )
                skull_mm = skull_verts * 1e3
                skull_mesh = Poly3DCollection(
                    skull_mm[skull_faces],
                    alpha=0.08,
                    edgecolor="none",
                    facecolor="slategray",
                )
                ax.add_collection3d(skull_mesh)
            except (RuntimeError, ValueError):
                pass  # skip skull if marching_cubes fails

    # Displacement isosurface
    verts_mm = verts * 1e3
    mesh_mm = Poly3DCollection(
        verts_mm[faces],
        alpha=0.3,
        edgecolor="none",
        facecolor="coral",
    )
    ax.add_collection3d(mesh_mm)

    ax.set_xlim(0, nx * dx2 * 1e3)
    ax.set_ylim(0, ny * dy2 * 1e3)
    ax.set_zlim(0, nz * dz2 * 1e3)
    ax.set_xlabel("x (depth) [mm]")
    ax.set_ylabel("y (lateral) [mm]")
    ax.set_zlabel("z (elevational) [mm]")
    ax.set_title(f"(f) 3D Displacement |u| = {level * 1e6:.1f} µm")

    save_fig(fig, out_dir, "fig_f_displacement_3d.png", docs_dir=docs_dir, dpi=150)


def _plot_displacement_3d_scatter(u, dx2, dy2, dz2, out_dir, *, c_ds=None, docs_dir=None):
    """Fallback 3D scatter when scikit-image is unavailable."""
    u_mag = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2 + u[..., 2] ** 2)
    threshold = 0.5 * u_mag.max()
    if threshold <= 0:
        return

    ix, iy, iz = np.where(u_mag > threshold)
    # Subsample for performance
    max_pts = 5000
    if len(ix) > max_pts:
        sel = np.random.default_rng(42).choice(len(ix), max_pts, replace=False)
        ix, iy, iz = ix[sel], iy[sel], iz[sel]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Skull scatter in background
    if c_ds is not None:
        skull_ix, skull_iy, skull_iz = np.where(c_ds > 1600)
        if len(skull_ix) > 0:
            max_skull = 3000
            if len(skull_ix) > max_skull:
                sel_s = np.random.default_rng(0).choice(len(skull_ix), max_skull, replace=False)
                skull_ix, skull_iy, skull_iz = skull_ix[sel_s], skull_iy[sel_s], skull_iz[sel_s]
            ax.scatter(
                skull_ix * dx2 * 1e3,
                skull_iy * dy2 * 1e3,
                skull_iz * dz2 * 1e3,
                c="slategray",
                s=1,
                alpha=0.05,
            )

    vals = u_mag[ix, iy, iz]
    sc = ax.scatter(
        ix * dx2 * 1e3,
        iy * dy2 * 1e3,
        iz * dz2 * 1e3,
        c=vals,
        cmap="hot",
        s=2,
        alpha=0.5,
    )
    ax.set_xlabel("x (depth) [mm]")
    ax.set_ylabel("y (lateral) [mm]")
    ax.set_zlabel("z (elevational) [mm]")
    ax.set_title("(f) 3D Displacement Magnitude")
    fig.colorbar(sc, ax=ax, label="|u| [m]", shrink=0.6)

    save_fig(fig, out_dir, "fig_f_displacement_3d.png", docs_dir=docs_dir, dpi=150)


def plot_displacement_time_trace(
    disp_trace,
    dt_shear,
    env_t,
    out_dir,
    *,
    docs_dir=None,
):
    """(g) Displacement time trace at peak ARF location with force envelope."""
    n = len(disp_trace)
    t_ms = np.arange(n) * dt_shear * 1e3  # time in ms

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Force envelope on secondary axis
    ax2 = ax1.twinx()
    env_plot = env_t[:n] if len(env_t) >= n else env_t
    ax2.fill_between(
        np.arange(len(env_plot)) * dt_shear * 1e3,
        env_plot,
        alpha=0.15,
        color="gray",
        label="Force envelope",
    )
    ax2.set_ylabel("Force envelope", color="gray")
    ax2.set_ylim(-0.05, 1.5)
    ax2.tick_params(axis="y", labelcolor="gray")

    # Displacement trace
    ax1.plot(t_ms, disp_trace * 1e6, "b-", linewidth=1.0, label="$u_x$ at peak")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Displacement $u_x$ [µm]")
    ax1.set_title("(g) Displacement at Peak ARF Location")
    ax1.grid(True, alpha=0.3)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig_g_displacement_trace.png", docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(
    nrrd_path=None,
    sparse_mat_path=None,
    out_dir=None,
    *,
    download=False,
    skip_acoustic=False,
    skip_shear=False,
    use_velocity=False,
    theta_burst=False,
):
    """Run end-to-end sparse transcranial shear wave demo.

    Parameters
    ----------
    nrrd_path : str, optional
        Path to the microCT NRRD skull file.
    sparse_mat_path : str, optional
        Path to Sparse_TransConnector.mat.
    out_dir : str or Path, optional
        Output directory. Defaults to ./outputs/sparse_transcranial_shear.
    download : bool
        If True, download the Halle skull from Zenodo if not already cached.
    skip_acoustic : bool
        If True, load saved acoustic results instead of running GPU simulation.
    skip_shear : bool
        If True, load saved shear results instead of running CPU simulation.
    use_velocity : bool
        If True, use Poynting vector (velocity-based) ARF instead of
        plane-wave pressure-based ARF.  Requires ``output_velocity=True``
        on the acoustic solver.
    theta_burst : bool
        If True, use theta-burst TUS pulse envelope: 360 µs tone-bursts
        at 1 kHz PRF, grouped in 3-pulse bursts at 5 Hz theta rhythm
        (200 ms period).  The ARF spatial pattern is used as-is from the
        acoustic simulation.

    """
    logging.basicConfig(level=logging.INFO)

    if sparse_mat_path is None:
        sparse_mat_path = DEFAULT_SPARSE_MAT
    out_dir = Path("./outputs/sparse_transcranial_shear") if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = DOCS_FIGURES_DIR

    # ================================================================
    # Part 1: Acoustic simulation
    # ================================================================

    #
    # --- define the computational grid ---
    #
    f0 = 1e6
    c0 = 1540
    focal_depth = 50e-3  # 50 mm focus
    element_size_m = 750e-6  # element pitch (meters)
    element_thickness_px = 3  # source layers in depth
    domain_size = (65e-3, 66e-3, 66e-3)  # depth x lateral x elevational
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=10)
    grid.print_info()

    element_size_px = round(element_size_m / grid.dx)
    print(f"Element size: {element_size_m * 1e6:.0f} um -> {element_size_px} pixels")

    #
    # --- load sparse transducer layout ---
    #
    print(f"\nLoading sparse connector from: {sparse_mat_path}")
    positions, element_ids = load_sparse_connector(sparse_mat_path)
    print(
        f"  {len(element_ids)} elements, position range: "
        f"x=[{positions[:, 0].min() * 1e3:.1f}, {positions[:, 0].max() * 1e3:.1f}] mm, "
        f"y=[{positions[:, 1].min() * 1e3:.1f}, {positions[:, 1].max() * 1e3:.1f}] mm",
    )

    yz_active, yz_element_ids, iy_centers, iz_centers = build_sparse_element_map(
        positions,
        element_ids,
        element_size_px,
        grid.dx,
        grid.ny,
        grid.nz,
    )

    #
    # --- Figure (a): Array surface footprint ---
    #
    print("\n--- Figure (a): Array surface ---")
    plot_array_surface(
        yz_element_ids,
        iy_centers,
        iz_centers,
        element_ids,
        out_dir,
        docs_dir=docs_dir,
    )

    # Sensor parameters (needed for both acoustic run and reload)
    mod_x = 4
    mod_y = 4
    mod_z = 4
    sampling_modulus_time = 7

    if not skip_acoustic:
        #
        # --- define the acoustic medium from microCT ---
        #
        if nrrd_path is None:
            if download:
                nrrd_path = str(download_halle_skull())
            else:
                nrrd_path = "skull_microCT_zenodo/halle_skull.nrrd"

        placement = Placement(
            xdc_center_lps=(140.6, 118.5, 85.0),
            normal=(-0.954, -0.298),
            tangent=(-0.298, 0.954),
            z_mm=85.0,
        )

        skull_domain = MicroCTSkullDomain(
            grid=grid,
            nrrd_path=nrrd_path,
            placement=placement,
            lower_threshold=700,
            upper_threshold=1973,
            speed_range=(1540, 2900),
            density_range=(1000, 2200),
            blur_sigma_m=0e-4,
        )

        background_domain = BackgroundDomain(grid=grid, background_property_name="tissue")

        mb = MediumBuilder(grid=grid)
        mb.register_domain(background_domain)
        mb.register_domain(skull_domain)
        medium = mb.run()

        #
        # --- Figure (b): Acoustical medium maps ---
        #
        print("\n--- Figure (b): Acoustical maps ---")
        plot_acoustical_maps(medium, grid, out_dir, docs_dir=docs_dir)

        # Convert to exponential attenuation
        medium_exp = medium.build_exponential()
        medium_exp.print_info()

        #
        # --- build sparse source mask ---
        #
        p_mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
        for layer in range(element_thickness_px):
            p_mask[layer][yz_active] = True

        n_active_yz = int(yz_active.sum())
        n_total_sources = p_mask.sum()
        print(
            f"\nSource: {n_active_yz} active pixels/layer x {element_thickness_px} layers "
            f"= {n_total_sources} total source points",
        )

        #
        # --- compute per-element focusing delays ---
        #
        focal_x_px = focal_depth / grid.dx
        focal_y_px = grid.ny / 2.0
        focal_z_px = grid.nz / 2.0

        dist_per_element = np.sqrt(
            focal_x_px**2 + (iy_centers - focal_y_px) ** 2 + (iz_centers - focal_z_px) ** 2,
        )
        delays_per_element = dist_per_element * grid.dx / c0
        delays_per_element = delays_per_element.max() - delays_per_element

        element_delays = np.zeros(element_ids.max() + 1)
        for m in range(len(element_ids)):
            element_delays[element_ids[m]] = delays_per_element[m]

        active_iy, active_iz = np.where(yz_active)
        pixel_delays = element_delays[yz_element_ids[active_iy, active_iz]]

        print(
            f"Focusing delays: min={pixel_delays.min() * 1e6:.1f} us, "
            f"max={pixel_delays.max() * 1e6:.1f} us, "
            f"spread={(pixel_delays.max() - pixel_delays.min()) * 1e6:.1f} us",
        )

        #
        # --- generate focused source signals ---
        #
        ncycles = 2
        drop_off = 2
        source_p0 = 1.5e6  # 1.5 MPa — typical clinical ARFI pressure

        worst_pulse = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,
            f0=f0,
            duration=duration,
            ncycles=ncycles,
            drop_off=drop_off,
            p0=source_p0,
            i_layer=element_thickness_px - 1,
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
            delay_sec=pixel_delays.max(),
        )
        threshold = np.abs(worst_pulse).max() * 1e-6
        active_idx = np.nonzero(np.abs(worst_pulse) > threshold)[0]
        n_tic = active_idx[-1] + 1 if len(active_idx) > 0 else grid.nt
        print(f"Source nTic={n_tic} (of nT={grid.nt})")

        # CW correction: a Gaussian-windowed pulse has lower mean(p²) than CW
        # at the same peak pressure.  For CW, <p²> = p_peak²/2.  For the short
        # pulse, <p²> = mean(p_src²).  The ratio scales ARF / heat source up to
        # the equivalent CW value.
        p_src = worst_pulse[:n_tic]
        cw_correction = 0.5 * np.max(p_src**2) / np.mean(p_src**2)
        print(f"CW correction factor: {cw_correction:.3f}")

        dt_val = duration / grid.nt
        omega0 = 2.0 * np.pi * f0
        coeff = 1.05 / (ncycles * np.pi)
        t_base = np.arange(n_tic, dtype=np.float64) * dt_val
        p0 = np.zeros((n_total_sources, n_tic), dtype=np.float32)

        chunk_size = 50000
        for i_layer in range(element_thickness_px):
            layer_delay = (grid.dt / grid.cfl) * i_layer
            base_idx = n_active_yz * i_layer

            for ch_start in range(0, n_active_yz, chunk_size):
                ch_end = min(ch_start + chunk_size, n_active_yz)
                t_offsets = ncycles / f0 + pixel_delays[ch_start:ch_end] + layer_delay
                t_shifted = t_base[np.newaxis, :] - t_offsets[:, np.newaxis]
                w_t = t_shifted * omega0
                a_sq = (coeff * w_t) ** 2
                env = np.exp(-a_sq * a_sq)  # drop_off=2
                p0[base_idx + ch_start : base_idx + ch_end, :] = (
                    env * np.sin(w_t) * source_p0
                ).astype(np.float32)

        source = fullwave.Source(p0, p_mask)

        #
        # --- define sensor and run ---
        #
        sensor = fullwave.Sensor(
            mod_x=mod_x,
            mod_y=mod_y,
            mod_z=mod_z,
            sampling_modulus_time=sampling_modulus_time,
        )

        # When velocity output is requested, use the private binary that
        # supports OUTPUT_VELOCITY (the published release binaries do not).
        vel_bin = None
        if use_velocity:
            vel_bin = (
                Path(fullwave.__file__).parent
                / "solver"
                / "private_bins"
                / "fullwave2_3d_exponential_attenuation_multi_gpu"
            )
            if not vel_bin.exists():
                raise FileNotFoundError(
                    f"Velocity-capable binary not found at {vel_bin}. "
                    "Ensure the private_bins directory is present."
                )
            print(f"Using velocity-capable binary: {vel_bin}")

        fw_solver = fullwave.Solver(
            work_dir=out_dir,
            grid=grid,
            medium=medium_exp,
            source=source,
            sensor=sensor,
            run_on_memory=False,
            use_exponential_attenuation=True,
            output_velocity=use_velocity,
            path_fullwave_simulation_bin=vel_bin,
        )
        fw_solver.print_info()
        sensor_output = fw_solver.run()

        #
        # --- reshape and save acoustic output ---
        #
        nx_ds = int(np.ceil(grid.nx / mod_x))
        ny_ds = int(np.ceil(grid.ny / mod_y))
        nz_ds = int(np.ceil(grid.nz / mod_z))

        if use_velocity:
            # sensor_output is a dict with pressure + velocity arrays
            pressure_raw = sensor_output["pressure"]
            nt_recorded = pressure_raw.shape[1]
            propagation_map = pressure_raw.T.reshape(nt_recorded, nx_ds, ny_ds, nz_ds)

            vel_u_map = sensor_output["velocity_u"].T.reshape(nt_recorded, nx_ds, ny_ds, nz_ds)
            vel_w_map = sensor_output["velocity_w"].T.reshape(nt_recorded, nx_ds, ny_ds, nz_ds)
            vel_v_map = sensor_output["velocity_v"].T.reshape(nt_recorded, nx_ds, ny_ds, nz_ds)
        else:
            nt_recorded = sensor_output.shape[1]
            propagation_map = sensor_output.T.reshape(nt_recorded, nx_ds, ny_ds, nz_ds)

        c_ds = medium_exp.sound_speed[::mod_x, ::mod_y, ::mod_z]
        rho_ds = medium_exp.density[::mod_x, ::mod_y, ::mod_z]
        alpha_ds = medium.alpha_coeff[::mod_x, ::mod_y, ::mod_z]

        # Save for skip_acoustic reloading
        np.save(out_dir / "propagation_map.npy", propagation_map)
        np.save(out_dir / "c_ds.npy", c_ds)
        np.save(out_dir / "rho_ds.npy", rho_ds)
        np.save(out_dir / "alpha_ds.npy", alpha_ds)
        if use_velocity:
            np.save(out_dir / "velocity_u.npy", vel_u_map)
            np.save(out_dir / "velocity_w.npy", vel_w_map)
            np.save(out_dir / "velocity_v.npy", vel_v_map)
        np.savez(
            out_dir / "grid_params.npz",
            dx=grid.dx,
            dy=grid.dy,
            dz=grid.dz,
            nx=grid.nx,
            ny=grid.ny,
            nz=grid.nz,
            f0=f0,
            mod_x=mod_x,
            mod_y=mod_y,
            mod_z=mod_z,
            n_tic=n_tic,
            sampling_modulus_time=sampling_modulus_time,
            use_velocity=use_velocity,
            cw_correction=cw_correction,
        )
        print(f"Saved acoustic outputs to {out_dir}")

    else:
        # --- Load saved acoustic results ---
        medium = None  # not available when skipping acoustic
        print("\n--- Loading saved acoustic results (--skip-acoustic) ---")
        propagation_map = np.load(out_dir / "propagation_map.npy")
        c_ds = np.load(out_dir / "c_ds.npy")
        rho_ds = np.load(out_dir / "rho_ds.npy")
        alpha_ds = np.load(out_dir / "alpha_ds.npy")
        gp = np.load(out_dir / "grid_params.npz")
        n_tic = int(gp["n_tic"])
        cw_correction = float(gp["cw_correction"]) if "cw_correction" in gp else 1.0

        # Check if saved data used velocity (overrides CLI flag)
        saved_use_velocity = bool(gp["use_velocity"]) if "use_velocity" in gp else False
        if use_velocity and not saved_use_velocity:
            print("  WARNING: --use-velocity requested but saved data has no velocity files.")
            print("  Re-run without --skip-acoustic to generate velocity data.")
            use_velocity = False
        elif saved_use_velocity:
            use_velocity = True

        if use_velocity:
            vel_u_map = np.load(out_dir / "velocity_u.npy")
            vel_w_map = np.load(out_dir / "velocity_w.npy")
            vel_v_map = np.load(out_dir / "velocity_v.npy")
            print(f"  Loaded velocity arrays: {vel_u_map.shape}")

        nt_recorded = propagation_map.shape[0]
        nx_ds = propagation_map.shape[1]
        ny_ds = propagation_map.shape[2]
        nz_ds = propagation_map.shape[3]
        print(f"  propagation_map: {propagation_map.shape}")

    nz_mid_ds = nz_ds // 2
    ny_mid_ds = ny_ds // 2

    #
    # --- Figure (c): Pressure propagation snapshots ---
    #
    print("\n--- Figure (c): Pressure propagation snapshots ---")
    dx_ds = grid.dx * mod_x
    dy_ds = grid.dy * mod_y
    dt_snapshot = grid.dt * sampling_modulus_time
    plot_pressure_snapshots(
        propagation_map,
        c_ds,
        nz_mid_ds,
        out_dir,
        dx_ds=dx_ds,
        dy_ds=dy_ds,
        dt_snapshot=dt_snapshot,
        docs_dir=docs_dir,
    )

    # --- Combined figure (b+c): Acoustical maps + pressure snapshots ---
    if medium is not None:
        print("\n--- Combined figure (b+c): Acoustical maps + pressure ---")
        nt = propagation_map.shape[0]
        # Pick 3 snapshots: early (~t_idx 100), near-focal, and late (~t_idx 400)
        t_early = min(100, nt - 1)
        t_focal = nt // 2
        t_late = min(400, nt - 1)
        plot_acoustical_maps_and_pressure(
            medium,
            grid,
            propagation_map,
            c_ds,
            nz_mid_ds,
            out_dir,
            mod_x=mod_x,
            mod_y=mod_y,
            sampling_modulus_time=sampling_modulus_time,
            t_indices=np.array([t_early, t_focal, t_late]),
            docs_dir=docs_dir,
        )

    # Also generate wave propagation videos
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map[:, :, :, nz_mid_ds],
        c_map=c_ds[:, :, nz_mid_ds],
        rho_map=rho_ds[:, :, nz_mid_ds],
        export_name=out_dir / "wave_propagation_x-y.mp4",
        vmax=1.5e6,
        vmin=-1.5e6,
        figsize=(6, 6),
    )

    # ================================================================
    # Part 2: Radiation force
    # ================================================================

    #
    # --- compute radiation force ---
    #
    pressure_4d = propagation_map.transpose(1, 2, 3, 0)  # (nx, ny, nz, nt)

    # Pulse-length normalization: two corrections map from simulated short-pulse
    # mean(p²) to equivalent CW intensity:
    #   1. silence_correction — the pulse occupies only part of the recording
    #   2. cw_correction — the Gaussian envelope reduces mean(p²) vs CW p_peak²/2
    n_source_steps = n_tic // sampling_modulus_time
    silence_correction = nt_recorded / n_source_steps
    pulse_norm = silence_correction * cw_correction
    print(
        f"\nPulse normalization: silence={silence_correction:.2f}x, "
        f"cw={cw_correction:.2f}x, total={pulse_norm:.2f}x",
    )

    skull_mask = c_ds > 1600

    # Brain mask: for each (y,z) column, mask everything at or above the
    # deepest skull voxel (removes coupling medium and skull interior).
    brain_mask = np.ones_like(skull_mask, dtype=bool)
    for j in range(c_ds.shape[1]):
        for k in range(c_ds.shape[2]):
            col = skull_mask[:, j, k]
            skull_indices = np.where(col)[0]
            if len(skull_indices) > 0:
                brain_mask[: skull_indices.max() + 1, j, k] = False
    non_brain_mask = ~brain_mask

    if use_velocity:
        print("Computing vector acoustic radiation force (Poynting vector)...")
        vel_u_4d = vel_u_map.transpose(1, 2, 3, 0)  # (nx, ny, nz, nt)
        vel_w_4d = vel_w_map.transpose(1, 2, 3, 0)
        vel_v_4d = vel_v_map.transpose(1, 2, 3, 0)
        bx, by, bz = compute_radiation_force_velocity(
            pressure_4d,
            vel_u_4d,
            vel_w_4d,
            vel_v_4d,
            c_ds,
            rho_ds,
            alpha_ds,
            f0,
            gate=slice(None),
        )
        bx *= pulse_norm
        by *= pulse_norm
        bz *= pulse_norm
        bx[non_brain_mask] = 0.0
        by[non_brain_mask] = 0.0
        bz[non_brain_mask] = 0.0
        b0 = bx  # for plotting / peak detection (depth component)
        print(f"  Max |bx|: {np.abs(bx).max():.3e} N/m³")
        print(f"  Max |by|: {np.abs(by).max():.3e} N/m³")
        print(f"  Max |bz|: {np.abs(bz).max():.3e} N/m³")
    else:
        print("Computing acoustic radiation force (plane-wave approx.)...")
        b0 = compute_radiation_force(
            pressure_4d,
            c_ds,
            rho_ds,
            alpha_ds,
            f0,
            gate=slice(None),
        )
        b0 *= pulse_norm
        print(f"  Max body force: {b0.max():.3e} N/m³")
        b0[non_brain_mask] = 0.0
        print(f"  Max body force (after brain masking): {b0.max():.3e} N/m³")
        bx, by, bz = b0, None, None

    #
    # --- Figure (d): Radiation force ---
    #
    print("\n--- Figure (d): Radiation force ---")
    plot_radiation_force(b0, nz_mid_ds, ny_mid_ds, out_dir, docs_dir=docs_dir)

    # ================================================================
    # Part 3: Shear wave simulation
    # ================================================================

    #
    # --- shear wave material parameters ---
    #
    # Brain tissue parameters from Espindola et al. (2017) and
    # Chandrasekaran, Tripathi, Espindola & Pinton (2021):
    #   cs = 2.14 m/s at 75 Hz in porcine brain, rho = 1000 kg/m^3
    #   => mu = rho * cs^2 = 4580 Pa, E = 3*mu = 13740 Pa
    e_young = 13.74e3  # Young's modulus [Pa]
    nu_poisson = 0.499999  # Poisson's ratio (nearly incompressible)
    mu = e_young / (2.0 * (1.0 + nu_poisson))
    rho0 = float(np.median(rho_ds))
    cs = math.sqrt(mu / rho0)
    eta_pa_s = 0.5  # Kelvin-Voigt viscosity [Pa·s]

    print(f"\nShear modulus: {mu:.1f} Pa")
    print(f"Shear wave speed: {cs:.3f} m/s")
    print(f"Density: {rho0:.1f} kg/m³")

    #
    # --- shear wave grid and time stepping ---
    #
    dx2 = grid.dx * mod_x
    dy2 = grid.dy * mod_y
    dz2 = grid.dz * mod_z
    print(
        f"Shear grid spacing: dx={dx2 * 1e3:.3f} mm, dy={dy2 * 1e3:.3f} mm, dz={dz2 * 1e3:.3f} mm",
    )

    # Elastic CFL
    cfl = 0.1
    dt_elastic = cfl * min(dx2, dy2, dz2) / (cs * math.sqrt(3))

    # Viscous stability: dt < rho * dx^2 / (2 * d * eta)
    dx_min = min(dx2, dy2, dz2)
    dt_viscous = rho0 * dx_min**2 / (2.0 * 3 * eta_pa_s)
    dt_shear = min(dt_elastic, 0.5 * dt_viscous)

    print(f"Elastic dt limit: {dt_elastic:.3e} s")
    print(f"Viscous dt limit: {dt_viscous:.3e} s")
    print(f"Shear time step: {dt_shear:.6e} s")

    if theta_burst:
        # Theta-burst TUS: 360 µs tone-bursts at 1 kHz PRF, grouped
        # in 3-pulse bursts at 5 Hz theta rhythm (200 ms period).
        t_burst = 360e-6  # tone-burst duration [s]
        prf = 1e3  # pulse repetition frequency [Hz]
        t_burst_period = 1.0 / prf  # 1 ms
        bursts_per_group = 3
        t_group_on = bursts_per_group * t_burst_period  # 3 ms
        theta_freq = 5.0  # Hz
        t_theta_period = 1.0 / theta_freq  # 200 ms
        n_theta_cycles = 1  # simulate one theta cycle
        t_max = n_theta_cycles * t_theta_period
        n_steps = math.ceil(t_max / dt_shear)

        # Build envelope: within each theta cycle, the first t_group_on
        # contains the burst train (360 µs on / 640 µs off at 1 kHz PRF).
        t = np.arange(n_steps, dtype=np.float64) * dt_shear
        t_in_theta = t % t_theta_period
        in_group = t_in_theta < t_group_on
        t_in_burst = t_in_theta % t_burst_period
        in_burst = t_in_burst < t_burst
        env_t = np.where(in_group & in_burst, 1.0, 0.0).astype(np.float32)
        n_on = int(env_t.sum())
        print(
            f"Theta-burst TUS: {t_burst * 1e6:.0f} µs bursts @ {prf / 1e3:.0f} kHz PRF, "
            f"{bursts_per_group} per group @ {theta_freq:.0f} Hz theta"
        )
    else:
        # Single push: 2 ms on, 8 ms off, no repeats
        t_pulse_on = 2.0e-3  # pulse ON duration
        t_pulse_off = 8.0e-3  # pulse OFF duration
        t_period = t_pulse_on + t_pulse_off  # 10 ms period
        n_pulses = 1
        t_max = n_pulses * t_period  # 10 ms total
        n_steps = math.ceil(t_max / dt_shear)

        t = np.arange(n_steps, dtype=np.float64) * dt_shear
        t_in_period = t % t_period
        env_t = np.where(t_in_period <= t_pulse_on, 1.0, 0.0).astype(np.float32)
        n_on = int(env_t.sum())
        print(
            f"Pulsed protocol: {t_pulse_on * 1e3:.1f} ms on / {t_pulse_off * 1e3:.1f} ms off "
            f"x {n_pulses} pulses ({n_on} ON steps / {n_steps} total, "
            f"DC={100 * n_on / n_steps:.0f}%)",
        )

    print(f"Number of steps: {n_steps}")
    print(f"Total time: {t_max * 1e3:.1f} ms")
    print(f"ON steps: {n_on}/{n_steps} (DC={100 * n_on / n_steps:.1f}%)")

    # Peak ARF location — record displacement time trace here
    peak_idx = np.unravel_index(b0.argmax(), b0.shape)
    print(f"Peak ARF voxel: {peak_idx}")

    if not skip_shear:
        #
        # --- run shear FDTD with snapshot recording ---
        #
        n_frames = 100
        save_every = max(1, n_steps // n_frames)
        snapshots_xy = []
        snapshots_xz = []
        disp_trace = []  # displacement at peak ARF location every step

        def record_snapshot(it, u_center):
            # Record time trace at peak location (every step)
            disp_trace.append(u_center[peak_idx[0], peak_idx[1], peak_idx[2], 0])
            if it % save_every == 0:
                snapshots_xy.append(u_center[:, :, nz_mid_ds, 0].copy())
                snapshots_xz.append(u_center[:, ny_mid_ds, :, 0].copy())

        print("\nRunning shear wave FDTD solver...")
        u, v = shear_fdtd(
            bx=bx,
            by=by,
            bz=bz,
            rho=rho0,
            mu=mu,
            dX=dx2,
            dY=dy2,
            dZ=dz2,
            dT=dt_shear,
            n_steps=n_steps,
            opts={
                "vel_damp": 0.02,
                "eta": eta_pa_s,
                "env_t": env_t,
                "verbose": True,
                "progress_every": max(1, n_steps // 10),
            },
            callback=record_snapshot,
        )
        disp_trace = np.array(disp_trace, dtype=np.float64)
        u_x_max = np.abs(u[..., 0]).max()
        print(f"Max |u_x|: {u_x_max:.3e} m ({u_x_max * 1e6:.2f} µm)")
        print(f"Recorded {len(snapshots_xy)} shear wave frames")

        # Save shear results
        np.save(out_dir / "displacement.npy", u)
        np.save(out_dir / "velocity.npy", v)
        np.save(out_dir / "radiation_force.npy", b0)
        np.save(out_dir / "snapshots_xy.npy", np.stack(snapshots_xy))
        np.save(out_dir / "snapshots_xz.npy", np.stack(snapshots_xz))
        np.save(out_dir / "disp_trace.npy", disp_trace)
        np.savez(
            out_dir / "shear_params.npz",
            dt_shear=dt_shear,
            save_every=save_every,
            n_steps=n_steps,
            mu=mu,
            rho0=rho0,
            cs=cs,
            eta=eta_pa_s,
            peak_idx=np.array(peak_idx),
        )
        print(f"Saved shear results to {out_dir}")

    else:
        # --- Load saved shear results ---
        print("\n--- Loading saved shear results (--skip-shear) ---")
        u = np.load(out_dir / "displacement.npy")
        snapshots_xy = list(np.load(out_dir / "snapshots_xy.npy"))
        snapshots_xz = list(np.load(out_dir / "snapshots_xz.npy"))
        disp_trace = np.load(out_dir / "disp_trace.npy")
        sp = np.load(out_dir / "shear_params.npz")
        save_every = int(sp["save_every"])
        print(f"  displacement shape: {u.shape}")
        print(f"  {len(snapshots_xy)} snapshot frames loaded")

    # ================================================================
    # Part 4: Shear wave visualization
    # ================================================================

    #
    # --- Figure (e): Displacement magnitude snapshots ---
    #
    print("\n--- Figure (e): Displacement magnitude snapshots ---")
    plot_displacement_snapshots(
        snapshots_xy,
        snapshots_xz,
        c_ds,
        nz_mid_ds,
        ny_mid_ds,
        dt_shear,
        save_every,
        out_dir,
        docs_dir=docs_dir,
    )

    #
    # --- Figure (f): 3D displacement rendering ---
    #
    print("\n--- Figure (f): 3D displacement rendering ---")
    plot_displacement_3d(u, dx2, dy2, dz2, out_dir, c_ds=c_ds, docs_dir=docs_dir)

    #
    # --- Figure (g): Displacement time trace ---
    #
    print("\n--- Figure (g): Displacement time trace ---")
    plot_displacement_time_trace(
        disp_trace,
        dt_shear,
        env_t,
        out_dir,
        docs_dir=docs_dir,
    )

    #
    # --- Additional: final displacement images (using plot_utils) ---
    #
    u_x = u[..., 0]
    u_max = np.abs(u_x).max()
    if u_max == 0:
        u_max = 1.0

    plot_utils.plot_array(
        u_x[:, :, nz_mid_ds],
        aspect=1,
        colorbar=True,
        cmap="RdBu_r",
        xlabel="lateral (y)",
        ylabel="depth (x)",
        vmax=u_max,
        vmin=-u_max,
        export_path=out_dir / "displacement_ux_xy.png",
    )
    plot_utils.plot_array(
        u_x[:, ny_mid_ds, :],
        aspect=1,
        colorbar=True,
        cmap="RdBu_r",
        xlabel="elevational (z)",
        ylabel="depth (x)",
        vmax=u_max,
        vmin=-u_max,
        export_path=out_dir / "displacement_ux_xz.png",
    )

    u_mag = np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2 + u[..., 2] ** 2)
    plot_utils.plot_array(
        u_mag[:, :, nz_mid_ds],
        aspect=1,
        colorbar=True,
        cmap="hot",
        xlabel="lateral (y)",
        ylabel="depth (x)",
        export_path=out_dir / "displacement_magnitude_xy.png",
    )

    #
    # --- Shear wave propagation videos ---
    #
    shear_movie_xy = np.stack(snapshots_xy, axis=0)
    shear_movie_xz = np.stack(snapshots_xz, axis=0)

    shear_vmax = np.abs(shear_movie_xy).max()
    if shear_vmax == 0:
        shear_vmax = 1.0

    n_movie_frames = shear_movie_xy.shape[0]
    print(f"\nShear wave movie: {shear_movie_xy.shape}, max |u_x|={shear_vmax:.3e} m")

    plot_utils.plot_wave_propagation_with_map(
        propagation_map=shear_movie_xy,
        c_map=c_ds[:, :, nz_mid_ds],
        rho_map=rho_ds[:, :, nz_mid_ds],
        export_name=out_dir / "shear_wave_propagation_x-y.mp4",
        vmax=shear_vmax,
        vmin=-shear_vmax,
        figsize=(6, 6),
        num_plot_image=n_movie_frames,
    )
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=shear_movie_xz,
        c_map=c_ds[:, ny_mid_ds, :],
        rho_map=rho_ds[:, ny_mid_ds, :],
        export_name=out_dir / "shear_wave_propagation_x-z.mp4",
        vmax=shear_vmax,
        vmin=-shear_vmax,
        figsize=(6, 6),
        num_plot_image=n_movie_frames,
    )

    print(f"\nAll results saved to {out_dir}")
    print(f"Docs figures copied to {docs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end sparse array transcranial shear wave demo",
    )
    parser.add_argument(
        "--connector-path",
        type=str,
        default=None,
        dest="sparse_mat_path",
        help=f"Path to Sparse_TransConnector.mat (default: {DEFAULT_SPARSE_MAT})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/sparse_transcranial_shear)",
    )
    parser.add_argument(
        "--nrrd-path",
        type=str,
        default=None,
        help="Path to microCT NRRD skull file",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Halle skull from Zenodo (~2.8 GB) if not already cached",
    )
    parser.add_argument(
        "--skip-acoustic",
        action="store_true",
        help="Load saved acoustic results, skip GPU simulation",
    )
    parser.add_argument(
        "--skip-shear",
        action="store_true",
        help="Load saved shear results, skip CPU simulation",
    )
    parser.add_argument(
        "--use-velocity",
        action="store_true",
        help="Use Poynting vector (velocity-based) ARF instead of plane-wave approximation",
    )
    parser.add_argument(
        "--theta-burst",
        action="store_true",
        help="Use theta-burst TUS protocol with neuromodulation-level ISPPA (~30 W/cm²)",
    )
    args = parser.parse_args()
    main(
        nrrd_path=args.nrrd_path,
        sparse_mat_path=args.sparse_mat_path,
        out_dir=args.out_dir,
        download=args.download,
        skip_acoustic=args.skip_acoustic,
        skip_shear=args.skip_shear,
        use_velocity=args.use_velocity,
        theta_burst=args.theta_burst,
    )
