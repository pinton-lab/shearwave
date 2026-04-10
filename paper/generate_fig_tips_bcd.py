#!/usr/bin/env python3
"""Generate combined TIPS acoustical maps + pressure + ARF figure from saved data.

Layout (3 rows × 3 cols):
  Row 0: Sound speed | Density | Attenuation (full-res XY slices)
  Row 1: Pressure t1 | Pressure t2 | Pressure t3 (interpolated XY)
  Row 2: ARF XY slice | ARF XZ slice | 3D rendering (skull + ARF isosurface)
"""

import matplotlib
import numpy as np
from scipy.ndimage import zoom

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# --- Paths ---
data_dir = Path("/home/gfp/fullwave25-private/outputs/tips_transcranial_shear")
out_dir = Path("/home/gfp/fullwave25-private/fullwave/shear_wave/docs/pmb_submission/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Load saved data ---
gp = np.load(data_dir / "grid_params.npz")
dx = float(gp["dx"])
dy = float(gp["dy"])
dz = float(gp["dz"])
nx = int(gp["nx"])
ny = int(gp["ny"])
nz = int(gp["nz"])
mod_x = int(gp["mod_x"])
mod_y = int(gp["mod_y"])
mod_z = int(gp["mod_z"])
sampling_modulus_time = int(gp["sampling_modulus_time"])

c_ds = np.load(data_dir / "c_ds.npy")
rho_ds = np.load(data_dir / "rho_ds.npy")
alpha_ds = np.load(data_dir / "alpha_ds.npy")
propagation_map = np.load(data_dir / "propagation_map.npy")
arf = np.load(data_dir / "radiation_force.npy")

nx_ds, ny_ds, nz_ds = c_ds.shape

# Grid spacings
dx_ds = dx * mod_x
dy_ds = dy * mod_y
dz_ds = dz * mod_z
dx_mm = dx * 1e3
dy_mm = dy * 1e3
dx_ds_mm = dx_ds * 1e3
dy_ds_mm = dy_ds * 1e3
dz_ds_mm = dz_ds * 1e3

# Time step per acoustic snapshot
c0 = 1540.0
cfl = 0.4
dt = cfl * dx / c0
dt_snapshot = dt * sampling_modulus_time

print(f"Grid: {nx}x{ny}x{nz}, dx={dx * 1e6:.1f} um")
print(f"Downsampled: {c_ds.shape}, dx_ds={dx_ds_mm:.3f} mm")

# --- Upsample medium maps to full resolution ---
print("Upsampling medium maps to full resolution...")
zoom_factors = (nx / c_ds.shape[0], ny / c_ds.shape[1], nz / c_ds.shape[2])
c_full = zoom(c_ds, zoom_factors, order=1)
rho_full = zoom(rho_ds, zoom_factors, order=1)
alpha_full = zoom(alpha_ds, zoom_factors, order=1)
print(f"  Full-res maps: {c_full.shape}")

# Mid-slice indices
nz_mid_full = nz // 2
nz_mid_ds = nz_ds // 2
ny_mid_ds = ny_ds // 2

# --- Brain mask for ARF ---
skull_mask_3d = c_ds > 1600
brain_mask = np.ones_like(skull_mask_3d, dtype=bool)
for j in range(ny_ds):
    for k in range(nz_ds):
        col = skull_mask_3d[:, j, k]
        skull_indices = np.where(col)[0]
        if len(skull_indices) > 0:
            brain_mask[: skull_indices.max() + 1, j, k] = False

arf_brain = arf.copy()
arf_brain[~brain_mask] = 0.0
arf_brain[arf_brain < 0] = 0.0
print(f"Brain-masked ARF range: [{arf_brain.min():.0f}, {arf_brain.max():.0f}] N/m³")

# --- Axis extents ---
x_extent_mm = nx_ds * dx_ds_mm
y_extent_mm = ny_ds * dy_ds_mm
z_extent_mm = nz_ds * dz_ds_mm
xy_extent_ds = [0, y_extent_mm, x_extent_mm, 0]
xz_extent_ds = [0, z_extent_mm, x_extent_mm, 0]

skull_xy = skull_mask_3d[:, :, nz_mid_ds]
skull_xz = skull_mask_3d[:, ny_mid_ds, :]
y_coords = np.linspace(0, y_extent_mm, ny_ds)
x_coords = np.linspace(0, x_extent_mm, nx_ds)
z_coords = np.linspace(0, z_extent_mm, nz_ds)

# --- Pick 3 time snapshots ---
nt = propagation_map.shape[0]
t_indices = [100, nt // 2, 500]
t_indices = [min(t, nt - 1) for t in t_indices]
print(f"Pressure snapshot indices: {t_indices} (of {nt})")
for t in t_indices:
    print(f"  t={t} -> {t * dt_snapshot * 1e6:.0f} us")

# --- Build 3x3 figure ---
fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.30)

# ===================== Row 0: Acoustical maps (full res) =====================
map_data = [
    (c_full[:, :, nz_mid_full], "Sound Speed", "m/s"),
    (rho_full[:, :, nz_mid_full], "Density", "kg/m³"),
    (alpha_full[:, :, nz_mid_full], "Attenuation α₀", "dB/cm/MHz"),
]
for col, (field, label, units) in enumerate(map_data):
    ax = fig.add_subplot(gs[0, col])
    n_vert, n_horiz = field.shape
    extent = [0, n_horiz * dy_mm, n_vert * dx_mm, 0]
    im = ax.imshow(field, origin="upper", cmap="turbo", aspect="equal", extent=extent)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(label)
    fig.colorbar(im, ax=ax, label=units)

# ===================== Row 1: Pressure snapshots (3x interpolated) ===========
power = 1.0 / 3.0
interp_factor = 3
all_compressed = []
for t in t_indices:
    p = propagation_map[t, :, :, nz_mid_ds]
    p_up = zoom(p, interp_factor, order=3)
    all_compressed.append(np.sign(p_up) * np.abs(p_up) ** power)
vmax = max(np.abs(f).max() for f in all_compressed)
if vmax == 0:
    vmax = 1.0

skull_mask_xy = c_ds[:, :, nz_mid_ds] > 1600
p_extent = [0, ny_ds * dy_ds_mm, nx_ds * dx_ds_mm, 0]
y_coords_p = np.linspace(p_extent[0], p_extent[1], ny_ds)
x_coords_p = np.linspace(p_extent[3], p_extent[2], nx_ds)

pressure_axes = []
for i, (t_idx, compressed) in enumerate(zip(t_indices, all_compressed)):
    ax = fig.add_subplot(gs[1, i])
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
    if skull_mask_xy.any():
        ax.contour(y_coords_p, x_coords_p, skull_mask_xy, levels=[0.5], colors="k", linewidths=0.8)
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

# ===================== Row 2: ARF slices + 3D =====================
arf_xy = arf_brain[:, :, nz_mid_ds]
arf_xz = arf_brain[:, ny_mid_ds, :]
arf_vmax = max(arf_xy.max(), arf_xz.max())
if arf_vmax == 0:
    arf_vmax = 1.0

# ARF XY slice
ax0 = fig.add_subplot(gs[2, 0])
im_arf = ax0.imshow(
    arf_xy,
    origin="upper",
    cmap="hot",
    aspect="equal",
    vmin=0,
    vmax=arf_vmax,
    extent=xy_extent_ds,
)
if skull_xy.any():
    ax0.contour(y_coords, x_coords, skull_xy, levels=[0.5], colors="w", linewidths=0.8)
ax0.set_xlabel("Lateral (mm)")
ax0.set_ylabel("Depth (mm)")
ax0.set_title("ARF $f_x$ — XY slice")
fig.colorbar(im_arf, ax=ax0, label="N/m³", shrink=0.85)

# ARF XZ slice
ax1 = fig.add_subplot(gs[2, 1])
im_arf2 = ax1.imshow(
    arf_xz,
    origin="upper",
    cmap="hot",
    aspect="equal",
    vmin=0,
    vmax=arf_vmax,
    extent=xz_extent_ds,
)
if skull_xz.any():
    ax1.contour(z_coords, x_coords, skull_xz, levels=[0.5], colors="w", linewidths=0.8)
ax1.set_xlabel("Elevational (mm)")
ax1.set_ylabel("Depth (mm)")
ax1.set_title("ARF $f_x$ — XZ slice")
fig.colorbar(im_arf2, ax=ax1, label="N/m³", shrink=0.85)

# 3D rendering: skull + ARF isosurface
ax3d = fig.add_subplot(gs[2, 2], projection="3d")

try:
    skull_smooth = skull_mask_3d.astype(float)
    verts_skull, faces_skull, _, _ = measure.marching_cubes(
        skull_smooth, level=0.5, spacing=(dx_ds_mm, dy_ds_mm, dz_ds_mm)
    )
    mesh_skull = Poly3DCollection(
        verts_skull[faces_skull],
        alpha=0.15,
        facecolor="lightgray",
        edgecolor="gray",
        linewidth=0.1,
    )
    ax3d.add_collection3d(mesh_skull)
    print(f"Skull surface: {len(faces_skull)} faces")
except Exception as e:
    print(f"Skull surface skipped: {e}")

arf_threshold = arf_vmax * 0.15
try:
    verts_arf, faces_arf, _, _ = measure.marching_cubes(
        arf_brain, level=arf_threshold, spacing=(dx_ds_mm, dy_ds_mm, dz_ds_mm)
    )
    mesh_arf = Poly3DCollection(
        verts_arf[faces_arf],
        alpha=0.6,
        facecolor="orangered",
        edgecolor="darkred",
        linewidth=0.2,
    )
    ax3d.add_collection3d(mesh_arf)
    print(f"ARF isosurface: {len(faces_arf)} faces at {arf_threshold:.0f} N/m³")
except Exception as e:
    print(f"ARF isosurface skipped: {e}")

ax3d.set_xlim(0, x_extent_mm)
ax3d.set_ylim(0, y_extent_mm)
ax3d.set_zlim(0, z_extent_mm)
ax3d.set_xlabel("Depth (mm)", fontsize=8, labelpad=2)
ax3d.set_ylabel("Lateral (mm)", fontsize=8, labelpad=2)
ax3d.set_zlabel("Elev. (mm)", fontsize=8, labelpad=2)
ax3d.set_title("3D: Skull + ARF")
ax3d.view_init(elev=25, azim=-60)
ax3d.tick_params(labelsize=7)

# --- Save ---
out_path = out_dir / "fig_tips_bcd_medium_pressure_arf.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
