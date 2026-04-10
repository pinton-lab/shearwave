#!/usr/bin/env python3
"""Generate TIPS acoustic strain and strain gradient figure from saved data.

Layout (2 rows × 3 cols):
  Row 0: Peak strain XY | Peak strain XZ | RMS strain XY
  Row 1: Strain gradient magnitude XY | Strain gradient magnitude XZ | RMS gradient XY
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec

# --- Paths ---
data_dir = Path("/home/gfp/fullwave25-private/outputs/tips_transcranial_shear")
out_dir = Path("/home/gfp/fullwave25-private/fullwave/shear_wave/docs/pmb_submission/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Load data ---
gp = np.load(data_dir / "grid_params.npz")
dx_ds_mm = float(gp["dx"]) * int(gp["mod_x"]) * 1e3
dy_ds_mm = float(gp["dy"]) * int(gp["mod_y"]) * 1e3
dz_ds_mm = float(gp["dz"]) * int(gp["mod_z"]) * 1e3

c_ds = np.load(data_dir / "c_ds.npy")
s = np.load(data_dir / "acoustic_strain.npz")

nx_ds, ny_ds, nz_ds = c_ds.shape
nz_mid = nz_ds // 2
ny_mid = ny_ds // 2

# Axis extents
x_extent_mm = nx_ds * dx_ds_mm
y_extent_mm = ny_ds * dy_ds_mm
z_extent_mm = nz_ds * dz_ds_mm
xy_extent = [0, y_extent_mm, x_extent_mm, 0]
xz_extent = [0, z_extent_mm, x_extent_mm, 0]

skull_xy = c_ds[:, :, nz_mid] > 1600
skull_xz = c_ds[:, ny_mid, :] > 1600
y_coords = np.linspace(0, y_extent_mm, ny_ds)
x_coords = np.linspace(0, x_extent_mm, nx_ds)
z_coords = np.linspace(0, z_extent_mm, nz_ds)

strain_peak = s["strain_peak"]
strain_rms = s["strain_rms"]
grad_peak = s["grad_peak_magnitude"]
grad_rms = s["grad_rms_magnitude"]

print(f"Grid: {nx_ds}x{ny_ds}x{nz_ds}, dx_ds={dx_ds_mm:.3f} mm")
print(f"Peak strain: {strain_peak.max():.3e}")
print(f"RMS strain: {strain_rms.max():.3e}")
print(f"Peak gradient: {grad_peak.max():.3e} /m")
print(f"RMS gradient: {grad_rms.max():.3e} /m")

# --- Build figure ---
fig = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.30)


# Helper to plot a slice with skull contour
def plot_slice(ax, data, extent, skull, coords_h, coords_v, title, cbar_label, cmap="inferno"):
    im = ax.imshow(data, origin="upper", cmap=cmap, aspect="equal", extent=extent, vmin=0)
    if skull.any():
        ax.contour(coords_h, coords_v, skull, levels=[0.5], colors="w", linewidths=0.5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
    return im


# Row 0: Strain
ax = fig.add_subplot(gs[0, 0])
plot_slice(
    ax,
    strain_peak[:, :, nz_mid],
    xy_extent,
    skull_xy,
    y_coords,
    x_coords,
    "Peak Strain — XY",
    "ε (dimensionless)",
)
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Depth (mm)")

ax = fig.add_subplot(gs[0, 1])
plot_slice(
    ax,
    strain_peak[:, ny_mid, :],
    xz_extent,
    skull_xz,
    z_coords,
    x_coords,
    "Peak Strain — XZ",
    "ε (dimensionless)",
)
ax.set_xlabel("Elevational (mm)")
ax.set_ylabel("Depth (mm)")

ax = fig.add_subplot(gs[0, 2])
plot_slice(
    ax,
    strain_rms[:, :, nz_mid],
    xy_extent,
    skull_xy,
    y_coords,
    x_coords,
    "RMS Strain — XY",
    "ε (dimensionless)",
)
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Depth (mm)")

# Row 1: Strain gradient
ax = fig.add_subplot(gs[1, 0])
plot_slice(
    ax,
    grad_peak[:, :, nz_mid],
    xy_extent,
    skull_xy,
    y_coords,
    x_coords,
    "Peak Strain Gradient — XY",
    "|∇ε| (m⁻¹)",
)
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Depth (mm)")

ax = fig.add_subplot(gs[1, 1])
plot_slice(
    ax,
    grad_peak[:, ny_mid, :],
    xz_extent,
    skull_xz,
    z_coords,
    x_coords,
    "Peak Strain Gradient — XZ",
    "|∇ε| (m⁻¹)",
)
ax.set_xlabel("Elevational (mm)")
ax.set_ylabel("Depth (mm)")

ax = fig.add_subplot(gs[1, 2])
plot_slice(
    ax,
    grad_rms[:, :, nz_mid],
    xy_extent,
    skull_xy,
    y_coords,
    x_coords,
    "RMS Strain Gradient — XY",
    "|∇ε| (m⁻¹)",
)
ax.set_xlabel("Lateral (mm)")
ax.set_ylabel("Depth (mm)")

# --- Save ---
out_path = out_dir / "fig_tips_strain.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
