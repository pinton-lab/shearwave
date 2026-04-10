#!/usr/bin/env python3
"""Generate TIPS displacement snapshots + time traces figure from saved data.

Layout (3 rows × 3 cols):
  Row 0: Displacement XY snapshots at 3 time points
  Row 1: Displacement XZ snapshots at 3 time points
  Row 2: Time trace at lateral offsets | depth offsets | sensor location map
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

# --- Load saved data ---
gp = np.load(data_dir / "grid_params.npz")
dx = float(gp["dx"])
dy = float(gp["dy"])
dz = float(gp["dz"])
mod_x = int(gp["mod_x"])
mod_y = int(gp["mod_y"])
mod_z = int(gp["mod_z"])

dx_ds = dx * mod_x
dy_ds = dy * mod_y
dz_ds = dz * mod_z
dx_ds_mm = dx_ds * 1e3
dy_ds_mm = dy_ds * 1e3
dz_ds_mm = dz_ds * 1e3

sp = np.load(data_dir / "shear_params.npz")
dt_shear = float(sp["dt_shear"])
save_every = int(sp["save_every"])

c_ds = np.load(data_dir / "c_ds.npy")
arf = np.load(data_dir / "radiation_force.npy")
snapshots_xy = np.load(data_dir / "snapshots_xy.npy")
snapshots_xz = np.load(data_dir / "snapshots_xz.npy")

nx_ds, ny_ds, nz_ds = c_ds.shape
nz_mid = nz_ds // 2
ny_mid = ny_ds // 2

dt_frame = save_every * dt_shear
n_frames = snapshots_xy.shape[0]
t_axis_ms = np.arange(n_frames) * dt_frame * 1e3

print(f"Downsampled grid: {nx_ds}x{ny_ds}x{nz_ds}, dx_ds={dx_ds_mm:.3f} mm")
print(f"Snapshots: {n_frames} frames, dt_frame={dt_frame * 1e3:.3f} ms")

# --- Axis extents ---
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

# --- Brain-masked ARF peak location ---
# The raw ARF peak may be in the coupling medium (before the skull).
# Use brain-masked ARF to find the true intracranial peak.
skull_mask_3d = c_ds > 1600
brain_mask = np.ones_like(skull_mask_3d, dtype=bool)
for j in range(ny_ds):
    for k in range(nz_ds):
        col = skull_mask_3d[:, j, k]
        si = np.where(col)[0]
        if len(si) > 0:
            brain_mask[: si.max() + 1, j, k] = False

arf_brain = arf.copy()
arf_brain[~brain_mask] = 0.0
arf_brain[arf_brain < 0] = 0.0

peak_idx = np.unravel_index(arf_brain.argmax(), arf_brain.shape)
ix0, iy0, iz0 = peak_idx
peak_depth_mm = ix0 * dx_ds_mm
peak_lat_mm = iy0 * dy_ds_mm
print(f"Brain-masked ARF peak at downsampled index ({ix0}, {iy0}, {iz0})")
print(f"  = ({peak_depth_mm:.1f}, {peak_lat_mm:.1f}) mm in (depth, lateral)")
print(f"  val={arf_brain[peak_idx]:.0f} N/m³ (raw peak={arf.max():.0f})")

# --- Pick 3 displacement time snapshots ---
frame_indices = [7, 17, 30]
frame_indices = [min(f, n_frames - 1) for f in frame_indices]
frame_times_ms = [f * dt_frame * 1e3 for f in frame_indices]

# Global colorscale for displacement
disp_vmax = 0.0
for idx in frame_indices:
    disp_vmax = max(disp_vmax, np.abs(snapshots_xy[idx]).max(), np.abs(snapshots_xz[idx]).max())
if disp_vmax == 0:
    disp_vmax = 1.0

# --- Define sensor points for time traces ---
# Lateral offsets from peak (same depth)
lat_offsets_vox = [0, 5, 10, 15]
lat_labels = []
lat_traces = []
for dy_vox in lat_offsets_vox:
    iy = iy0 + dy_vox
    if 0 <= iy < ny_ds:
        tr = np.abs(snapshots_xy[:, ix0, iy])
        dist_mm = dy_vox * dy_ds_mm
        lat_traces.append(tr)
        if dy_vox == 0:
            lat_labels.append("Focus (0 mm)")
        else:
            lat_labels.append(f"+{dist_mm:.1f} mm")

# Depth offsets from peak (same lateral)
depth_offsets_vox = [0, 5, 10, 15]
depth_labels = []
depth_traces = []
for dx_vox in depth_offsets_vox:
    ixx = ix0 + dx_vox
    if 0 <= ixx < nx_ds:
        tr = np.abs(snapshots_xy[:, ixx, iy0])
        dist_mm = dx_vox * dx_ds_mm
        depth_traces.append(tr)
        if dx_vox == 0:
            depth_labels.append("Focus (0 mm)")
        else:
            depth_labels.append(f"+{dist_mm:.1f} mm")

# Print trace info
for lbl, tr in zip(lat_labels, lat_traces):
    print(f"  Lateral {lbl}: peak={tr.max() * 1e6:.2f} um at frame {tr.argmax()}")
for lbl, tr in zip(depth_labels, depth_traces):
    print(f"  Depth {lbl}: peak={tr.max() * 1e6:.2f} um at frame {tr.argmax()}")

# --- Build figure ---
fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

# ===================== Row 0: XY displacement snapshots =====================
disp_xy_axes = []
for col, idx in enumerate(frame_indices):
    ax = fig.add_subplot(gs[0, col])
    disp_xy_axes.append(ax)
    im_d = ax.imshow(
        np.abs(snapshots_xy[idx]),
        origin="upper",
        cmap="hot",
        aspect="equal",
        vmin=0,
        vmax=disp_vmax,
        extent=xy_extent,
    )
    if skull_xy.any():
        ax.contour(y_coords, x_coords, skull_xy, levels=[0.5], colors="w", linewidths=0.5)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"|u| XY — t = {frame_times_ms[col]:.1f} ms")

# ===================== Row 1: XZ displacement snapshots =====================
disp_xz_axes = []
for col, idx in enumerate(frame_indices):
    ax = fig.add_subplot(gs[1, col])
    disp_xz_axes.append(ax)
    im_d = ax.imshow(
        np.abs(snapshots_xz[idx]),
        origin="upper",
        cmap="hot",
        aspect="equal",
        vmin=0,
        vmax=disp_vmax,
        extent=xz_extent,
    )
    if skull_xz.any():
        ax.contour(z_coords, x_coords, skull_xz, levels=[0.5], colors="w", linewidths=0.5)
    ax.set_xlabel("Elevational (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"|u| XZ — t = {frame_times_ms[col]:.1f} ms")

# Shared colorbar for snapshots
fig.colorbar(
    im_d,
    ax=disp_xy_axes + disp_xz_axes,
    shrink=0.7,
    pad=0.02,
    label="|u| (m)",
)

# ===================== Row 2: Time traces =====================
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

# Panel 1: Lateral offsets
ax_lat = fig.add_subplot(gs[2, 0])
for i, (tr, lbl) in enumerate(zip(lat_traces, lat_labels)):
    ax_lat.plot(t_axis_ms, tr * 1e6, label=lbl, color=colors[i], linewidth=1.5)
ax_lat.set_xlabel("Time (ms)")
ax_lat.set_ylabel("|u| (μm)")
ax_lat.set_title("Displacement — lateral offsets")
ax_lat.legend(fontsize=8)
ax_lat.set_xlim(0, t_axis_ms[-1])
ax_lat.grid(True, alpha=0.3)

# Panel 2: Depth offsets
ax_dep = fig.add_subplot(gs[2, 1])
for i, (tr, lbl) in enumerate(zip(depth_traces, depth_labels)):
    ax_dep.plot(t_axis_ms, tr * 1e6, label=lbl, color=colors[i], linewidth=1.5)
ax_dep.set_xlabel("Time (ms)")
ax_dep.set_ylabel("|u| (μm)")
ax_dep.set_title("Displacement — depth offsets")
ax_dep.legend(fontsize=8)
ax_dep.set_xlim(0, t_axis_ms[-1])
ax_dep.grid(True, alpha=0.3)

# Panel 3: Sensor locations on ARF XY map
ax_map = fig.add_subplot(gs[2, 2])
arf_xy = arf[:, :, nz_mid].copy()
arf_xy[arf_xy < 0] = 0
im_loc = ax_map.imshow(
    arf_xy,
    origin="upper",
    cmap="hot",
    aspect="equal",
    vmin=0,
    vmax=arf_xy.max(),
    extent=xy_extent,
)
if skull_xy.any():
    ax_map.contour(y_coords, x_coords, skull_xy, levels=[0.5], colors="w", linewidths=0.5)

# Mark lateral offset sensors
for i, dy_vox in enumerate(lat_offsets_vox):
    iy = iy0 + dy_vox
    if 0 <= iy < ny_ds:
        yy = iy * dy_ds_mm
        xx = ix0 * dx_ds_mm
        ax_map.plot(
            yy,
            xx,
            "o",
            color=colors[i],
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

# Mark depth offset sensors (skip 0 — already plotted)
for i, dx_vox in enumerate(depth_offsets_vox):
    if dx_vox == 0:
        continue
    ixx = ix0 + dx_vox
    if 0 <= ixx < nx_ds:
        yy = iy0 * dy_ds_mm
        xx = ixx * dx_ds_mm
        ax_map.plot(
            yy,
            xx,
            "s",
            color=colors[i],
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

ax_map.set_xlabel("Lateral (mm)")
ax_map.set_ylabel("Depth (mm)")
ax_map.set_title("Sensor locations")
fig.colorbar(im_loc, ax=ax_map, label="ARF (N/m³)", shrink=0.85)

# --- Save ---
out_path = out_dir / "fig_tips_de_displacement_and_traces.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
