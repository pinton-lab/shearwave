#!/usr/bin/env python3
"""Generate composite TIPS transcranial figure combining figures 7-10.

Layout (3 rows × 3 cols via GridSpec(3, 9)):
  Row 0 (a,b,c): Axial skull overview | Sagittal skull | 3D transducer + ARF + skull
  Row 1 (d,e,f): Pressure t=28 us | Pressure t=104 us | ARF XY slice
  Row 2 (g,h):   Peak strain XY | Displacement profiles (lateral offset)
"""

import matplotlib
import numpy as np
from scipy.ndimage import maximum_filter, median_filter, zoom

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# --- Paths ---
data_dir = Path("/home/gfp/fullwave25-private/outputs/tips_transcranial_shear")
nrrd_path = Path.home() / ".cache/fullwave25/skull_microCT_zenodo/halle_skull.nrrd"
out_dir = Path("/home/gfp/fullwave25-private/fullwave/shear_wave/docs/pmb_submission/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# --- TIPS geometry constants ---
ROC_MM = 80.0  # radius of curvature
INNER_R_MM = 20.5  # inner aperture radius
OUTER_R_MM = 46.0  # outer aperture radius
XDC_CENTER_LPS = (140.6, 118.5, 85.0)  # transducer center in LPS mm
NORMAL_LP = (-0.954, -0.298)  # inward beam direction (L, P)
TANGENT_LP = (-0.298, 0.954)  # along-face direction (L, P)

# Geometric focal spot: ROC along the normal from transducer center
FOCAL_L = XDC_CENTER_LPS[0] + ROC_MM * NORMAL_LP[0]  # 64.3 mm
FOCAL_P = XDC_CENTER_LPS[1] + ROC_MM * NORMAL_LP[1]  # 94.7 mm
FOCAL_S = XDC_CENTER_LPS[2]  # 85.0 mm

# --- Load saved simulation data ---
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
arf = np.load(data_dir / "radiation_force.npy")
propagation_map = np.load(data_dir / "propagation_map.npy")

sp = np.load(data_dir / "shear_params.npz")
dt_shear = float(sp["dt_shear"])
save_every = int(sp["save_every"])

snapshots_xy = np.load(data_dir / "snapshots_xy.npy")
strain_data = np.load(data_dir / "acoustic_strain.npz")
strain_peak = strain_data["strain_peak"]

nx_ds, ny_ds, nz_ds = c_ds.shape

# Grid spacings
dx_ds = dx * mod_x
dy_ds = dy * mod_y
dz_ds = dz * mod_z
dx_ds_mm = dx_ds * 1e3
dy_ds_mm = dy_ds * 1e3
dz_ds_mm = dz_ds * 1e3

# Acoustic time step
c0 = 1540.0
cfl = 0.4
dt = cfl * dx / c0
dt_snapshot = dt * sampling_modulus_time

# Shear time step
dt_frame = save_every * dt_shear
n_frames = snapshots_xy.shape[0]
t_axis_ms = np.arange(n_frames) * dt_frame * 1e3

nz_mid = nz_ds // 2
ny_mid = ny_ds // 2

print(f"Downsampled grid: {nx_ds}x{ny_ds}x{nz_ds}, dx_ds={dx_ds_mm:.3f} mm")

# --- Axis extents ---
x_extent_mm = nx_ds * dx_ds_mm
y_extent_mm = ny_ds * dy_ds_mm
z_extent_mm = nz_ds * dz_ds_mm
xy_extent = [0, y_extent_mm, x_extent_mm, 0]

skull_mask_3d = c_ds > 1600
skull_xy = skull_mask_3d[:, :, nz_mid]
y_coords = np.linspace(0, y_extent_mm, ny_ds)
x_coords = np.linspace(0, x_extent_mm, nx_ds)
z_coords = np.linspace(0, z_extent_mm, nz_ds)

# --- Brain mask for ARF and strain ---
# Build a smooth skull-depth surface.  Per-column deepest skull voxel,
# fill gaps with maximum_filter, smooth with median_filter, then ensure
# the smoothed depth never regresses above the raw boundary.
raw_depth = np.full((ny_ds, nz_ds), -1.0)
for j in range(ny_ds):
    for k in range(nz_ds):
        col = skull_mask_3d[:, j, k]
        skull_indices = np.where(col)[0]
        if len(skull_indices) > 0:
            raw_depth[j, k] = skull_indices.max()

smooth_depth = raw_depth.copy()
for _ in range(10):
    expanded = maximum_filter(smooth_depth, size=5)
    no_data = smooth_depth < 0
    smooth_depth[no_data] = expanded[no_data]
smooth_depth = median_filter(smooth_depth, size=7)

# Never let the smoothed depth sit shallower than the raw skull boundary
has_raw = raw_depth >= 0
smooth_depth[has_raw] = np.maximum(smooth_depth[has_raw], raw_depth[has_raw])

brain_mask = np.zeros_like(skull_mask_3d, dtype=bool)
for j in range(ny_ds):
    for k in range(nz_ds):
        d = int(smooth_depth[j, k])
        if d >= 0:
            brain_mask[d + 1 :, j, k] = True

arf_brain = arf.copy()
arf_brain[~brain_mask] = 0.0
arf_brain[arf_brain < 0] = 0.0
print(f"Brain-masked ARF range: [{arf_brain.min():.0f}, {arf_brain.max():.0f}] N/m3")

# Apply same mask to strain
strain_brain = strain_peak.copy()
strain_brain[~brain_mask] = 0.0

# --- ARF peak location (for displacement traces) ---
peak_idx = np.unravel_index(arf_brain.argmax(), arf_brain.shape)
ix0, iy0, iz0 = peak_idx
print(f"Brain-masked ARF peak at ({ix0}, {iy0}, {iz0})")

# --- Reconstruct transducer source mask from icc.dat ---
icc = np.fromfile(data_dir / "txrx_0" / "icc.dat", dtype=np.int32)
n_src = len(icc) // 3
src_coords = icc.reshape(n_src, 3)
cx_ds = np.clip(src_coords[:, 0] // mod_x, 0, nx_ds - 1)
cy_ds = np.clip(src_coords[:, 1] // mod_y, 0, ny_ds - 1)
cz_ds = np.clip(src_coords[:, 2] // mod_z, 0, nz_ds - 1)
src_mask_ds = np.zeros((nx_ds, ny_ds, nz_ds), dtype=bool)
src_mask_ds[cx_ds, cy_ds, cz_ds] = True
print(f"Transducer source mask: {src_mask_ds.sum()} voxels")

# --- Pressure snapshot indices for 28 us and 104 us ---
t_targets_us = [28, 104]
t_indices = [int(round(t * 1e-6 / dt_snapshot)) for t in t_targets_us]
nt = propagation_map.shape[0]
t_indices = [min(t, nt - 1) for t in t_indices]
for t_us, idx in zip(t_targets_us, t_indices):
    print(f"Pressure target {t_us} us -> frame {idx} ({idx * dt_snapshot * 1e6:.1f} us)")

# --- Load full skull NRRD for overview panels ---
print("Loading full skull NRRD for overview panels...")
import nrrd

nrrd_data, nrrd_header = nrrd.read(str(nrrd_path))
nrrd_spacing = np.array([np.linalg.norm(d) for d in nrrd_header["space directions"]])
print(f"  NRRD shape: {nrrd_data.shape}, spacing: {nrrd_spacing} mm")

# Axial slice at S = 85 mm
s_slice_idx = int(round(XDC_CENTER_LPS[2] / nrrd_spacing[2]))
s_slice_idx = min(s_slice_idx, nrrd_data.shape[2] - 1)
axial_slice = nrrd_data[:, :, s_slice_idx].astype(float)
print(f"  Axial slice at S={XDC_CENTER_LPS[2]} mm (index {s_slice_idx})")

# Sagittal slice at L = focal_L (passes through the geometric focus)
l_slice_idx = int(round(FOCAL_L / nrrd_spacing[0]))
l_slice_idx = min(l_slice_idx, nrrd_data.shape[0] - 1)
sagittal_slice = nrrd_data[l_slice_idx, :, :].astype(float)
print(f"  Sagittal slice at L={FOCAL_L:.1f} mm (index {l_slice_idx})")

# Coordinate axes
l_axis_mm = np.arange(nrrd_data.shape[0]) * nrrd_spacing[0]
p_axis_mm = np.arange(nrrd_data.shape[1]) * nrrd_spacing[1]
s_axis_mm = np.arange(nrrd_data.shape[2]) * nrrd_spacing[2]

del nrrd_data
print("  NRRD volume freed from memory")

# --- Compute TIPS bowl arc for 2D overlays ---
norm_l, norm_p = NORMAL_LP
tang_l, tang_p = TANGENT_LP
xdc_l, xdc_p, xdc_s = XDC_CENTER_LPS

# Parameterize arc by lateral distance r along the tangent
r_all = np.linspace(-OUTER_R_MM, OUTER_R_MM, 800)
# Annular mask: only where |r| >= inner radius
annular = np.abs(r_all) >= INNER_R_MM
r_arc = r_all[annular]
d_arc = ROC_MM - np.sqrt(ROC_MM**2 - r_arc**2)
arc_l = xdc_l + d_arc * norm_l + r_arc * tang_l
arc_p = xdc_p + d_arc * norm_p + r_arc * tang_p

# Split into two arcs (positive and negative r halves for clean plotting)
r_pos = r_arc >= 0
r_neg = r_arc < 0

# --- Compute TIPS bowl ring projections for sagittal view ---
# Project the outer and inner rims of the annular bowl onto the P-S plane.
# On each rim, r_total = const, so depth d is constant.
# Parameterize by angle theta around the bowl aperture circle.
theta_ring = np.linspace(0, 2 * np.pi, 400)

d_outer = ROC_MM - np.sqrt(ROC_MM**2 - OUTER_R_MM**2)
ring_outer_p = xdc_p + d_outer * norm_p + OUTER_R_MM * np.cos(theta_ring) * tang_p
ring_outer_s = xdc_s + OUTER_R_MM * np.sin(theta_ring)

d_inner = ROC_MM - np.sqrt(ROC_MM**2 - INNER_R_MM**2)
ring_inner_p = xdc_p + d_inner * norm_p + INNER_R_MM * np.cos(theta_ring) * tang_p
ring_inner_s = xdc_s + INNER_R_MM * np.sin(theta_ring)

print(f"Geometric focus at LPS=({FOCAL_L:.1f}, {FOCAL_P:.1f}, {FOCAL_S:.1f}) mm")

# =========================================================================
# Build composite figure
# =========================================================================
fig = plt.figure(figsize=(18, 15.5), facecolor="white")
gs = gridspec.GridSpec(
    3,
    9,
    figure=fig,
    hspace=0.28,
    wspace=0.45,
    height_ratios=[1.0, 0.95, 0.95],
)

label_fontsize = 13
arc_color = "gold"
focus_color = "cyan"
beam_color = "gold"

# ===================== Row 0: Skull overviews + 3D panel =====================

# --- (a) Axial skull overview with TIPS arc + focal spot ---
ax_ax = fig.add_subplot(gs[0, 0:3])
ax_ax.set_facecolor("black")

bone_ax = np.clip(axial_slice, -200, 1973)
skull_extent_ax = [0, p_axis_mm[-1], l_axis_mm[-1], 0]
ax_ax.imshow(
    bone_ax,
    origin="upper",
    cmap="bone",
    aspect="equal",
    extent=skull_extent_ax,
)

# Draw TIPS bowl arc (two halves of the annulus)
ax_ax.plot(arc_p[r_neg], arc_l[r_neg], color=arc_color, linewidth=2.5, solid_capstyle="round")
ax_ax.plot(arc_p[r_pos], arc_l[r_pos], color=arc_color, linewidth=2.5, solid_capstyle="round")

# Beam axis: dashed line from transducer center to focal spot
ax_ax.plot(
    [xdc_p, FOCAL_P],
    [xdc_l, FOCAL_L],
    color=beam_color,
    linewidth=1.2,
    linestyle="--",
    alpha=0.8,
)

# Geometric focal spot
ax_ax.plot(
    FOCAL_P,
    FOCAL_L,
    "*",
    color=focus_color,
    markersize=14,
    markeredgecolor="white",
    markeredgewidth=0.5,
)

ax_ax.set_xlabel("Posterior (mm)")
ax_ax.set_ylabel("Left (mm)")
ax_ax.set_title("(a) Axial \u2014 TIPS array + geometric focus", fontsize=label_fontsize)

# --- (b) Sagittal skull overview with TIPS + focal spot ---
ax_sag = fig.add_subplot(gs[0, 3:6])
ax_sag.set_facecolor("black")

bone_sag = np.clip(sagittal_slice, -200, 1973)
# Transpose so P is on x-axis, S on y-axis (superior up)
skull_extent_sag = [0, p_axis_mm[-1], 0, s_axis_mm[-1]]
ax_sag.imshow(
    bone_sag.T,
    origin="lower",
    cmap="bone",
    aspect="equal",
    extent=skull_extent_sag,
)

# TIPS bowl: draw outer and inner rim projections (annular ring)
ax_sag.plot(ring_outer_p, ring_outer_s, color=arc_color, linewidth=2.0, solid_capstyle="round")
ax_sag.plot(ring_inner_p, ring_inner_s, color=arc_color, linewidth=2.0, solid_capstyle="round")

# Beam axis projected into P-S plane
beam_ext = 1.3
ext_p = xdc_p + beam_ext * (xdc_p - FOCAL_P)
ext_s = xdc_s + beam_ext * (xdc_s - FOCAL_S)
ax_sag.plot(
    [ext_p, FOCAL_P],
    [ext_s, FOCAL_S],
    color=beam_color,
    linewidth=1.2,
    linestyle="--",
    alpha=0.8,
)

# Geometric focal spot
ax_sag.plot(
    FOCAL_P,
    FOCAL_S,
    "*",
    color=focus_color,
    markersize=14,
    markeredgecolor="white",
    markeredgewidth=0.5,
)

ax_sag.set_xlabel("Posterior (mm)")
ax_sag.set_ylabel("Superior (mm)")
ax_sag.set_title("(b) Sagittal \u2014 TIPS array + geometric focus", fontsize=label_fontsize)

# --- (c) Combined 3D: TIPS transducer + ARF + skull ---
ax_3d = fig.add_subplot(gs[0, 6:9], projection="3d")

# Skull isosurface
try:
    skull_smooth = skull_mask_3d.astype(float)
    verts_skull, faces_skull, _, _ = measure.marching_cubes(
        skull_smooth, level=0.5, spacing=(dx_ds_mm, dy_ds_mm, dz_ds_mm)
    )
    mesh_skull = Poly3DCollection(
        verts_skull[faces_skull],
        alpha=0.10,
        facecolor="lightblue",
        edgecolor="steelblue",
        linewidth=0.03,
    )
    ax_3d.add_collection3d(mesh_skull)
    print(f"(c) Skull surface: {len(faces_skull)} faces")
except Exception as e:
    print(f"(c) Skull surface skipped: {e}")

# Transducer isosurface
try:
    src_smooth = src_mask_ds.astype(float)
    verts_src, faces_src, _, _ = measure.marching_cubes(
        src_smooth, level=0.5, spacing=(dx_ds_mm, dy_ds_mm, dz_ds_mm)
    )
    mesh_src = Poly3DCollection(
        verts_src[faces_src],
        alpha=0.7,
        facecolor="gold",
        edgecolor="darkorange",
        linewidth=0.1,
    )
    ax_3d.add_collection3d(mesh_src)
    print(f"(c) Transducer surface: {len(faces_src)} faces")
except Exception as e:
    print(f"(c) Transducer surface skipped: {e}")

# ARF isosurface (colored by intensity)
arf_vmax_3d = arf_brain.max()
arf_threshold = arf_vmax_3d * 0.15
cmap_arf = plt.cm.hot
norm_arf = plt.Normalize(vmin=0, vmax=arf_vmax_3d)
try:
    verts_arf, faces_arf, _, _ = measure.marching_cubes(
        arf_brain, level=arf_threshold, spacing=(dx_ds_mm, dy_ds_mm, dz_ds_mm)
    )
    face_centers = verts_arf[faces_arf].mean(axis=1)
    fc_vox = (face_centers / np.array([dx_ds_mm, dy_ds_mm, dz_ds_mm])).astype(int)
    fc_vox = np.clip(fc_vox, 0, np.array(arf_brain.shape) - 1)
    face_vals = arf_brain[fc_vox[:, 0], fc_vox[:, 1], fc_vox[:, 2]]
    face_colors = cmap_arf(norm_arf(face_vals))
    face_colors[:, 3] = 0.7

    mesh_arf = Poly3DCollection(
        verts_arf[faces_arf],
        facecolors=face_colors,
        edgecolor="none",
    )
    ax_3d.add_collection3d(mesh_arf)
    print(f"(c) ARF isosurface: {len(faces_arf)} faces at {arf_threshold:.0f} N/m3")
except Exception as e:
    print(f"(c) ARF isosurface skipped: {e}")

ax_3d.set_xlim(0, x_extent_mm)
ax_3d.set_ylim(0, y_extent_mm)
ax_3d.set_zlim(0, z_extent_mm)
ax_3d.set_xlabel("Depth (mm)", fontsize=8, labelpad=2)
ax_3d.set_ylabel("Lateral (mm)", fontsize=8, labelpad=2)
ax_3d.set_zlabel("Elev. (mm)", fontsize=8, labelpad=2)
ax_3d.set_title("(c) 3D: Transducer + ARF + Skull", fontsize=label_fontsize)
ax_3d.view_init(elev=25, azim=-55)
ax_3d.tick_params(labelsize=7)

# Colorbar for 3D ARF
sm_arf = plt.cm.ScalarMappable(cmap=cmap_arf, norm=norm_arf)
sm_arf.set_array([])
fig.colorbar(sm_arf, ax=ax_3d, shrink=0.55, pad=0.08, label="ARF (N/m\u00b3)")

# ===================== Row 1: Pressure + ARF 2D =====================

# Prepare pressure data
power = 1.0 / 3.0
interp_factor = 3
all_compressed = []
for t in t_indices:
    p = propagation_map[t, :, :, nz_mid]
    p_up = zoom(p, interp_factor, order=3)
    all_compressed.append(np.sign(p_up) * np.abs(p_up) ** power)
vmax_p = max(np.abs(f).max() for f in all_compressed)
if vmax_p == 0:
    vmax_p = 1.0

skull_mask_xy_p = c_ds[:, :, nz_mid] > 1600
p_extent = [0, ny_ds * dy_ds_mm, nx_ds * dx_ds_mm, 0]
y_coords_p = np.linspace(p_extent[0], p_extent[1], ny_ds)
x_coords_p = np.linspace(p_extent[3], p_extent[2], nx_ds)

# --- (d) Pressure at 28 us ---
ax_p1 = fig.add_subplot(gs[1, 0:3])
im_p1 = ax_p1.imshow(
    all_compressed[0],
    origin="upper",
    cmap="seismic",
    aspect="equal",
    vmin=-vmax_p,
    vmax=vmax_p,
    extent=p_extent,
)
if skull_mask_xy_p.any():
    ax_p1.contour(y_coords_p, x_coords_p, skull_mask_xy_p, levels=[0.5], colors="k", linewidths=0.8)
ax_p1.set_xlabel("Lateral (mm)")
ax_p1.set_ylabel("Depth (mm)")
t_us_1 = t_indices[0] * dt_snapshot * 1e6
ax_p1.set_title(f"(d) Pressure \u2014 t = {t_us_1:.0f} \u03bcs", fontsize=label_fontsize)

# --- (e) Pressure at 104 us ---
ax_p2 = fig.add_subplot(gs[1, 3:6])
im_p2 = ax_p2.imshow(
    all_compressed[1],
    origin="upper",
    cmap="seismic",
    aspect="equal",
    vmin=-vmax_p,
    vmax=vmax_p,
    extent=p_extent,
)
if skull_mask_xy_p.any():
    ax_p2.contour(y_coords_p, x_coords_p, skull_mask_xy_p, levels=[0.5], colors="k", linewidths=0.8)
ax_p2.set_xlabel("Lateral (mm)")
ax_p2.set_ylabel("Depth (mm)")
t_us_2 = t_indices[1] * dt_snapshot * 1e6
ax_p2.set_title(f"(e) Pressure \u2014 t = {t_us_2:.0f} \u03bcs", fontsize=label_fontsize)

# Shared colorbar for pressure
fig.colorbar(
    im_p2,
    ax=[ax_p1, ax_p2],
    shrink=0.85,
    pad=0.02,
    label="sgn(p) |p|$^{1/3}$ [Pa$^{1/3}$]",
)

# --- (f) ARF XY slice ---
ax_arf2d = fig.add_subplot(gs[1, 6:9])
arf_xy = arf_brain[:, :, nz_mid]
arf_vmax = arf_xy.max()
if arf_vmax == 0:
    arf_vmax = 1.0
im_arf = ax_arf2d.imshow(
    arf_xy,
    origin="upper",
    cmap="hot",
    aspect="equal",
    vmin=0,
    vmax=arf_vmax,
    extent=xy_extent,
)
if skull_xy.any():
    ax_arf2d.contour(y_coords, x_coords, skull_xy, levels=[0.5], colors="w", linewidths=0.8)
ax_arf2d.set_xlabel("Lateral (mm)")
ax_arf2d.set_ylabel("Depth (mm)")
ax_arf2d.set_title("(f) ARF $f_x$ \u2014 XY slice", fontsize=label_fontsize)
fig.colorbar(im_arf, ax=ax_arf2d, label="N/m\u00b3", shrink=0.85)

# ===================== Row 2: Strain + Displacement traces =====================

# --- (g) Peak strain XY ---
ax_strain = fig.add_subplot(gs[2, 0:4])
strain_xy = strain_brain[:, :, nz_mid]
im_strain = ax_strain.imshow(
    strain_xy,
    origin="upper",
    cmap="inferno",
    aspect="equal",
    vmin=0,
    extent=xy_extent,
)
if skull_xy.any():
    ax_strain.contour(y_coords, x_coords, skull_xy, levels=[0.5], colors="w", linewidths=0.5)
ax_strain.set_xlabel("Lateral (mm)")
ax_strain.set_ylabel("Depth (mm)")
ax_strain.set_title("(g) Peak Acoustic Strain \u2014 XY slice", fontsize=label_fontsize)
fig.colorbar(im_strain, ax=ax_strain, label="\u03b5 (dimensionless)", shrink=0.85)

# --- (h) Displacement profiles vs lateral offset ---
ax_disp = fig.add_subplot(gs[2, 4:9])

lat_offsets_vox = [0, 5, 10, 15]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
for i, dy_vox in enumerate(lat_offsets_vox):
    iy = iy0 + dy_vox
    if 0 <= iy < ny_ds:
        tr = np.abs(snapshots_xy[:, ix0, iy])
        dist_mm = dy_vox * dy_ds_mm
        if dy_vox == 0:
            lbl = "Focus (0 mm)"
        else:
            lbl = f"+{dist_mm:.1f} mm"
        ax_disp.plot(t_axis_ms, tr * 1e6, label=lbl, color=colors[i], linewidth=1.5)
        print(f"  Lateral {lbl}: peak={tr.max() * 1e6:.2f} um")

ax_disp.set_xlabel("Time (ms)")
ax_disp.set_ylabel("|u| (\u03bcm)")
ax_disp.set_title("(h) Displacement \u2014 lateral offsets", fontsize=label_fontsize)
ax_disp.legend(fontsize=10)
ax_disp.set_xlim(0, t_axis_ms[-1])
ax_disp.grid(True, alpha=0.3)

# --- Save ---
out_path = out_dir / "fig_tips_composite.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
