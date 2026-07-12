#!/usr/bin/env python3
"""Generate QIBA validation multi-panel figure for PMB manuscript.

Layout (2 rows × 3 cols):
  (a) ARF body force (zoomed, interpolated)
  (b) Space-time kymograph for representative phantom (E15k)
  (c) Measured vs analytical wavespeed for all 4 phantoms
"""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import zoom

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---
# repo layout: shearwave/paper/generate_fig_qiba.py -> shearwave/validation/qiba_run
data_dir = Path(__file__).parent.parent / "validation" / "qiba_run"
out_dir = Path(__file__).parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# --- QIBA phantom definitions ---
PHANTOMS = {
    "E3k": {"G_Pa": 1.0e3, "label": "1 kPa"},
    "E6k": {"G_Pa": 2.0e3, "label": "2 kPa"},
    "E15k": {"G_Pa": 5.0e3, "label": "5 kPa"},
    "E30k": {"G_Pa": 10.0e3, "label": "10 kPa"},
}
RHO = 1000.0



def _build_gaussian_arf(dx):
    """Build the GaussExc Gaussian ARF matching validate_qiba.py parameters."""
    F0 = 3.0e6
    C_TISSUE = 1540.0
    F_NUMBER = 2.0
    nx, ny, nz = 240, 200, 200  # 60x50x50 mm

    lam_ac = C_TISSUE / F0
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2.0)))
    sigma_lat = lam_ac * F_NUMBER * fwhm_to_sigma / dx
    sigma_ax = 3.5 * lam_ac * F_NUMBER**2 * fwhm_to_sigma / dx
    fnum_elev = 50e-3 / 14e-3
    sigma_elev = lam_ac * fnum_elev * fwhm_to_sigma / dx
    focus_idx = nx // 2

    xg = np.arange(nx, dtype=np.float32) - focus_idx
    yg = np.arange(ny, dtype=np.float32) - ny // 2
    zg = np.arange(nz, dtype=np.float32) - nz // 2
    gx = np.exp(-((xg / sigma_ax) ** 2))
    gy = np.exp(-((yg / sigma_lat) ** 2))
    gz = np.exp(-((zg / sigma_elev) ** 2))
    bx = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
    bx /= bx.max()
    return bx


def main():
    dx = 2.5e-4  # 0.25 mm grid spacing
    b0 = _build_gaussian_arf(dx)

    # Load summary
    summary = np.load(data_dir / "qiba_summary.npz", allow_pickle=True)
    phantom_names = list(summary["phantoms"])
    cs_analytical = np.array(summary["cs_analytical"], dtype=float)
    cs_measured = np.array(summary["cs_measured"], dtype=float)
    error_pct = np.array(summary["error_pct"], dtype=float)
    G_Pa = np.array(summary["G_Pa"], dtype=float)

    # Use E15k as representative (mid-range stiffness)
    kymo_name = "E15k"

    # Run shear sim to capture traces at many lateral offsets
    kymo_result = _run_kymograph_sim(kymo_name, b0, dx)

    # --- Create figure: single row, 3 panels ---
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(14, 4.2),
        gridspec_kw={"width_ratios": [1, 1.3, 1], "wspace": 0.45},
    )

    _plot_arf(ax_a, b0, dx)
    _plot_kymograph(ax_b, kymo_result, kymo_name)
    _plot_wavespeed_comparison(ax_c, cs_analytical, cs_measured, error_pct, G_Pa)

    # Panel labels
    for ax, label in zip([ax_a, ax_b, ax_c], ["(a)", "(b)", "(c)"]):
        ax.text(
            -0.12, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom"
        )

    fig.savefig(out_dir / "fig_qiba_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'fig_qiba_validation.png'}")


def _plot_arf(ax, b0, dx):
    """Plot ARF body force field (mid-elevation slice) in physical units.

    Depth (x) on vertical axis with 0 at top, lateral (y) on horizontal.
    Computes actual ARF magnitude from QIBA acoustic parameters.
    """
    if b0.ndim == 3:
        b0_slice = b0[:, :, b0.shape[2] // 2]
    else:
        b0_slice = b0

    # Convert normalized Gaussian to physical ARF [N/m³]
    # b = 2 * alpha_Np * I_sppa / c, with I_sppa from QIBA Isppa = 1000 W/cm²
    f0_mhz = 3.0
    alpha_db_cm_mhz = 0.45
    alpha_np_per_m = alpha_db_cm_mhz * f0_mhz / 8.686 * 100.0  # Np/m
    c = 1540.0
    isppa_w_m2 = 1000.0 * 1e4  # 1000 W/cm² → W/m²
    arf_peak = 2.0 * alpha_np_per_m * isppa_w_m2 / c  # N/m³
    b0_phys = b0_slice * arf_peak

    nx, ny = b0_phys.shape
    depth_mm = np.array([0, nx * dx * 1e3])
    lat_mm = np.array([-ny / 2, ny / 2]) * dx * 1e3

    b0_interp = zoom(b0_phys, 4, order=3)

    im = ax.imshow(
        b0_interp,
        origin="upper",
        extent=[lat_mm[0], lat_mm[1], depth_mm[1], depth_mm[0]],
        cmap="hot",
        aspect="equal",
        interpolation="bilinear",
    )
    ax.set_ylabel("Depth (mm)", fontsize=8)
    ax.set_xlabel("Lateral (mm)", fontsize=8)
    ax.set_title("Radiation force", fontsize=9)
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("N/m$^3$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()


def _run_kymograph_sim(phantom_name, b0, dx):
    """Run shear FDTD with dense lateral sampling for kymograph."""
    from shearwave import shear_fdtd_staggered

    ph = PHANTOMS[phantom_name]
    mu = ph["G_Pa"]
    cs = np.sqrt(mu / RHO)
    nx_s, ny_s, nz_s = b0.shape

    CFL = 0.2
    dT = CFL * dx / (cs * np.sqrt(3.0))
    PUSH_DURATION = 167e-6
    T_prop = 15e-3 / cs
    Tmax = T_prop + PUSH_DURATION + 2e-3
    n_steps = int(np.ceil(Tmax / dT))
    t_vec = np.arange(n_steps, dtype=np.float64) * dT

    t = np.arange(n_steps, dtype=np.float64) * dT
    env_t = np.where(t <= PUSH_DURATION, 1.0, 0.0).astype(np.float32)

    # Dense sensors: every grid point from 0 to 15mm lateral
    arf_peak = np.unravel_index(np.argmax(b0), b0.shape)
    max_offset = int(15e-3 / dx)
    sensor_y = np.arange(arf_peak[1], min(arf_peak[1] + max_offset, ny_s - 2))
    mid_z = nz_s // 2

    # Subsample time for kymograph storage
    save_every = max(1, n_steps // 500)
    kymo_times = []
    kymo_frames = []

    def callback(it, u):
        if it % save_every == 0:
            kymo_times.append(t_vec[it])
            kymo_frames.append(u[arf_peak[0], sensor_y, mid_z, 0].copy())

    opts = {
        "vel_damp": 0.0005,
        "eta": 0.0,
        "env_t": env_t,
        "verbose": False,
    }

    print(f"Running kymograph sim for {phantom_name}...")
    shear_fdtd_staggered(
        bx=b0,
        by=None,
        bz=None,
        rho=RHO,
        mu=mu,
        dX=dx,
        dY=dx,
        dZ=dx,
        dT=dT,
        n_steps=n_steps,
        opts=opts,
        callback=callback,
    )

    kymo = np.array(kymo_frames)  # (n_times, n_lateral)
    t_kymo = np.array(kymo_times)
    lat_mm = (sensor_y - arf_peak[1]) * dx * 1e3

    return {
        "kymo": kymo,
        "t_ms": t_kymo * 1e3,
        "lat_mm": lat_mm,
        "cs": cs,
    }


def _plot_kymograph(ax, kymo_result, phantom_name):
    """Plot space-time kymograph of shear wave propagation."""
    kymo = kymo_result["kymo"] * 1e6  # to micrometers
    t_ms = kymo_result["t_ms"]
    lat_mm = kymo_result["lat_mm"]
    cs = kymo_result["cs"]

    vmax = np.abs(kymo).max() * 0.6
    ax.imshow(
        kymo.T,
        origin="lower",
        extent=[t_ms[0], t_ms[-1], lat_mm[0], lat_mm[-1]],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Overlay analytical wavespeed line
    t_line = np.array([0, t_ms[-1]])
    ax.plot(t_line, cs * t_line, "k--", lw=1.0, alpha=0.7, label=f"$c_s = {cs:.2f}$ m/s")

    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Lateral distance (mm)", fontsize=8)
    ph = PHANTOMS[phantom_name]
    ax.set_title(f"Shear wave kymograph ($G = {ph['label']}$)", fontsize=9)
    ax.set_ylim([0, 14])
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)


def _plot_wavespeed_comparison(ax, cs_analytical, cs_measured, error_pct, G_Pa):
    """Plot measured vs analytical wavespeed for all 4 phantoms."""
    # Identity line
    c_range = np.array([0, cs_analytical.max() * 1.15])
    ax.plot(c_range, c_range, "k-", lw=0.8, alpha=0.5, label="Identity")

    # Data points
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cs_analytical)))
    for i in range(len(cs_analytical)):
        label = f"$G = {G_Pa[i] / 1e3:.0f}$ kPa ({error_pct[i]:.3f}%)"
        ax.plot(
            cs_analytical[i],
            cs_measured[i],
            "o",
            color=colors[i],
            ms=9,
            mec="k",
            mew=0.5,
            label=label,
            zorder=5,
        )

    ax.set_xlabel("Analytical SWS (m/s)", fontsize=8)
    ax.set_ylabel("Measured SWS (m/s)", fontsize=8)
    ax.set_xlim(c_range)
    ax.set_ylim(c_range)
    ax.set_aspect("equal")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)


if __name__ == "__main__":
    main()
