"""Validate shear FDTD against QIBA US-SWS elastic phantom configurations.

Reproduces the standardized digital phantoms from:
  Palmeri ML et al., "Guidelines for Finite-Element Modeling of Acoustic
  Radiation Force-Induced Shear Wave Propagation in Tissue-Mimicking Media,"
  IEEE TUFFC, 64(1):78-92, Jan 2017.

Four elastic phantoms (G = 1, 2, 5, 10 kPa) are simulated with a Gaussian
approximation of the ARF body force matching the QIBA f/2.0 curvilinear
transducer configuration.  Group velocity is measured via time-to-peak at
multiple lateral offsets and compared to the analytical cs = sqrt(G/rho).

If QIBA reference displacement data (res_sim.mat from Zenodo DOI
10.5281/zenodo.3497594 or Duke DOI 10.7924/r4sj1f98c) is available, the
script also loads it and overlays the LS-DYNA displacement traces.

Usage:
    python validate_qiba.py                          # run all 4 elastic phantoms
    python validate_qiba.py --phantoms E3k E30k      # run subset
    python validate_qiba.py --qiba-dir ./qiba_data   # compare against LS-DYNA
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from shearwave import shear_fdtd_staggered

try:
    from scipy.io import loadmat

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# QIBA phantom definitions (Palmeri et al. 2017, Table I)
# ---------------------------------------------------------------------------
QIBA_PHANTOMS = {
    "E3k": {"E_Pa": 3.0e3, "G_Pa": 1.0e3, "nu": 0.495, "rho": 1000.0},
    "E6k": {"E_Pa": 6.0e3, "G_Pa": 2.0e3, "nu": 0.495, "rho": 1000.0},
    "E15k": {"E_Pa": 15.0e3, "G_Pa": 5.0e3, "nu": 0.495, "rho": 1000.0},
    "E30k": {"E_Pa": 30.0e3, "G_Pa": 10.0e3, "nu": 0.495, "rho": 1000.0},
}

# Acoustic parameters for ARF beam profile
F0_HZ = 3.0e6  # transducer centre frequency
C_TISSUE = 1540.0  # acoustic sound speed [m/s]
ALPHA_DB_CM_MHZ = 0.45  # attenuation
F_NUMBER = 2.0  # QIBA f/2.0 configuration
PUSH_DURATION_S = 167e-6  # 500 cycles at 3 MHz


def _gaussian_arf(nx, ny, nz, dx, focus_idx, sigma_ax, sigma_lat, sigma_elev):
    """3-D Gaussian body force following the QIBA GaussExc convention.

    Uses: A * exp( -((x-cx)/sx)^2 - ((y-cy)/sy)^2 - ((z-cz)/sz)^2 )

    This matches GaussExc.py from mlp6/fem (Palmeri et al.), which is the
    standard Gaussian excitation alternative to Field II for QIBA digital
    phantom simulations.  The sigma parameters are the 1/e amplitude widths.
    """
    xg = np.arange(nx, dtype=np.float32) - focus_idx
    yg = np.arange(ny, dtype=np.float32) - ny // 2
    zg = np.arange(nz, dtype=np.float32) - nz // 2
    gx = np.exp(-((xg / sigma_ax) ** 2))
    gy = np.exp(-((yg / sigma_lat) ** 2))
    gz = np.exp(-((zg / sigma_elev) ** 2))
    bx = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
    bx /= bx.max()
    return bx


def _push_envelope(n_steps, dT, duration=PUSH_DURATION_S):
    """Trapezoidal push-on / push-off temporal envelope."""
    t = np.arange(n_steps, dtype=np.float64) * dT
    ramp = 5.0 / F0_HZ  # 5-cycle ramp
    env = np.zeros(n_steps, dtype=np.float32)
    on = t < duration
    ramp_up = t < ramp
    ramp_down = (t >= duration - ramp) & on
    env[on] = 1.0
    env[ramp_up] = t[ramp_up] / ramp
    env[ramp_down] = (duration - t[ramp_down]) / ramp
    return env


def _measure_group_velocity(traces, distances, t_vec, cs_expected):
    """Group velocity via cross-correlation between adjacent sensor pairs.

    More robust than TTP for broadband ARF-driven shear waves whose
    waveform evolves with distance due to diffraction.

    Returns (c_group, peak_times, vel_traces).
    """
    dT = t_vec[1] - t_vec[0]
    vel_traces = np.diff(traces, axis=1) / dT
    t_vel = t_vec[1:]
    n_sensors = traces.shape[0]

    # Cross-correlation of velocity traces between adjacent pairs.
    # Using velocity (dU/dt) instead of displacement removes the quasi-static
    # push component and isolates the propagating shear wave.
    # Skip pairs where either sensor is < 4 mm from push axis.
    min_dist = 4e-3
    pair_speeds = []
    for i in range(n_sensors - 1):
        if distances[i] < min_dist:
            continue
        d_pair = distances[i + 1] - distances[i]
        if d_pair <= 0:
            continue
        t_arrive = distances[i] / cs_expected
        t_end = distances[i + 1] / cs_expected * 2.0
        idx_start = max(0, int(0.3 * t_arrive / dT))
        idx_end = min(vel_traces.shape[1], int(t_end / dT))
        if idx_end - idx_start < 10:
            continue
        s1 = vel_traces[i, idx_start:idx_end]
        s2 = vel_traces[i + 1, idx_start:idx_end]
        cc = np.correlate(s2, s1, mode="full")
        lags = np.arange(len(cc)) - (len(s1) - 1)
        pos = lags > 0
        if not pos.any():
            continue
        peak_lag = lags[pos][np.argmax(cc[pos])]
        dt_pair = peak_lag * dT
        if dt_pair > 0:
            c_pair = d_pair / dt_pair
            if 0.3 * cs_expected < c_pair < 3.0 * cs_expected:
                pair_speeds.append(c_pair)

    if len(pair_speeds) >= 2:
        c_group = float(np.median(pair_speeds))
    else:
        c_group = float("nan")

    # TTP for plotting
    peak_times = np.full(n_sensors, np.nan)
    for i in range(n_sensors):
        t_start = 0.4 * distances[i] / cs_expected
        idx0 = max(0, int(t_start / dT))
        seg = np.abs(vel_traces[i, idx0:])
        if seg.size == 0:
            continue
        peak_times[i] = t_vel[idx0 + int(np.argmax(seg))]

    # Fallback if cross-correlation failed
    if np.isnan(c_group):
        fit_mask = (
            np.isfinite(peak_times) & (peak_times > 0) & (distances >= 2e-3) & (distances <= 8e-3)
        )
        if np.count_nonzero(fit_mask) >= 2:
            coeffs = np.polyfit(peak_times[fit_mask], distances[fit_mask], 1)
            c_group = float(coeffs[0])

    return c_group, peak_times, vel_traces


def run_phantom(
    phantom_name,
    *,
    dx=2.5e-4,
    lateral_extent=50e-3,
    axial_extent=60e-3,
    elev_extent=50e-3,
    CFL=0.2,
    save_dir=None,
    qiba_ref_path=None,
    make_plots=True,
):
    """Run one QIBA elastic phantom and return results dict."""
    ph = QIBA_PHANTOMS[phantom_name]
    rho = ph["rho"]
    mu = ph["G_Pa"]
    cs = np.sqrt(mu / rho)

    # Grid dimensions — lateral extent large enough for 15mm propagation
    # without boundary contamination
    nx = int(axial_extent / dx)
    ny = int(lateral_extent / dx)
    nz = int(elev_extent / dx)
    dT = CFL * dx / (cs * np.sqrt(3.0))

    # Simulation time: push + propagation to 15mm + margin
    T_prop = 15e-3 / cs  # time for wave to reach 15mm
    Tmax = T_prop + PUSH_DURATION_S + 2e-3
    n_steps = int(np.ceil(Tmax / dT))
    t_vec = np.arange(n_steps, dtype=np.float64) * dT

    print(f"\n{'=' * 60}")
    print(f"QIBA Phantom {phantom_name}: G={mu:.0f} Pa, cs={cs:.3f} m/s")
    print(f"  Grid {nx}x{ny}x{nz}, dx={dx * 1e3:.2f} mm")
    print(f"  dT={dT * 1e6:.2f} us, {n_steps} steps, Tmax={Tmax * 1e3:.1f} ms")
    print(f"{'=' * 60}")

    # ARF body force — Gaussian excitation matching GaussExc.py convention
    # (Palmeri et al., mlp6/fem). Sigma = 1/e amplitude width.
    # FWHM = 2 * sigma * sqrt(ln 2); sigma = FWHM / (2 * sqrt(ln 2)).
    lam_ac = C_TISSUE / F0_HZ  # acoustic wavelength [m]
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2.0)))  # 0.6006
    sigma_lat = lam_ac * F_NUMBER * fwhm_to_sigma / dx  # grid pts
    sigma_ax = 3.5 * lam_ac * F_NUMBER**2 * fwhm_to_sigma / dx
    # Elevation: lens focus at 50 mm, element height 14 mm → f/#_elev ≈ 3.57
    fnum_elev = 50e-3 / 14e-3
    sigma_elev = lam_ac * fnum_elev * fwhm_to_sigma / dx
    focus_idx = nx // 2

    bx = _gaussian_arf(nx, ny, nz, dx, focus_idx, sigma_ax, sigma_lat, sigma_elev)
    env_t = _push_envelope(n_steps, dT)

    # Sensors: lateral offsets 1–14 mm from push axis at focal depth
    offsets_mm = np.arange(1.0, 15.0, 1.0)
    offsets_idx = (offsets_mm * 1e-3 / dx).astype(int)
    sensor_y = ny // 2 + offsets_idx
    keep = sensor_y < ny - 2
    offsets_mm, offsets_idx, sensor_y = offsets_mm[keep], offsets_idx[keep], sensor_y[keep]
    distances = offsets_mm * 1e-3
    mid_z = nz // 2

    traces = np.zeros((len(sensor_y), n_steps), dtype=np.float32)

    def callback(it, u):
        traces[:, it] = u[focus_idx, sensor_y, mid_z, 0]  # axial disp

    opts = {
        "vel_damp": 0.0005,  # minimal damping for late-time stability
        "eta": 0.0,  # pure elastic
        "env_t": env_t,
        "verbose": True,
        "progress_every": max(1, n_steps // 6),
    }

    shear_fdtd_staggered(
        bx, None, None, rho, mu, dx, dx, dx, dT, n_steps, opts=opts, callback=callback
    )

    # Group velocity (measured from velocity traces = dU/dt)
    c_group, peak_times, vel_traces = _measure_group_velocity(traces, distances, t_vec, cs)
    err_pct = abs(c_group - cs) / cs * 100.0

    traces_um = traces * 1e6  # convert to micrometers (QIBA convention)

    print(f"\n  Analytical cs  = {cs:.4f} m/s")
    print(f"  Measured  cs   = {c_group:.4f} m/s")
    print(f"  Error          = {err_pct:.2f}%")

    # ---- Optional: load QIBA LS-DYNA reference data ----
    qiba_traces = None
    qiba_t = None
    qiba_lat = None
    if qiba_ref_path is not None:
        qiba_traces, qiba_t, qiba_lat = _load_qiba_reference(qiba_ref_path, focus_idx, dx)

    result = {
        "phantom": phantom_name,
        "G_Pa": mu,
        "cs_analytical": cs,
        "cs_measured": c_group,
        "error_pct": err_pct,
        "traces_um": traces_um,
        "vel_traces": vel_traces,
        "t_s": t_vec,
        "sensor_mm": offsets_mm,
        "sensor_distances": distances,
        "peak_times": peak_times,
        "qiba_traces_um": qiba_traces,
        "qiba_t_s": qiba_t,
        "qiba_lat_mm": qiba_lat,
    }

    if make_plots and save_dir is not None:
        _plot_results(result, save_dir)

    return result


def _load_qiba_reference(mat_path, focus_idx, dx):
    """Load QIBA res_sim.mat and extract displacement traces.

    Expected variables:
      arfidata (lat, axial, time) in micrometers
      lat      1-D in mm
      axial    1-D in mm
      t        1-D in seconds
    """
    if not _HAS_SCIPY:
        print("  scipy not available — skipping QIBA reference load")
        return None, None, None
    mat_path = Path(mat_path)
    if not mat_path.exists():
        print(f"  QIBA reference not found: {mat_path}")
        return None, None, None
    data = loadmat(str(mat_path))
    arfi = data["arfidata"]  # (lat, axial, time)
    lat = data["lat"].ravel()  # mm
    axial = data["axial"].ravel()  # mm
    t = data["t"].ravel()  # s

    # Find axial index closest to our focal depth
    focus_mm = focus_idx * dx * 1e3
    ax_idx = int(np.argmin(np.abs(axial - focus_mm)))

    # Extract traces at focal depth for each lateral position
    qiba_traces = arfi[:, ax_idx, :]  # (n_lat, n_time) in um
    print(f"  Loaded QIBA reference: {arfi.shape}, focal depth={axial[ax_idx]:.1f} mm")
    return qiba_traces, t, lat


def _plot_results(result, save_dir):
    """Generate comparison plots for one phantom."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = result["phantom"]

    traces = result["traces_um"]
    vel_traces = result["vel_traces"]
    t_ms = result["t_s"] * 1e3
    t_vel_ms = t_ms[1:]
    offsets = result["sensor_mm"]
    cs_a = result["cs_analytical"]
    cs_m = result["cs_measured"]

    n_show = min(6, traces.shape[0])

    # --- Displacement + velocity traces ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i in range(n_show):
        ax1.plot(t_ms, traces[i], label=f"{offsets[i]:.0f} mm")
        ax2.plot(t_vel_ms, vel_traces[i], label=f"{offsets[i]:.0f} mm")
    ax1.set_ylabel("Axial displacement (um)")
    ax1.set_title(
        f"QIBA {name}: Fullwave shear FDTD  (cs={cs_m:.3f} m/s, analytical={cs_a:.3f} m/s)"
    )
    ax1.legend(title="Lateral offset", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Velocity dU/dt (m/s)")
    ax2.set_title("Velocity traces (used for TTP measurement)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"qiba_{name}_traces.png", dpi=150)
    plt.close(fig)

    # --- Time-to-peak vs distance (group velocity fit) ---
    pt = result["peak_times"]
    distances = result["sensor_distances"]
    valid = np.isfinite(pt) & (pt > 0)
    fit_mask = valid & (distances >= 3e-3) & (distances <= 12e-3)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pt[valid] * 1e3, offsets[valid], "ko", ms=5, alpha=0.4, label="All TTP")
    if np.count_nonzero(fit_mask) >= 1:
        ax.plot(pt[fit_mask] * 1e3, offsets[fit_mask], "ko", ms=7, label="Fit range (3-12 mm)")
    t_max = np.nanmax(pt[valid]) * 1.1 if np.any(valid) else 1.0
    t_fit = np.linspace(0, t_max, 50)
    ax.plot(t_fit * 1e3, cs_m * t_fit * 1e3, "r-", lw=1.5, label=f"Fit: {cs_m:.3f} m/s")
    ax.plot(t_fit * 1e3, cs_a * t_fit * 1e3, "b--", lw=1, label=f"Analytical: {cs_a:.3f} m/s")
    ax.set_xlabel("Time-to-peak velocity (ms)")
    ax.set_ylabel("Lateral distance (mm)")
    ax.set_title(f"QIBA {name}: Group velocity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"qiba_{name}_group_velocity.png", dpi=150)
    plt.close(fig)

    # --- Overlay with QIBA LS-DYNA reference if available ---
    if result["qiba_traces_um"] is not None:
        _plot_comparison(result, save_dir)

    print(f"  Plots saved to {save_dir}")


def _plot_comparison(result, save_dir):
    """Overlay Fullwave and LS-DYNA displacement traces."""
    save_dir = Path(save_dir)
    name = result["phantom"]
    fw_traces = result["traces_um"]
    fw_t = result["t_s"] * 1e3
    fw_offsets = result["sensor_mm"]
    q_traces = result["qiba_traces_um"]
    q_t = result["qiba_t_s"] * 1e3
    q_lat = result["qiba_lat_mm"]

    # Match lateral positions (QIBA sensor spacing is 0.1 mm)
    compare_mm = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

    for ax, target_mm in zip(axes.ravel(), compare_mm):
        # Fullwave trace
        fw_idx = int(np.argmin(np.abs(fw_offsets - target_mm)))
        ax.plot(fw_t, fw_traces[fw_idx], "b-", lw=1.2, label="Fullwave FDTD")

        # QIBA trace
        q_idx = int(np.argmin(np.abs(q_lat - target_mm)))
        ax.plot(q_t, q_traces[q_idx], "r--", lw=1.2, label="LS-DYNA (QIBA)")

        ax.set_title(f"{target_mm:.0f} mm lateral")
        ax.grid(True, alpha=0.3)
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Axial disp (um)")

    fig.suptitle(
        f"QIBA {name}: Fullwave vs LS-DYNA reference",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_dir / f"qiba_{name}_comparison.png", dpi=150)
    plt.close(fig)


def run_all(*, phantoms=None, save_dir=None, qiba_dir=None, **kwargs):
    """Run all (or selected) QIBA elastic phantoms and print summary."""
    if phantoms is None:
        phantoms = list(QIBA_PHANTOMS.keys())
    if save_dir is None:
        save_dir = Path(__file__).parent / "qiba_run"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in phantoms:
        qiba_ref = None
        if qiba_dir is not None:
            # Convention: res_sim files named by Young's modulus, e.g.
            # Foc30mmF2p0_E3k_T167/res_sim.mat
            candidates = list(Path(qiba_dir).rglob(f"*{name}*/**/res_sim.mat"))
            if candidates:
                qiba_ref = candidates[0]
                print(f"  Found QIBA reference: {qiba_ref}")
        res = run_phantom(
            name,
            save_dir=save_dir / name,
            qiba_ref_path=qiba_ref,
            **kwargs,
        )
        results.append(res)

    # Summary table
    print(f"\n{'=' * 60}")
    print("QIBA Elastic Phantom Validation Summary")
    print(f"{'=' * 60}")
    print(f"{'Phantom':<10} {'G (kPa)':>8} {'cs_theory':>10} {'cs_meas':>10} {'Error':>8}")
    print(f"{'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")
    for r in results:
        print(
            f"{r['phantom']:<10} {r['G_Pa'] / 1e3:>8.1f} "
            f"{r['cs_analytical']:>10.4f} {r['cs_measured']:>10.4f} "
            f"{r['error_pct']:>7.2f}%"
        )
    print(f"{'=' * 60}")

    # Save summary
    np.savez(
        save_dir / "qiba_summary.npz",
        phantoms=[r["phantom"] for r in results],
        G_Pa=[r["G_Pa"] for r in results],
        cs_analytical=[r["cs_analytical"] for r in results],
        cs_measured=[r["cs_measured"] for r in results],
        error_pct=[r["error_pct"] for r in results],
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate shear FDTD against QIBA elastic phantoms."
    )
    parser.add_argument(
        "--phantoms",
        nargs="+",
        choices=list(QIBA_PHANTOMS.keys()),
        default=None,
        help="Which phantoms to run (default: all four).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory (default: validation/qiba_run/).",
    )
    parser.add_argument(
        "--qiba-dir",
        type=str,
        default=None,
        help="Path to downloaded QIBA reference data (contains res_sim.mat files).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=2.5e-4,
        help="Grid spacing in metres (default: 0.25 mm).",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.2,
        help="CFL number (default: 0.2).",
    )
    args = parser.parse_args()

    run_all(
        phantoms=args.phantoms,
        save_dir=args.save_dir,
        qiba_dir=args.qiba_dir,
        dx=args.dx,
        CFL=args.cfl,
    )


if __name__ == "__main__":
    main()
