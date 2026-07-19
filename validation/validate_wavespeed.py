"""Non-viscous shear wavespeed recovery validation.

Launches a 3D shear FDTD simulation with a Gaussian impulse (via curl of a
vector potential) and measures wave arrival times at multiple sensor offsets.
The recovered wave speed is compared to the analytical value cs = sqrt(mu/rho).

Reference: Section 4.1 of the shear FDTD physics derivation document.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from shearwave import hann1d, shear_fdtd


def run_validation(
    *,
    nX=180,
    nY=60,
    nZ=180,
    dX=2.0e-4,
    dY=None,
    dZ=None,
    rho=1000.0,
    cs_target=2.0,
    CFL=0.2,
    Tmax=6.0e-3,
    amplitude=3e-5,
    sensor_offsets=None,
    eta=0.5,
    sigma_xy=None,
    vel_damp=None,
    save_dir=None,
    make_plots=True,
    make_movie=False,
    movie_every=10,
    movie_dir=None,
):
    """Run wavespeed recovery validation and return results dict.

    Args:
        nX: Grid points in x (depth).
        nY: Grid points in y (lateral).
        nZ: Grid points in z (elevational).
        dX: Grid spacing in x (m).
        dY: Grid spacing in y (m), defaults to dX.
        dZ: Grid spacing in z (m), defaults to dX.
        rho: Density (kg/m^3).
        cs_target: Target shear wave speed (m/s).
        CFL: CFL number for time step computation.
        Tmax: Total simulation time (s).
        amplitude: Amplitude of the initial vector potential.
        sensor_offsets: Array of sensor offsets (grid points) from center.
        eta: Viscosity (Pa*s).
        sigma_xy: Gaussian source width in grid points. Defaults based on grid size.
        vel_damp: Velocity damping coefficient. Defaults based on grid size.
        save_dir: Directory for output files.
        make_plots: Whether to generate plots.
        make_movie: Whether to save movie frames.
        movie_every: Frame save interval for movie.
        movie_dir: Directory for movie frames.

    Returns:
        Dictionary with wavespeed results including sensor traces,
        arrival times, and recovered speeds.

    """
    dY = dX if dY is None else dY
    dZ = dX if dZ is None else dZ

    mu = rho * cs_target**2
    dT = CFL * min(dX, dY, dZ) / (cs_target * np.sqrt(3.0))
    n_steps = int(np.ceil(Tmax / dT))
    t_vec = np.arange(n_steps, dtype=np.float64) * dT

    # Spatial Gaussian body force (via vector potential)
    xg, yg, zg = np.meshgrid(
        np.arange(nX) - nX // 2,
        np.arange(nY) - nY // 2,
        np.arange(nZ) - nZ // 2,
        indexing="ij",
    )
    if sigma_xy is None:
        sigma_xy = 5.0 if max(nX, nZ) > 150 else 3.0
    sigma_z = sigma_xy
    Gxyz = np.exp(
        -0.5 * ((xg / sigma_xy) ** 2 + (yg / sigma_xy) ** 2 + (zg / sigma_z) ** 2)
    ).astype(np.float32)
    Gxyz /= np.max(Gxyz)
    window = hann1d(nX)[:, None, None] * hann1d(nY)[None, :, None] * hann1d(nZ)[None, None, :]
    Gxyz *= window

    psi = amplitude * Gxyz
    dpsi_dx, _, dpsi_dz = np.gradient(psi, dX, dY, dZ, edge_order=2)
    vx0 = -dpsi_dz.astype(np.float32)
    vy0 = np.zeros_like(vx0)
    vz0 = dpsi_dx.astype(np.float32)
    initial_v = np.stack([vx0, vy0, vz0], axis=-1)

    # Zero forcing — use env_t to avoid allocating a large 4D array.
    bz = np.zeros((nX, nY, nZ), dtype=np.float32)
    env_t = np.zeros(n_steps, dtype=np.float32)
    if sensor_offsets is None:
        if max(nX, nZ) > 150:
            sensor_offsets = np.array([18, 30, 42, 54, 66, 78], dtype=int)
        else:
            sensor_offsets = np.array([12, 18, 24, 30, 36, 42], dtype=int)
    else:
        sensor_offsets = np.asarray(sensor_offsets, dtype=int)
    sensor_idx = nX // 2 + sensor_offsets
    sensor_distances = sensor_offsets * dX
    mid_y = nY // 2
    mid_z = nZ // 2
    sensor_traces = np.zeros((len(sensor_idx), n_steps), dtype=np.float32)
    movie_frames = []
    if make_movie:
        movie_dir_path = Path(movie_dir) if movie_dir else Path.cwd() / "movies"
        movie_dir_path.mkdir(parents=True, exist_ok=True)

    def record_callback(it, u_center):
        uz = u_center[..., 2]
        sensor_traces[:, it] = uz[sensor_idx, mid_y, mid_z]
        if make_movie and (it % movie_every == 0 or it == n_steps - 1):
            movie_frames.append((it, uz.copy()))

    opts = {
        "vel_damp": vel_damp if vel_damp is not None else (0.001 if max(nX, nZ) > 150 else 0.002),
        "poisson_tol": 5e-5,
        "poisson_max_iters": 400,
        "verbose": True,
        "progress_every": max(1, n_steps // 6),
        "initial_v": initial_v,
        "eta": eta,
        "env_t": env_t,
    }

    print("Running collocated shear FDTD impulse response...")
    shear_fdtd(
        None,
        None,
        bz,
        rho,
        mu,
        dX,
        dY,
        dZ,
        dT,
        n_steps,
        opts=opts,
        callback=record_callback,
    )

    vel_traces = np.diff(sensor_traces, axis=1) / dT
    arrival_idx = np.full(len(sensor_idx), np.nan, dtype=np.float64)
    for i, dist in enumerate(sensor_distances):
        tail = vel_traces[i]
        if tail.size == 0:
            continue
        start_time = 0.5 * dist / cs_target
        idx_start = int(np.searchsorted(t_vec[1:], start_time))
        if idx_start >= tail.size:
            continue
        segment = tail[idx_start:]
        if segment.size == 0:
            continue
        peak_idx = int(np.argmax(np.abs(segment)))
        arrival_idx[i] = idx_start + peak_idx

    arrival_times = (arrival_idx + 1) * dT
    valid = np.isfinite(arrival_idx)
    speeds = np.full_like(arrival_times, np.nan)
    speeds[valid] = sensor_distances[valid] / arrival_times[valid]

    mean_speed = float(np.nanmean(speeds))
    if np.count_nonzero(valid) >= 2:
        coeffs = np.polyfit(arrival_times[valid], sensor_distances[valid], 1)
        c_fit = float(coeffs[0])
    else:
        c_fit = float("nan")

    result = {
        "dT": dT,
        "t_vec": t_vec,
        "sensor_offsets": sensor_offsets,
        "sensor_distances": sensor_distances,
        "arrival_idx": arrival_idx,
        "arrival_times": arrival_times,
        "speeds": speeds,
        "mean_speed": mean_speed,
        "c_fit": c_fit,
        "traces": sensor_traces,
        "vel_traces": vel_traces,
    }

    if make_movie and movie_frames:
        movie_dir_path = Path(movie_dir) if movie_dir else Path.cwd() / "movies"
        movie_dir_path.mkdir(parents=True, exist_ok=True)
        x_mm = (np.arange(nX) - nX // 2) * dX * 1e3
        y_mm = (np.arange(nY) - nY // 2) * dY * 1e3
        z_mm = (np.arange(nZ) - nZ // 2) * dZ * 1e3
        for it, uz in movie_frames:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"right": 0.88})
            xy_slice = uz[:, :, mid_z].T
            xz_slice = uz[:, mid_y, :].T
            yz_slice = uz[nX // 2, :, :].T
            vmax_xz = max(np.max(np.abs(xy_slice)), np.max(np.abs(xz_slice)), 1e-30)
            im_kw = {"origin": "lower", "cmap": "RdBu", "vmin": -vmax_xz, "vmax": vmax_xz}
            axes[0].imshow(
                xy_slice,
                extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
                **im_kw,
            )
            axes[0].set_title(f"XY slice (step {it})")
            axes[0].set_xlabel("x (mm)")
            axes[0].set_ylabel("y (mm)")
            axes[0].set_aspect("equal")
            cs1 = axes[1].imshow(
                xz_slice,
                extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]],
                **im_kw,
            )
            axes[1].set_title("XZ slice")
            axes[1].set_xlabel("x (mm)")
            axes[1].set_ylabel("z (mm)")
            axes[1].set_aspect("equal")
            # YZ slice at x=0: u_z ≈ 0 by symmetry (curl source drives
            # motion in the x-z plane, so the center x-plane is a nodal
            # surface for u_z).
            axes[2].imshow(
                yz_slice,
                extent=[y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]],
                **im_kw,
            )
            axes[2].set_title("YZ slice ($u_z \\approx 0$ by symmetry)")
            axes[2].set_xlabel("y (mm)")
            axes[2].set_ylabel("z (mm)")
            axes[2].set_aspect("equal")
            cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
            fig.colorbar(cs1, cax=cbar_ax, label="u_z (m)")
            fig.subplots_adjust(wspace=0.35)
            frame_path = movie_dir_path / f"shear_slices_{it:05d}.png"
            fig.savefig(frame_path, dpi=150)
            plt.close(fig)

    if make_plots or save_dir is not None:
        fig_dir = Path(save_dir) if save_dir else Path.cwd()
        fig_dir.mkdir(parents=True, exist_ok=True)
        sensor_mm = sensor_distances * 1e3
        n_plot = min(4, len(sensor_mm))
        regime_label = "inviscid" if abs(float(eta)) < 1e-12 else "viscous"

        fig, ax = plt.subplots(figsize=(8, 5))
        for tr, mm in zip(sensor_traces[:n_plot], sensor_mm[:n_plot]):
            ax.plot(t_vec * 1e3, tr * 1e6, label=f"{mm:.1f} mm")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("u_z (um)")
        ax.set_title(f"Shear displacement ({regime_label})")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "shear_large_displacement.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for vt, mm in zip(vel_traces[:n_plot], sensor_mm[:n_plot]):
            ax.plot(t_vec[1:] * 1e3, vt, label=f"{mm:.1f} mm")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("du_z/dt (m/s)")
        ax.set_title(f"Velocity proxy ({regime_label})")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "shear_large_velocity.png", dpi=150)
        plt.close(fig)

        np.savez(fig_dir / "shear_wavespeed_largefield.npz", **result)

    return result


def main():
    res = run_validation(make_plots=True, save_dir=None)
    sensor_distances = res["sensor_distances"]
    arrival_times = res["arrival_times"]
    speeds = res["speeds"]
    mean_speed = res["mean_speed"]
    c_fit = res["c_fit"]
    cs_target = 2.0

    print("\nSensor summary (distance mm, arrival us, speed m/s):")
    for dist, t_arr, c in zip(sensor_distances, arrival_times, speeds):
        dist_mm = dist * 1e3
        t_us = t_arr * 1e6
        print(f"  {dist_mm:6.3f} mm : {t_us:7.3f} us : {c:6.3f} m/s")

    if np.count_nonzero(np.isfinite(speeds)) >= 1:
        print(f"\nMean estimated speed {mean_speed:.3f} m/s (target {cs_target:.3f} m/s)")
    if np.isfinite(c_fit):
        print(f"Linear regression speed {c_fit:.3f} m/s")


if __name__ == "__main__":
    main()
