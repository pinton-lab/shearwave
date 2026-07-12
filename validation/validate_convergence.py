"""Grid refinement convergence study for the shear FDTD solver.

Sweeps grid sizes, calls the wavespeed validation at each level, and computes
convergence order from the log-log slope of wavespeed error vs grid spacing.

Reference: Section 4.2 of the shear FDTD physics derivation document.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from validate_wavespeed import run_validation


def _safe_polyfit_speed(distances_m, arrival_times_s):
    valid = np.isfinite(arrival_times_s) & (arrival_times_s > 0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    coeffs = np.polyfit(arrival_times_s[valid], distances_m[valid], 1)
    return float(coeffs[0])


def _parse_levels(levels_text):
    vals = []
    for tok in levels_text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if len(vals) < 2:
        msg = "Need at least two grid levels."
        raise ValueError(msg)
    return sorted(set(vals))


def _load_existing_level(level_dir):
    npz_path = level_dir / "shear_wavespeed_largefield.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return {
        "dT": float(data["dT"]),
        "t_vec": np.array(data["t_vec"], dtype=float),
        "speeds": np.array(data["speeds"], dtype=float),
        "arrival_times": np.array(data["arrival_times"], dtype=float),
        "vel_traces": np.array(data["vel_traces"], dtype=float),
    }


def main():
    parser = argparse.ArgumentParser(description="Run 3D shear-wave convergence study.")
    parser.add_argument(
        "--levels",
        default="144,180,216",
        help="Comma-separated grid sizes, e.g. 120,144,180,216",
    )
    parser.add_argument(
        "--out-dir",
        default="convergence",
        help="Output folder for convergence results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-level NPZ results when available",
    )
    parser.add_argument(
        "--L", type=float, default=0.036, help="Physical domain size in each direction (m)"
    )
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--cs-target", type=float, default=2.0)
    parser.add_argument("--CFL", type=float, default=0.2)
    parser.add_argument("--Tmax", type=float, default=6.0e-3)
    parser.add_argument("--amplitude", type=float, default=3e-5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--sigma-xy",
        type=float,
        default=5.0,
        help="Gaussian source width in grid points (fixed across levels)",
    )
    parser.add_argument(
        "--vel-damp",
        type=float,
        default=0.001,
        help="Velocity damping coefficient (fixed across levels)",
    )
    parser.add_argument(
        "--sigma-phys",
        type=float,
        default=None,
        help="Physical Gaussian source width in metres, held constant across "
        "levels.  Overrides --sigma-xy; per level sigma_xy = sigma_phys / dX so "
        "that every grid launches the same continuum source (fair convergence "
        "test).  When omitted, the legacy grid-point --sigma-xy is used.",
    )
    parser.add_argument(
        "--damp-mode",
        choices=["fixed", "constant-total"],
        default="fixed",
        help="'fixed': vel-damp is applied per step at every level (legacy; "
        "total damping over Tmax then depends on the grid because n_steps ~ 1/dX). "
        "'constant-total': vel-damp is scaled per level so the total damping over "
        "Tmax is grid-independent.",
    )
    args = parser.parse_args()

    Lx = Ly = Lz = float(args.L)
    n_levels = _parse_levels(args.levels)

    rho = float(args.rho)
    cs_target = float(args.cs_target)
    CFL = float(args.CFL)
    Tmax = float(args.Tmax)
    amplitude = float(args.amplitude)
    eta = float(args.eta)
    sigma_xy = float(args.sigma_xy)
    vel_damp = float(args.vel_damp)
    sigma_phys = None if args.sigma_phys is None else float(args.sigma_phys)
    damp_mode = args.damp_mode

    def _n_steps(nn):
        dT = CFL * (Lx / nn) / (cs_target * np.sqrt(3.0))
        return int(np.ceil(Tmax / dT))

    n_steps_coarse = _n_steps(min(n_levels))

    # Keep physical sensor positions fixed across grids (first 4 points only)
    sensor_distances_m = np.array([3.6, 6.0, 8.4, 10.8], dtype=float) * 1e-3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    traces_by_n = {}
    times_by_n = {}

    regime = "inviscid" if abs(eta) < 1e-12 else "viscous"
    print(f"Running convergence study ({regime}, first 4 sensors)...", flush=True)
    for n in n_levels:
        dX = Lx / n
        dY = Ly / n
        dZ = Lz / n

        sensor_offsets = np.maximum(1, np.round(sensor_distances_m / dX).astype(int))
        level_dir = out_dir / f"n{n}"
        level_dir.mkdir(parents=True, exist_ok=True)

        # Physical source width (fair test) vs legacy grid-point width.
        level_sigma_xy = (sigma_phys / dX) if sigma_phys is not None else sigma_xy
        # Constant-total damping: scale per step so (1-vel_damp)^n_steps is
        # grid-independent (n_steps ~ 1/dX), else apply vel_damp per step.
        if damp_mode == "constant-total":
            level_vel_damp = 1.0 - (1.0 - vel_damp) ** (n_steps_coarse / _n_steps(n))
        else:
            level_vel_damp = vel_damp

        print(
            f"\nLevel n={n}: dX={dX:.3e} m, offsets={sensor_offsets.tolist()}, "
            f"sigma_xy={level_sigma_xy:.2f} cells, vel_damp={level_vel_damp:.5f}",
            flush=True,
        )
        res = None
        if args.resume:
            cached = _load_existing_level(level_dir)
            if cached is not None:
                print(f"  Reusing cached results from {level_dir}", flush=True)
                res = cached

        if res is None:
            res = run_validation(
                nX=n,
                nY=n,
                nZ=n,
                dX=dX,
                dY=dY,
                dZ=dZ,
                rho=rho,
                cs_target=cs_target,
                CFL=CFL,
                Tmax=Tmax,
                amplitude=amplitude,
                sensor_offsets=sensor_offsets,
                eta=eta,
                sigma_xy=level_sigma_xy,
                vel_damp=level_vel_damp,
                save_dir=level_dir,
                make_plots=False,
                make_movie=False,
            )

        speeds = np.array(res["speeds"], dtype=float)[: len(sensor_offsets)]
        arrival_times = np.array(res["arrival_times"], dtype=float)[: len(sensor_offsets)]
        trace = np.array(res["vel_traces"], dtype=float)[: len(sensor_offsets), :]
        t_vec = np.array(res["t_vec"], dtype=float)

        mean4 = float(np.nanmean(speeds))
        cfit4 = _safe_polyfit_speed(sensor_distances_m, arrival_times)
        err_mean = abs(mean4 - cs_target)
        err_fit = abs(cfit4 - cs_target) if np.isfinite(cfit4) else float("nan")

        rows.append(
            {
                "n": n,
                "dX": dX,
                "dT": float(res["dT"]),
                "mean4": mean4,
                "cfit4": cfit4,
                "err_mean": err_mean,
                "err_fit": err_fit,
                "speeds": speeds.tolist(),
                "arrival_times": arrival_times.tolist(),
            }
        )
        traces_by_n[n] = trace
        times_by_n[n] = t_vec[1:]  # vel_traces correspond to t_vec[1:]

    # Trace error vs finest grid (aggregate over first 4 sensors)
    n_ref = max(n_levels)
    tref = times_by_n[n_ref]
    vref = traces_by_n[n_ref]
    for row in rows:
        n = row["n"]
        if n == n_ref:
            row["trace_rel_l2_vs_ref"] = 0.0
            continue
        tc = times_by_n[n]
        vc = traces_by_n[n]
        rels = []
        for k in range(vref.shape[0]):
            vc_i = np.interp(tref, tc, vc[k])
            num = np.linalg.norm(vc_i - vref[k])
            den = np.linalg.norm(vref[k]) + 1e-12
            rels.append(float(num / den))
        row["trace_rel_l2_vs_ref"] = float(np.mean(rels))

    # Observed orders using successive levels
    rows_sorted = sorted(rows, key=lambda r: r["dX"], reverse=True)
    for i in range(len(rows_sorted) - 1):
        r1 = rows_sorted[i]
        r2 = rows_sorted[i + 1]
        if r1["err_fit"] > 0 and r2["err_fit"] > 0:
            p_fit = np.log(r1["err_fit"] / r2["err_fit"]) / np.log(r1["dX"] / r2["dX"])
        else:
            p_fit = float("nan")
        if r1["err_mean"] > 0 and r2["err_mean"] > 0:
            p_mean = np.log(r1["err_mean"] / r2["err_mean"]) / np.log(r1["dX"] / r2["dX"])
        else:
            p_mean = float("nan")
        r2["p_fit_from_prev"] = float(p_fit)
        r2["p_mean_from_prev"] = float(p_mean)

    # Save structured outputs
    with open(out_dir / "convergence_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)

    with open(out_dir / "convergence_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n",
                "dX_m",
                "dT_s",
                "mean4_mps",
                "cfit4_mps",
                "err_mean_mps",
                "err_fit_mps",
                "trace_rel_l2_vs_ref",
                "p_mean_from_prev",
                "p_fit_from_prev",
            ]
        )
        for r in rows_sorted:
            writer.writerow(
                [
                    r["n"],
                    r["dX"],
                    r["dT"],
                    r["mean4"],
                    r["cfit4"],
                    r["err_mean"],
                    r["err_fit"],
                    r.get("trace_rel_l2_vs_ref", float("nan")),
                    r.get("p_mean_from_prev", float("nan")),
                    r.get("p_fit_from_prev", float("nan")),
                ]
            )

    # Plot error vs dX (log-log)
    dx = np.array([r["dX"] for r in rows_sorted], dtype=float)
    efit = np.array([r["err_fit"] for r in rows_sorted], dtype=float)
    emean = np.array([r["err_mean"] for r in rows_sorted], dtype=float)
    etrace = np.array([r.get("trace_rel_l2_vs_ref", np.nan) for r in rows_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dx, np.maximum(efit, 1e-12), "o-", label=r"$|c_\mathrm{fit} - c_\mathrm{target}|$")
    ax.loglog(dx, np.maximum(emean, 1e-12), "s-", label=r"$|\bar{c} - c_\mathrm{target}|$")
    finite_trace = np.isfinite(etrace) & (etrace > 0)
    if np.any(finite_trace):
        ax.loglog(
            dx[finite_trace],
            etrace[finite_trace],
            "^-",
            label=r"trace relative $L_2$ vs finest",
        )
    ax.set_xlabel(r"Grid spacing $\Delta x$ (m)")
    ax.set_ylabel("Error metric")
    ax.set_title("Shear-wave convergence (inviscid)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_errors.png", dpi=150)
    plt.close(fig)

    # Mean4-only convergence plot with fitted slope
    valid_mean = np.isfinite(emean) & (emean > 0)
    if np.count_nonzero(valid_mean) >= 2:
        log_dx = np.log(dx[valid_mean])
        log_em = np.log(emean[valid_mean])
        coeffs = np.polyfit(log_dx, log_em, 1)
        slope = coeffs[0]
        r_vals = np.corrcoef(log_dx, log_em)[0, 1]
        r2 = r_vals**2

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(dx[valid_mean], emean[valid_mean], "s-", label=r"$|\bar{c} - c_\mathrm{target}|$")
        dx_fit = np.linspace(dx[valid_mean].min(), dx[valid_mean].max(), 50)
        ax.loglog(
            dx_fit,
            np.exp(np.polyval(coeffs, np.log(dx_fit))),
            "--",
            color="gray",
            label=f"slope = {slope:.3f} ($R^2$ = {r2:.4f})",
        )
        ax.set_xlabel(r"Grid spacing $\Delta x$ (m)")
        ax.set_ylabel(r"$|\bar{c} - c_\mathrm{target}|$ (m/s)")
        ax.set_title("Shear-wave wavespeed convergence (inviscid)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_mean_error_only.png", dpi=150)
        plt.close(fig)

    # Trace L2 convergence plot with fitted slope
    valid_trace = np.isfinite(etrace) & (etrace > 0)
    if np.count_nonzero(valid_trace) >= 2:
        log_dx_t = np.log(dx[valid_trace])
        log_et = np.log(etrace[valid_trace])
        coeffs_t = np.polyfit(log_dx_t, log_et, 1)
        slope_t = coeffs_t[0]
        r_vals_t = np.corrcoef(log_dx_t, log_et)[0, 1]
        r2_t = r_vals_t**2

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(
            dx[valid_trace],
            etrace[valid_trace],
            "^-",
            color="C2",
            label=r"trace relative $L_2$ vs finest",
        )
        dx_fit_t = np.linspace(dx[valid_trace].min(), dx[valid_trace].max(), 50)
        ax.loglog(
            dx_fit_t,
            np.exp(np.polyval(coeffs_t, np.log(dx_fit_t))),
            "--",
            color="gray",
            label=f"slope = {slope_t:.2f} ($R^2$ = {r2_t:.4f})",
        )
        # Reference slope triangles
        dx_mid = np.exp(0.5 * (log_dx_t[0] + log_dx_t[-1]))
        e_mid = np.exp(np.polyval(coeffs_t, np.log(dx_mid)))
        for ref_p, ls in [(1.0, ":"), (2.0, "--")]:
            ref_line = e_mid * (dx_fit_t / dx_mid) ** ref_p
            ax.loglog(
                dx_fit_t,
                ref_line,
                ls,
                color="0.6",
                alpha=0.5,
                label=f"O($\\Delta x^{{{ref_p:.0f}}}$)",
            )
        ax.set_xlabel(r"Grid spacing $\Delta x$ (m)")
        ax.set_ylabel(r"Relative $L_2$ trace error vs finest grid")
        ax.set_title(r"Shear-wave convergence: trace $L_2$ (inviscid)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_trace_l2.png", dpi=150)
        plt.close(fig)

    # Console summary
    print("\nConvergence summary:", flush=True)
    for r in rows_sorted:
        print(
            f"n={r['n']:3d}, dX={r['dX']:.3e}, mean4={r['mean4']:.3f}, "
            f"cfit4={r['cfit4']:.3f}, err_mean={r['err_mean']:.3e}, "
            f"err_fit={r['err_fit']:.3e}, "
            f"trace_rel={r.get('trace_rel_l2_vs_ref', float('nan')):.3e}",
            flush=True,
        )

    print(f"\nSaved results to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
