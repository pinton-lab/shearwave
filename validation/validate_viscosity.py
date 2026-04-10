"""3D viscosity sweep validation for the shear FDTD solver.

Runs the wavespeed validation at multiple viscosity values and measures
extra attenuation relative to the inviscid case. Compares to the analytical
Kelvin-Voigt prediction.

Reference: Section 4.4 of the shear FDTD physics derivation document.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from validate_wavespeed import run_validation


def _parse_eta_list(eta_text):
    vals = [float(x.strip()) for x in eta_text.split(",") if x.strip()]
    if not vals:
        msg = "At least one eta value is required."
        raise ValueError(msg)
    return np.array(vals, dtype=np.float64)


def _dominant_frequency_hz(t_vel, vel_traces):
    dt = float(t_vel[1] - t_vel[0])
    spec = np.mean(np.abs(np.fft.rfft(vel_traces, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(vel_traces.shape[1], d=dt)
    if spec.size <= 1:
        return 0.0
    idx = 1 + int(np.argmax(spec[1:]))  # Ignore DC.
    return float(freqs[idx])


def _arrival_peak_amplitude(trace, t, start_time):
    i0 = int(np.searchsorted(t, start_time))
    if i0 >= trace.size:
        return np.nan
    seg = trace[i0:]
    if seg.size == 0:
        return np.nan
    return float(np.max(np.abs(seg)))


def _kelvin_voigt_alpha_cp(omega, rho, mu, eta):
    if omega <= 0.0:
        return np.nan, np.nan
    G_star = mu + 1j * omega * eta
    k = omega * np.sqrt(rho / G_star)
    alpha = float(np.abs(np.imag(k)))
    cp = float(omega / np.abs(np.real(k)))
    return alpha, cp


def run_viscosity_validation(
    *,
    eta_values=(0.0, 0.1, 0.25, 0.5, 1.0),
    nX=180,
    nY=180,
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
    save_dir=None,
    make_plots=True,
):
    """Run viscosity sweep validation and return results dict."""
    eta_values = np.asarray(eta_values, dtype=np.float64)
    if eta_values.ndim != 1:
        msg = "eta_values must be 1D."
        raise ValueError(msg)
    if np.any(eta_values < 0.0):
        msg = "eta_values must be non-negative."
        raise ValueError(msg)
    if not np.any(np.isclose(eta_values, 0.0)):
        eta_values = np.concatenate(([0.0], eta_values))

    case_results = []
    for eta in eta_values:
        print(f"\nRunning viscosity case eta={eta:.4g} Pa*s")
        res = run_validation(
            nX=nX,
            nY=nY,
            nZ=nZ,
            dX=dX,
            dY=dY,
            dZ=dZ,
            rho=rho,
            cs_target=cs_target,
            CFL=CFL,
            Tmax=Tmax,
            amplitude=amplitude,
            sensor_offsets=sensor_offsets,
            eta=float(eta),
            save_dir=None,
            make_plots=False,
            make_movie=False,
        )
        case_results.append(res)

    ref_idx = int(np.argmin(np.abs(eta_values)))
    ref = case_results[ref_idx]
    sensor_dist = np.asarray(ref["sensor_distances"], dtype=np.float64)
    t_vel = np.asarray(ref["t_vec"][1:], dtype=np.float64)
    if t_vel.size < 2:
        msg = "Not enough time samples to estimate dominant frequency."
        raise RuntimeError(msg)

    vel_ref = np.asarray(ref["vel_traces"], dtype=np.float64)
    f_dom = _dominant_frequency_hz(t_vel, vel_ref)
    omega_dom = 2.0 * np.pi * f_dom

    amplitudes = np.full((len(eta_values), sensor_dist.size), np.nan, dtype=np.float64)
    speed_measured = np.full(len(eta_values), np.nan, dtype=np.float64)
    alpha_extra_measured = np.full(len(eta_values), np.nan, dtype=np.float64)
    alpha_theory = np.full(len(eta_values), np.nan, dtype=np.float64)
    cp_theory = np.full(len(eta_values), np.nan, dtype=np.float64)

    for i, (eta, res) in enumerate(zip(eta_values, case_results)):
        speed_measured[i] = float(res["c_fit"])
        vel = np.asarray(res["vel_traces"], dtype=np.float64)
        for j, dist in enumerate(sensor_dist):
            start_time = 0.5 * dist / cs_target
            amplitudes[i, j] = _arrival_peak_amplitude(vel[j], t_vel, start_time)

        alpha_theory[i], cp_theory[i] = _kelvin_voigt_alpha_cp(
            omega_dom, rho, rho * cs_target**2, float(eta)
        )

    amp_ref = amplitudes[ref_idx]
    for i in range(len(eta_values)):
        ratio = amplitudes[i] / np.maximum(amp_ref, np.finfo(np.float64).eps)
        valid = np.isfinite(ratio) & (ratio > 0.0) & np.isfinite(sensor_dist)
        if np.count_nonzero(valid) >= 2:
            x = sensor_dist[valid]
            y = np.log(ratio[valid])
            slope = np.polyfit(x, y, 1)[0]
            alpha_extra_measured[i] = float(-slope)

    sort_idx = np.argsort(eta_values)
    alpha_sorted = alpha_extra_measured[sort_idx]
    finite = np.isfinite(alpha_sorted)
    monotonic = bool(np.all(np.diff(alpha_sorted[finite]) >= -1e-9))

    result = {
        "eta_values": eta_values,
        "sensor_distances": sensor_dist,
        "f_dom_hz": f_dom,
        "omega_dom_rad_s": omega_dom,
        "speed_measured": speed_measured,
        "speed_theory": cp_theory,
        "alpha_extra_measured": alpha_extra_measured,
        "alpha_theory": alpha_theory,
        "amplitudes": amplitudes,
        "monotonic_alpha": monotonic,
    }

    if make_plots or save_dir is not None:
        out_dir = Path(save_dir) if save_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            eta_values,
            alpha_extra_measured,
            "o-",
            label="Measured alpha (extra vs eta=0)",
        )
        ax.plot(eta_values, alpha_theory, "s--", label="Theory alpha")
        ax.set_xlabel("Viscosity eta (Pa*s)")
        ax.set_ylabel("Attenuation alpha (Np/m)")
        ax.set_title("Viscosity attenuation validation")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "shear_viscosity_attenuation.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(eta_values, speed_measured, "o-", label="Measured c_fit")
        ax.plot(eta_values, cp_theory, "s--", label="Theory c_phase")
        ax.set_xlabel("Viscosity eta (Pa*s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Viscosity phase-speed validation")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "shear_viscosity_speed.png", dpi=150)
        plt.close(fig)

        np.savez(out_dir / "shear_viscosity_validation.npz", **result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate viscosity effect for shear FDTD.")
    parser.add_argument("--etas", type=str, default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--nx", type=int, default=180)
    parser.add_argument("--ny", type=int, default=180)
    parser.add_argument("--nz", type=int, default=180)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=6e-3)
    parser.add_argument("--amplitude", type=float, default=3e-5)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    etas = _parse_eta_list(args.etas)
    res = run_viscosity_validation(
        eta_values=etas,
        nX=args.nx,
        nY=args.ny,
        nZ=args.nz,
        CFL=args.cfl,
        Tmax=args.tmax,
        amplitude=args.amplitude,
        save_dir=args.save_dir,
        make_plots=True,
    )

    print("\nViscosity validation summary")
    print(f"Dominant frequency: {res['f_dom_hz']:.2f} Hz")
    print("eta (Pa*s) | alpha_meas (Np/m) | alpha_theory (Np/m) | c_meas (m/s) | c_theory (m/s)")
    for eta, am, at, cm, ct in zip(
        res["eta_values"],
        res["alpha_extra_measured"],
        res["alpha_theory"],
        res["speed_measured"],
        res["speed_theory"],
    ):
        print(f"{eta:9.4f} | {am:17.4e} | {at:18.4e} | {cm:11.4f} | {ct:12.4f}")
    print(f"Monotonic attenuation vs eta: {res['monotonic_alpha']}")


if __name__ == "__main__":
    main()
