"""1D Kelvin-Voigt viscosity validation benchmark.

Self-contained 1D shear FDTD solver with implicit viscous stepping.
Drives a harmonic source at the left boundary, measures steady-state
attenuation and phase speed at sensor locations, and compares to the
analytical Kelvin-Voigt dispersion relation.

Features adaptive frequency scheduling to keep attenuation measurable
across a wide range of viscosity values.

Reference: Section 4.3 of the shear FDTD physics derivation document.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _parse_etas(text):
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        msg = "No eta values were provided."
        raise ValueError(msg)
    return np.asarray(vals, dtype=np.float64)


def _build_eta_grid(eta_text, eta_min, eta_max, eta_num):
    if eta_text:
        return _parse_etas(eta_text)
    if eta_num < 2:
        msg = "eta_num must be >= 2 when using eta_min/eta_max."
        raise ValueError(msg)
    if eta_max < eta_min:
        msg = "eta_max must be >= eta_min."
        raise ValueError(msg)
    return np.linspace(float(eta_min), float(eta_max), int(eta_num), dtype=np.float64)


def _lap1d_center(u, dx):
    lap = np.zeros_like(u, dtype=u.dtype)
    lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    return lap


def _build_sponge(nx, sponge_frac=0.2, sponge_max=0.08):
    damp = np.zeros(nx, dtype=np.float64)
    n_sponge = max(4, int(round(sponge_frac * nx)))
    start = nx - n_sponge
    ramp = np.linspace(0.0, 1.0, n_sponge)
    damp[start:] = sponge_max * ramp * ramp
    return damp


def _solve_tridiagonal(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal systems."""
    n = rhs.size
    c = np.empty(n - 1, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)

    d[0] = rhs[0] / diag[0]
    c[0] = upper[0] / diag[0]
    for i in range(1, n - 1):
        den = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / den
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / den
    den = diag[n - 1] - lower[n - 2] * c[n - 2]
    d[n - 1] = (rhs[n - 1] - lower[n - 2] * d[n - 2]) / den

    x = np.empty(n, dtype=np.float64)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _theory_alpha_cp(f0_hz, rho, mu, eta):
    omega = 2.0 * np.pi * f0_hz
    g_star = mu + 1j * omega * eta
    k = omega * np.sqrt(rho / g_star)
    alpha = float(np.abs(np.imag(k)))
    cp = float(omega / np.abs(np.real(k)))
    return alpha, cp


def _alpha_small_damping(f_hz, rho, mu, eta):
    omega = 2.0 * np.pi * f_hz
    cs = np.sqrt(mu / rho)
    return float(eta * omega * omega / (2.0 * rho * cs * cs * cs))


def _steady_complex_amplitude(signal_t, time_t, omega):
    phasor = np.exp(-1j * omega * time_t)
    return (2.0 / signal_t.size) * np.sum(signal_t * phasor)


def _estimate_noise_floor(signal_t, time_t, omega, c_amp):
    recon = np.real(c_amp * np.exp(1j * omega * time_t))
    resid = signal_t - recon
    return float(np.sqrt(np.mean(resid * resid)))


def _select_close_in_mask(valid_amp, eta, high_eta_threshold, min_sensors):
    mask = np.zeros_like(valid_amp, dtype=bool)
    if np.count_nonzero(valid_amp) < min_sensors:
        return mask
    if eta < high_eta_threshold:
        return valid_amp.copy()

    first_valid = int(np.argmax(valid_amp))
    if not valid_amp[first_valid]:
        return mask
    stop = first_valid
    while stop < valid_amp.size and valid_amp[stop]:
        stop += 1
    mask[first_valid:stop] = True
    if np.count_nonzero(mask) < min_sensors:
        mask[:] = valid_amp
    return mask


def _adaptive_sensor_indices(f_hz, eta, rho, mu, dx, nx, n_sensors, n_efold=3.0, min_start_idx=3):
    """Compute sensor indices that span ~n_efold e-folding lengths.

    For each (f_hz, eta), the expected attenuation alpha is computed from theory.
    Sensors are placed linearly from min_start_idx to min(n_efold/alpha, 0.18*L)
    in grid indices, ensuring good SNR at all sensors.
    """
    if eta <= 0.0:
        # Inviscid: use default spread (3%-18% of domain)
        x_start = max(min_start_idx, int(round(0.03 * nx)))
        x_end = int(round(0.18 * nx))
        return np.linspace(x_start, x_end, n_sensors, dtype=int)

    alpha_exp, _ = _theory_alpha_cp(f_hz, rho, mu, eta)
    if alpha_exp <= 0.0 or not np.isfinite(alpha_exp):
        x_start = max(min_start_idx, int(round(0.03 * nx)))
        x_end = int(round(0.18 * nx))
        return np.linspace(x_start, x_end, n_sensors, dtype=int)

    efold_len = 1.0 / alpha_exp
    x_max_m = n_efold * efold_len
    L = nx * dx
    x_max_m = min(x_max_m, 0.18 * L)

    x_start_idx = max(min_start_idx, int(round(0.5 * efold_len / dx)))
    x_end_idx = max(x_start_idx + n_sensors, int(round(x_max_m / dx)))
    x_end_idx = min(x_end_idx, nx - 2)
    if x_end_idx <= x_start_idx + n_sensors:
        x_end_idx = min(x_start_idx + n_sensors * 2, nx - 2)

    return np.linspace(x_start_idx, x_end_idx, n_sensors, dtype=int)


def _eta_drive_multiplier(eta, high_eta_threshold, drive_eta_slope, max_drive_mult):
    if eta <= high_eta_threshold:
        return 1.0
    mult = 1.0 + drive_eta_slope * (eta - high_eta_threshold)
    return float(min(max_drive_mult, max(1.0, mult)))


def _eta_frequency_hz(
    eta,
    f0_base_hz,
    adaptive_f0,
    f0_mid_eta,
    f0_high_eta,
    f0_mid_scale,
    f0_high_scale,
):
    if not adaptive_f0:
        return float(f0_base_hz)
    if eta >= f0_high_eta:
        return float(f0_base_hz * f0_high_scale)
    if eta >= f0_mid_eta:
        return float(f0_base_hz * f0_mid_scale)
    return float(f0_base_hz)


def _solve_frequency_for_alpha_target(
    eta,
    alpha_target,
    f_low,
    f_high,
    rho,
    mu,
    n_iter=40,
):
    a_low, _ = _theory_alpha_cp(f_low, rho, mu, eta)
    a_high, _ = _theory_alpha_cp(f_high, rho, mu, eta)
    if alpha_target <= a_low:
        return float(f_low)
    if alpha_target >= a_high:
        return float(f_high)
    lo = f_low
    hi = f_high
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        a_mid, _ = _theory_alpha_cp(mid, rho, mu, eta)
        if a_mid < alpha_target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _build_frequency_schedule(
    eta_values,
    *,
    f0_base_hz,
    mode,
    rho,
    mu,
    f0_mid_eta,
    f0_high_eta,
    f0_mid_scale,
    f0_high_scale,
    f0_eta_ref,
    f0_alpha_exp,
    f0_min_scale,
    f0_smooth,
    f0_dim_beta,
):
    eta_values = np.asarray(eta_values, dtype=np.float64)
    if mode == "off":
        return np.full_like(eta_values, float(f0_base_hz), dtype=np.float64)

    if mode == "piecewise":
        return np.array(
            [
                _eta_frequency_hz(
                    float(eta),
                    float(f0_base_hz),
                    True,
                    float(f0_mid_eta),
                    float(f0_high_eta),
                    float(f0_mid_scale),
                    float(f0_high_scale),
                )
                for eta in eta_values
            ],
            dtype=np.float64,
        )

    if mode == "dimensional":
        eta_ref = max(float(f0_eta_ref), np.finfo(np.float64).eps)
        f_ref = float(f0_base_hz)
        f_floor = float(f0_base_hz) * float(f0_min_scale)

        if f0_dim_beta is None:
            p_alpha = max(0.0, min(float(f0_alpha_exp), 1.0))
            beta = 0.5 * (1.0 - p_alpha)
        else:
            beta = float(f0_dim_beta)
        beta = max(0.0, min(beta, 0.49))

        f_raw = np.full_like(eta_values, f_ref, dtype=np.float64)
        for i, eta in enumerate(eta_values):
            e = float(eta)
            if e <= eta_ref:
                f_raw[i] = f_ref
            else:
                f_raw[i] = f_ref * ((e / eta_ref) ** (-beta))
            f_raw[i] = max(f_floor, min(f_ref, f_raw[i]))

        rho_f = float(rho)
        mu_f = float(mu)
        f_corr = np.full_like(f_raw, f_ref, dtype=np.float64)
        for i, eta in enumerate(eta_values):
            e = float(eta)
            if e <= 0.0:
                f_corr[i] = f_ref
                continue
            alpha_target = _alpha_small_damping(float(f_raw[i]), rho_f, mu_f, e)
            f_corr[i] = _solve_frequency_for_alpha_target(
                e,
                alpha_target,
                f_floor,
                f_ref,
                rho_f,
                mu_f,
            )

        f_sched = f_corr.copy()
        for i in range(1, f_sched.size):
            f_sched[i] = min(f_sched[i], f_sched[i - 1])
            f_sched[i] = min(
                f_sched[i - 1],
                float(f0_smooth) * f_sched[i - 1] + (1.0 - float(f0_smooth)) * f_sched[i],
            )
        return f_sched

    # Intelligent mode: dimensional-analysis-guided target with exact correction.
    eta_ref = max(float(f0_eta_ref), np.finfo(np.float64).eps)
    f_low = float(f0_base_hz) * float(f0_min_scale)
    f_high = float(f0_base_hz)
    alpha_ref_sd = _alpha_small_damping(float(f0_base_hz), rho, mu, eta_ref)
    f_raw = np.full_like(eta_values, float(f0_base_hz), dtype=np.float64)

    for i, eta in enumerate(eta_values):
        e = float(eta)
        if e <= eta_ref:
            f_raw[i] = float(f0_base_hz)
            continue
        alpha_target = alpha_ref_sd * ((e / eta_ref) ** float(f0_alpha_exp))
        f_raw[i] = _solve_frequency_for_alpha_target(e, alpha_target, f_low, f_high, rho, mu)

    f_sched = f_raw.copy()
    for i in range(1, f_sched.size):
        f_sched[i] = min(f_sched[i], f_sched[i - 1])
        f_sched[i] = min(
            f_sched[i - 1],
            float(f0_smooth) * f_sched[i - 1] + (1.0 - float(f0_smooth)) * f_sched[i],
        )
    return f_sched


def run_1d_benchmark(
    *,
    eta_values=(0.0, 0.1, 0.25, 0.5, 1.0),
    nx=1200,
    dx=1.0e-4,
    rho=1000.0,
    cs=2.0,
    f0_hz=800.0,
    f0_mode="dimensional",
    f0_mid_eta=0.3,
    f0_high_eta=0.6,
    f0_mid_scale=0.75,
    f0_high_scale=0.5,
    f0_eta_ref=0.2,
    f0_alpha_exp=0.6,
    f0_min_scale=0.4,
    f0_smooth=0.7,
    f0_dim_beta=None,
    cfl=0.3,
    n_cycles=120,
    settle_cycles=70,
    drive_amp=2.0e-7,
    high_eta_threshold=0.2,
    drive_eta_slope=6.0,
    max_drive_mult=6.0,
    snr_min=6.0,
    snr_min_relaxed=2.0,
    min_fit_sensors=3,
    n_amp_sensors=8,
    sponge_frac=0.2,
    sponge_max=0.08,
    save_dir=None,
    make_plots=True,
):
    """Run the 1D Kelvin-Voigt benchmark and return results dict."""
    eta_values = np.asarray(eta_values, dtype=np.float64)
    if eta_values.ndim != 1:
        msg = "eta_values must be 1D."
        raise ValueError(msg)
    if np.any(eta_values < 0.0):
        msg = "eta values must be non-negative."
        raise ValueError(msg)
    if settle_cycles >= n_cycles:
        msg = "settle_cycles must be smaller than n_cycles."
        raise ValueError(msg)

    mu = rho * cs * cs
    dt = cfl * dx / cs
    x = np.arange(nx, dtype=np.float64) * dx
    damp = _build_sponge(nx, sponge_frac=sponge_frac, sponge_max=sponge_max)
    keep = 1.0 - damp

    phase_base = max(3, int(round(0.03 * nx)))
    # ~lambda/6 spacing at base f0
    phase_stride = max(2, int(round((cs / f0_hz) / (6.0 * dx))))
    sensor_idx_phase = phase_base + phase_stride * np.arange(8, dtype=int)
    sensor_idx_phase = sensor_idx_phase[sensor_idx_phase < int(round(0.25 * nx))]
    if sensor_idx_phase.size < 4:
        sensor_idx_phase = np.arange(phase_base, min(phase_base + 24, nx - 2), 3, dtype=int)
    sensor_x_phase = x[sensor_idx_phase]

    alpha_meas = np.full(eta_values.size, np.nan, dtype=np.float64)
    alpha_meas_raw = np.full(eta_values.size, np.nan, dtype=np.float64)
    cp_meas = np.full(eta_values.size, np.nan, dtype=np.float64)
    alpha_theory = np.full(eta_values.size, np.nan, dtype=np.float64)
    cp_theory = np.full(eta_values.size, np.nan, dtype=np.float64)
    amp_by_eta = np.full((eta_values.size, n_amp_sensors), np.nan, dtype=np.float64)
    phase_by_eta = np.full((eta_values.size, sensor_idx_phase.size), np.nan, dtype=np.float64)
    noise_floor_by_eta = np.full((eta_values.size, n_amp_sensors), np.nan, dtype=np.float64)
    snr_by_eta = np.full((eta_values.size, n_amp_sensors), np.nan, dtype=np.float64)
    used_sensor_mask_amp = np.zeros((eta_values.size, n_amp_sensors), dtype=bool)
    sensor_idx_amp_by_eta = np.full((eta_values.size, n_amp_sensors), -1, dtype=np.int32)
    sensor_x_amp_by_eta = np.full((eta_values.size, n_amp_sensors), np.nan, dtype=np.float64)
    drive_amp_by_eta = np.full(eta_values.size, np.nan, dtype=np.float64)
    f0_by_eta = _build_frequency_schedule(
        eta_values,
        f0_base_hz=float(f0_hz),
        mode=str(f0_mode),
        rho=float(rho),
        mu=float(mu),
        f0_mid_eta=float(f0_mid_eta),
        f0_high_eta=float(f0_high_eta),
        f0_mid_scale=float(f0_mid_scale),
        f0_high_scale=float(f0_high_scale),
        f0_eta_ref=float(f0_eta_ref),
        f0_alpha_exp=float(f0_alpha_exp),
        f0_min_scale=float(f0_min_scale),
        f0_smooth=float(f0_smooth),
        f0_dim_beta=f0_dim_beta,
    )

    for i, eta in enumerate(eta_values):
        f_case = float(f0_by_eta[i])
        period = 1.0 / f_case
        total_time = n_cycles * period
        n_steps = int(np.ceil(total_time / dt))
        time = np.arange(n_steps, dtype=np.float64) * dt
        omega = 2.0 * np.pi * f_case

        drive_mult = _eta_drive_multiplier(
            float(eta),
            float(high_eta_threshold),
            float(drive_eta_slope),
            float(max_drive_mult),
        )
        drive_amp_case = drive_amp * drive_mult
        drive_amp_by_eta[i] = drive_amp_case
        sensor_idx_amp = _adaptive_sensor_indices(
            f_case,
            float(eta),
            rho,
            mu,
            dx,
            nx,
            n_amp_sensors,
        )
        sensor_x_amp = x[sensor_idx_amp]
        sensor_mode = "adaptive"
        sensor_idx_amp_by_eta[i, :] = sensor_idx_amp
        sensor_x_amp_by_eta[i, :] = sensor_x_amp

        t_start = max(settle_cycles * period, sensor_x_amp.max() / cs + 3.0 * period)
        t_reflect = (2.0 * x[-1] - sensor_x_phase.min()) / cs
        t_end = min(time[-1], 0.9 * t_reflect)
        if t_end <= t_start + 5.0 * period:
            t_start = max(sensor_x_amp.max() / cs + 2.0 * period, 0.5 * time[-1])
            t_end = time[-1]
        i_start = int(np.searchsorted(time, t_start))
        i_end = int(np.searchsorted(time, t_end))
        if i_end - i_start < 20:
            msg = "Not enough analysis samples. Increase n_cycles or reduce frequency."
            raise RuntimeError(msg)
        t_ss = time[i_start:i_end]

        print(
            f"Running 1D Kelvin-Voigt case eta={eta:.4g} Pa*s "
            f"(f0={f_case:.1f} Hz, drive x{drive_mult:.2f}, "
            f"{sensor_mode} sensors)"
        )
        u = np.zeros(nx, dtype=np.float64)
        v = np.zeros(nx, dtype=np.float64)
        r = dt * eta / (rho * dx * dx)
        if r > 0.0:
            nint = nx - 2
            lower = -r * np.ones(nint - 1, dtype=np.float64)
            diag = (1.0 + 2.0 * r) * np.ones(nint, dtype=np.float64)
            upper = -r * np.ones(nint - 1, dtype=np.float64)

        sensor_hist_amp = np.zeros((n_amp_sensors, n_steps), dtype=np.float64)
        sensor_hist_phase = np.zeros((sensor_idx_phase.size, n_steps), dtype=np.float64)
        for it, t in enumerate(time):
            lap_u = _lap1d_center(u, dx)
            v_star = v + dt * (mu / rho) * lap_u

            if r > 0.0:
                rhs = v_star[1:-1].copy()
                rhs[0] += r * v_star[0]
                rhs[-1] += r * v_star[-1]
                v_new_int = _solve_tridiagonal(lower, diag, upper, rhs)
                v[1:-1] = v_new_int
                v[0] = v_star[0]
                v[-1] = v_star[-1]
            else:
                v = v_star

            v *= keep
            u = u + dt * v

            # Left harmonic drive, right fixed with sponge nearby.
            u[0] = drive_amp_case * np.sin(omega * t)
            u[-1] = u[-2]
            v[0] = drive_amp_case * omega * np.cos(omega * t)
            v[-1] = v[-2]

            sensor_hist_amp[:, it] = u[sensor_idx_amp]
            sensor_hist_phase[:, it] = u[sensor_idx_phase]

        # Frequency-domain amplitude/phase at f0.
        cvals_amp = np.array(
            [
                _steady_complex_amplitude(sensor_hist_amp[j, i_start:i_end], t_ss, omega)
                for j in range(n_amp_sensors)
            ]
        )
        cvals_phase = np.array(
            [
                _steady_complex_amplitude(sensor_hist_phase[j, i_start:i_end], t_ss, omega)
                for j in range(sensor_idx_phase.size)
            ]
        )
        amps = np.abs(cvals_amp)
        phases = np.unwrap(np.angle(cvals_phase))
        amp_by_eta[i] = amps
        phase_by_eta[i] = phases

        noise_floor = np.array(
            [
                _estimate_noise_floor(sensor_hist_amp[j, i_start:i_end], t_ss, omega, cvals_amp[j])
                for j in range(n_amp_sensors)
            ]
        )
        noise_floor_by_eta[i] = noise_floor
        snr = amps / np.maximum(noise_floor, np.finfo(np.float64).eps)
        snr_by_eta[i] = snr

        valid_amp = np.isfinite(amps) & (amps > 0.0) & np.isfinite(noise_floor) & (snr >= snr_min)
        fit_mask = _select_close_in_mask(valid_amp, float(eta), high_eta_threshold, min_fit_sensors)
        used_sensor_mask_amp[i] = fit_mask

        # Raw estimate for visualization/debugging (no SNR floor gating).
        valid_amp_raw = np.isfinite(amps) & (amps > 0.0)
        fit_mask_raw = _select_close_in_mask(
            valid_amp_raw, float(eta), high_eta_threshold, min_fit_sensors
        )
        if np.count_nonzero(fit_mask_raw) >= min_fit_sensors:
            slope_amp_raw = np.polyfit(sensor_x_amp[fit_mask_raw], np.log(amps[fit_mask_raw]), 1)[0]
            alpha_meas_raw[i] = float(-slope_amp_raw)

        low_count = int(np.count_nonzero(~valid_amp))
        if low_count > 0:
            print(
                f"  eta={eta:.3g}: flagged {low_count}/{n_amp_sensors} "
                f"sensors below floor (SNR<{snr_min:.1f})."
            )

        if np.count_nonzero(fit_mask) < min_fit_sensors:
            snr_floor_relaxed = max(0.5, min(float(snr_min), float(snr_min_relaxed)))
            valid_amp_relaxed = (
                np.isfinite(amps)
                & (amps > 0.0)
                & np.isfinite(noise_floor)
                & (snr >= snr_floor_relaxed)
            )
            fit_mask_relaxed = _select_close_in_mask(
                valid_amp_relaxed, float(eta), high_eta_threshold, min_fit_sensors
            )
            if np.count_nonzero(fit_mask_relaxed) >= min_fit_sensors:
                fit_mask = fit_mask_relaxed
                used_sensor_mask_amp[i] = fit_mask
                print(
                    f"  eta={eta:.3g}: using relaxed SNR floor "
                    f"{snr_floor_relaxed:.1f} for attenuation fit."
                )
            elif np.count_nonzero(fit_mask_relaxed) >= 2:
                fit_mask = _select_close_in_mask(
                    valid_amp_relaxed, float(eta), high_eta_threshold, 2
                )
                used_sensor_mask_amp[i] = fit_mask
                print(
                    f"  eta={eta:.3g}: using 2-sensor relaxed attenuation "
                    f"fit (SNR floor {snr_floor_relaxed:.1f})."
                )
            elif np.count_nonzero(fit_mask_raw) >= 2:
                fit_mask = _select_close_in_mask(valid_amp_raw, float(eta), high_eta_threshold, 2)
                used_sensor_mask_amp[i] = fit_mask
                print(
                    f"  eta={eta:.3g}: using raw 2-sensor attenuation fit "
                    "(insufficient gated sensors)."
                )

        if np.count_nonzero(fit_mask) >= min_fit_sensors:
            slope_amp = np.polyfit(sensor_x_amp[fit_mask], np.log(amps[fit_mask]), 1)[0]
            alpha_meas[i] = float(-slope_amp)

        valid_ph = np.isfinite(phases)
        if np.count_nonzero(valid_ph) >= min_fit_sensors:
            slope_ph = np.polyfit(sensor_x_phase[valid_ph], phases[valid_ph], 1)[0]
            beta = np.abs(slope_ph)
            if beta > np.finfo(np.float64).eps:
                cp_meas[i] = float(omega / beta)

        alpha_theory[i], cp_theory[i] = _theory_alpha_cp(f_case, rho, mu, float(eta))

    result = {
        "eta_values": eta_values,
        "f0_hz": f0_hz,
        "dx": dx,
        "dt": dt,
        "nx": nx,
        "sensor_idx_amp_by_eta": sensor_idx_amp_by_eta,
        "sensor_x_amp_by_eta": sensor_x_amp_by_eta,
        "drive_amp_by_eta": drive_amp_by_eta,
        "f0_by_eta": f0_by_eta,
        "sensor_idx_phase": sensor_idx_phase,
        "sensor_x_phase": sensor_x_phase,
        "alpha_meas": alpha_meas,
        "alpha_meas_raw": alpha_meas_raw,
        "cp_meas": cp_meas,
        "alpha_theory": alpha_theory,
        "cp_theory": cp_theory,
        "amp_by_eta": amp_by_eta,
        "phase_by_eta": phase_by_eta,
        "noise_floor_by_eta": noise_floor_by_eta,
        "snr_by_eta": snr_by_eta,
        "used_sensor_mask_amp": used_sensor_mask_amp,
        "snr_min": float(snr_min),
        "snr_min_relaxed": float(snr_min_relaxed),
        "high_eta_threshold": float(high_eta_threshold),
        "drive_eta_slope": float(drive_eta_slope),
        "max_drive_mult": float(max_drive_mult),
        "adaptive_f0": bool(f0_mode != "off"),
        "f0_mode": str(f0_mode),
        "f0_eta_ref": float(f0_eta_ref),
        "f0_alpha_exp": float(f0_alpha_exp),
        "f0_min_scale": float(f0_min_scale),
        "f0_smooth": float(f0_smooth),
        "f0_dim_beta": (float(f0_dim_beta) if f0_dim_beta is not None else np.nan),
    }

    if make_plots or save_dir is not None:
        out_dir = Path(save_dir) if save_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        if np.any(np.isfinite(alpha_meas_raw)):
            ax.plot(
                eta_values,
                alpha_meas_raw,
                "o-",
                label="Measured alpha (no SNR gate)",
            )
        ax.plot(eta_values, alpha_theory, "s--", label="Theory alpha")
        ax.set_xlabel("Viscosity eta (Pa*s)")
        ax.set_ylabel("Attenuation alpha (Np/m)")
        ax.set_title("1D Kelvin-Voigt attenuation validation")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "kelvin_voigt_1d_alpha.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(eta_values, cp_meas, "o-", label="Measured c_phase")
        ax.plot(eta_values, cp_theory, "s--", label="Theory c_phase")
        ax.set_xlabel("Viscosity eta (Pa*s)")
        ax.set_ylabel("Phase speed (m/s)")
        ax.set_title("1D Kelvin-Voigt speed validation")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "kelvin_voigt_1d_speed.png", dpi=150)
        plt.close(fig)

        np.savez(out_dir / "kelvin_voigt_1d_validation.npz", **result)

    return result


def main():
    parser = argparse.ArgumentParser(description="1D Kelvin-Voigt shear-wave validation benchmark.")
    parser.add_argument(
        "--etas",
        type=str,
        default=None,
        help="Comma-separated eta list. Overrides eta-min/max/num when provided.",
    )
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--eta-num", type=int, default=11)
    parser.add_argument("--nx", type=int, default=1200)
    parser.add_argument("--dx", type=float, default=1.0e-4)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--cs", type=float, default=2.0)
    parser.add_argument("--f0", type=float, default=800.0, help="Drive frequency in Hz.")
    parser.add_argument(
        "--f0-mode",
        type=str,
        choices=("off", "piecewise", "intelligent", "dimensional"),
        default="dimensional",
        help="Frequency schedule strategy versus eta.",
    )
    parser.add_argument("--f0-mid-eta", type=float, default=0.3)
    parser.add_argument("--f0-high-eta", type=float, default=0.6)
    parser.add_argument("--f0-mid-scale", type=float, default=0.75)
    parser.add_argument("--f0-high-scale", type=float, default=0.5)
    parser.add_argument("--f0-eta-ref", type=float, default=0.2)
    parser.add_argument("--f0-alpha-exp", type=float, default=0.6)
    parser.add_argument("--f0-min-scale", type=float, default=0.4)
    parser.add_argument("--f0-smooth", type=float, default=0.7)
    parser.add_argument(
        "--f0-dim-beta",
        type=float,
        default=None,
        help=(
            "Optional fixed beta for dimensional schedule f~eta^-beta. "
            "If omitted, beta is inferred from f0-alpha-exp via beta=(1-p)/2."
        ),
    )
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--n-cycles", type=int, default=120)
    parser.add_argument("--settle-cycles", type=int, default=70)
    parser.add_argument("--drive-amp", type=float, default=2.0e-7)
    parser.add_argument("--high-eta-threshold", type=float, default=0.2)
    parser.add_argument("--drive-eta-slope", type=float, default=6.0)
    parser.add_argument("--max-drive-mult", type=float, default=6.0)
    parser.add_argument("--snr-min", type=float, default=6.0)
    parser.add_argument("--snr-min-relaxed", type=float, default=2.0)
    parser.add_argument("--min-fit-sensors", type=int, default=3)
    parser.add_argument("--n-amp-sensors", type=int, default=8)
    parser.add_argument("--sponge-frac", type=float, default=0.2)
    parser.add_argument("--sponge-max", type=float, default=0.08)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    etas = _build_eta_grid(args.etas, args.eta_min, args.eta_max, args.eta_num)
    res = run_1d_benchmark(
        eta_values=etas,
        nx=args.nx,
        dx=args.dx,
        rho=args.rho,
        cs=args.cs,
        f0_hz=args.f0,
        f0_mode=args.f0_mode,
        f0_mid_eta=args.f0_mid_eta,
        f0_high_eta=args.f0_high_eta,
        f0_mid_scale=args.f0_mid_scale,
        f0_high_scale=args.f0_high_scale,
        f0_eta_ref=args.f0_eta_ref,
        f0_alpha_exp=args.f0_alpha_exp,
        f0_min_scale=args.f0_min_scale,
        f0_smooth=args.f0_smooth,
        f0_dim_beta=args.f0_dim_beta,
        cfl=args.cfl,
        n_cycles=args.n_cycles,
        settle_cycles=args.settle_cycles,
        drive_amp=args.drive_amp,
        high_eta_threshold=args.high_eta_threshold,
        drive_eta_slope=args.drive_eta_slope,
        max_drive_mult=args.max_drive_mult,
        snr_min=args.snr_min,
        snr_min_relaxed=args.snr_min_relaxed,
        min_fit_sensors=args.min_fit_sensors,
        n_amp_sensors=args.n_amp_sensors,
        sponge_frac=args.sponge_frac,
        sponge_max=args.sponge_max,
        save_dir=args.save_dir,
        make_plots=True,
    )

    print("\n1D Kelvin-Voigt summary")
    print("eta (Pa*s) | alpha_meas (Np/m) | alpha_theory (Np/m) | cp_meas (m/s) | cp_theory (m/s)")
    for eta, am, at, cm, ct in zip(
        res["eta_values"],
        res["alpha_meas"],
        res["alpha_theory"],
        res["cp_meas"],
        res["cp_theory"],
    ):
        print(f"{eta:9.4f} | {am:17.4e} | {at:18.4e} | {cm:13.5f} | {ct:13.5f}")


if __name__ == "__main__":
    main()
