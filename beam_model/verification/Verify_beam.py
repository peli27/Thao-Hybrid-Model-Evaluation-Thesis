"""This script checks the nonlinear beam simulator against analytical and numerical references.

Inputs: The thesis simulator, the paper benchmark and fixed verification settings.
Outputs: Printed comparisons and interactive response, convergence and regime plots.

Running this script performs simulations, including the long-time benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# ----- PROJECT PATHS -----

VERIFICATION_DIR = Path(__file__).resolve().parent
BEAM_DIR = VERIFICATION_DIR.parent
if not (BEAM_DIR / "nonlinear_beam.py").exists():
    # Also support all three files in a single folder for ad hoc testing.
    BEAM_DIR = VERIFICATION_DIR
if not (BEAM_DIR / "nonlinear_beam.py").exists():
    raise FileNotFoundError(
        "Expected original nonlinear_beam.py beside verification/ directory "
        "or in the same folder as this script. Do NOT replace the original simulator."
    )
sys.path.insert(0, str(BEAM_DIR))
from nonlinear_beam import simulate_beam  # noqa: E402
import paper_benchmark as paper  # noqa: E402

# ----- VERIFICATION SETTINGS -----

# SI units: length, width and thickness [m], modulus [Pa], density [kg/m^3].
# Use seven modes and 1% damping unless a check overrides these values.
DEFAULT = dict(L=2.0, b=0.04, h=0.01, E=206e9, rho=7850.0, zeta=0.01, N=7)

# ----- COMPARISON AND REPORTING HELPERS -----


def percent_difference(value, reference):
    # Keep the comparison finite when the reference is close to zero.
    return float(100.0 * abs(value - reference) / max(abs(reference), 1e-30))


class Report:
    # Collect the checks and notes for JSON output, and histories for plotting.
    def __init__(self):
        self.checks = []
        self.notes = {}
        self.plot_data = {}

    def check(self, label, observed, acceptance_limit, units):
        observed = float(observed)
        passed = bool(np.isfinite(observed) and observed <= acceptance_limit)
        self.checks.append(
            dict(
                name=label,
                measured=observed,
                acceptance_limit=float(acceptance_limit),
                units=units,
                passed=passed,
            )
        )
        print(
            f"{'PASS' if passed else 'FAIL'} | {label:47s} "
            f"{observed:.8g} {units} (limit {acceptance_limit:g})"
        )

    def note(self, label, value):
        self.notes[label] = value
        print(f"INFO | {label}: {value}")


# ----- INDEPENDENT FOURTH-ORDER RUNGE-KUTTA SOLVER -----


def independent_rk4(rhs, times, initial):
    # Store one state per column, matching solve_ivp's output layout.
    out = np.empty((len(initial), len(times)), dtype=float)
    out[:, 0] = initial
    for j, step in enumerate(np.diff(times)):
        t, y = times[j], out[:, j]
        k1 = rhs(t, y)
        k2 = rhs(t + step / 2, y + step * k1 / 2)
        k3 = rhs(t + step / 2, y + step * k2 / 2)
        k4 = rhs(t + step, y + step * k3)
        out[:, j + 1] = y + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(out[:, j + 1])):
            raise RuntimeError("RK4 diverged; reduce the timestep")
    return out


# ----- ANALYTICAL STRUCTURAL CHECKS -----


def analytical_checks(r):
    print("\nA. Analytical structural checks")
    # Set t_end very short: frequencies do not depend on forcing integration.
    reference = simulate_beam(**DEFAULT, pressure_amplitude=0, t_end=0.001, dt=0.001)
    modes = np.arange(1, 8)
    k = modes * np.pi / DEFAULT["L"]
    area = DEFAULT["b"] * DEFAULT["h"]
    inertia = DEFAULT["b"] * DEFAULT["h"] ** 3 / 12
    f_rotary = (
        k**2
        / (2 * np.pi)
        * np.sqrt(DEFAULT["E"] * inertia / (DEFAULT["rho"] * (area + inertia * k**2)))
    )
    f_classic = (
        k**2 / (2 * np.pi) * np.sqrt(DEFAULT["E"] * inertia / (DEFAULT["rho"] * area))
    )
    r.check(
        "Rotary-inertia frequency vs analytical",
        np.max(np.abs(reference["frequencies"] - f_rotary) / f_rotary) * 100,
        1e-8,
        "%",
    )
    r.note(
        "Classical E-B vs rotary-inertia frequency difference (expected)",
        f"{np.max(np.abs(reference['frequencies']-f_classic)/f_classic)*100:.7f}%",
    )
    r.note("Seven natural frequencies (Hz)", reference["frequencies"].tolist())

    # Compare against the exact undamped, unforced linear modal response.
    q0 = np.array([8e-4, 1.5e-4, 4e-5, 0, 0, 0, 0])
    v0 = np.array([0, 8e-3, 0, 0, 0, 0, 0])
    result = simulate_beam(
        **{**DEFAULT, "zeta": 0},
        pressure_amplitude=0,
        use_nonlinearity=False,
        y0=np.r_[q0, v0],
        t_end=0.35,
        dt=0.00025,
        rtol=1e-11,
        atol=1e-13,
    )
    omega = 2 * np.pi * result["frequencies"]
    t = result["time"]
    expected = (
        q0[:, None] * np.cos(omega[:, None] * t)
        + (v0 / omega)[:, None] * np.sin(omega[:, None] * t)
    )
    r.check(
        "Exact unforced linear modal response",
        np.max(np.abs(result["q_history"] - expected)),
        5e-9,
        "m",
    )

    # The nonlinear potential must have a gradient equal to the coded force.
    init = np.r_[
        np.array([0.004, 0.0007, 0.0002, 0, 0, 0, 0]),
        np.array([0, 0.003, 0, 0, 0, 0, 0]),
    ]
    result = simulate_beam(
        **{**DEFAULT, "zeta": 0},
        pressure_amplitude=0,
        y0=init,
        t_end=0.5,
        dt=0.0002,
        rtol=1e-11,
        atol=1e-13,
    )
    # The simulator returns displacement histories but not velocities.
    # Integrate its equations again on the same grid to obtain both for energy.
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        result["beam_equations"],
        (0, 0.5),
        init,
        t_eval=result["time"],
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    q, v = sol.y[:7], sol.y[7:]
    kinetic = 0.5 * np.sum(result["modal_mass"][:, None] * v**2, axis=0)
    bending = 0.5 * np.sum(result["modal_stiffness"][:, None] * q**2, axis=0)
    potential = (
        DEFAULT["E"]
        * area
        * DEFAULT["L"]
        / 32
        * np.sum(k[:, None] ** 2 * q**2, axis=0) ** 2
    )
    mechanical_energy = kinetic + bending + potential
    r.check(
        "Undamped/unforced nonlinear energy drift",
        100
        * (np.max(mechanical_energy) - np.min(mechanical_energy))
        / mechanical_energy[0],
        0.002,
        "%",
    )


# ----- SOLVER AND NONLINEAR RESPONSE CHECKS -----


def solver_and_response_checks(r):
    print("\nB. Numerical and nonlinear response checks")
    # At small loads, nonlinear and linear responses should nearly agree.
    small = dict(DEFAULT, pressure_amplitude=0.1)
    lin = simulate_beam(**small, use_nonlinearity=False)
    nl = simulate_beam(**small, use_nonlinearity=True)
    r.check(
        "Small-load nonlinear vs linear peak",
        percent_difference(nl["peak_displacement"], lin["peak_displacement"]),
        0.1,
        "%",
    )

    high = dict(DEFAULT, pressure_amplitude=15.0)
    lin = simulate_beam(**high, use_nonlinearity=False)
    nl = simulate_beam(**high, use_nonlinearity=True)
    r.note(
        "Nonlinear vs linear peak at 15 Pa (descriptive)",
        f"{percent_difference(nl['peak_displacement'], lin['peak_displacement']):.6g}%",
    )
    r.plot_data["linear_comparison"] = (
        lin["time"],
        lin["displacement"],
        nl["displacement"],
    )

    # Compare the simulator with fixed-step RK4 using the same modal equations.
    sol = simulate_beam(
        **DEFAULT, pressure_amplitude=5.0, t_end=0.5, dt=0.0001, rtol=1e-11, atol=1e-13
    )
    rk4 = independent_rk4(sol["beam_equations"], sol["time"], np.zeros(14))
    w_rk4 = np.sin(np.arange(1, 8) * np.pi * 0.65) @ rk4[:7]
    error = (
        np.max(np.abs(sol["displacement"] - w_rk4))
        / max(np.max(np.abs(sol["displacement"])), 1e-30)
        * 100
    )
    r.check("Independent RK4 vs DOP853 displacement", error, 0.02, "%")
    r.plot_data["solver"] = (sol["time"], sol["displacement"], w_rk4)


# ----- MODAL AND NUMERICAL CONVERGENCE -----


def modal_and_numerical_convergence(r, extended=False):
    print("\nC. Modal and numerical convergence checks")
    freqs = simulate_beam(
        **DEFAULT, pressure_amplitude=0, t_end=0.001, dt=0.001
    )["frequencies"]
    convergence_time = 2.5 if extended else 1.25
    modal_table = []

    # Compare seven, nine and eleven modes near three natural frequencies.
    for mode in (1, 3, 5):
        f_exc = 0.95 * freqs[mode - 1]
        spacing = min(0.0005, 1 / (40 * f_exc))
        common = dict(
            DEFAULT,
            pressure_amplitude=15.0,
            excitation_frequency=f_exc,
            t_end=convergence_time,
            dt=spacing,
        )
        p = {n: simulate_beam(**{**common, "N": n}) for n in (7, 9, 11)}
        peak = percent_difference(p[7]["peak_displacement"], p[11]["peak_displacement"])
        rms = percent_difference(p[7]["rms_displacement"], p[11]["rms_displacement"])
        r.check(f"Seven-mode peak convergence near mode {mode}", peak, 1.0, "%")
        r.check(f"Seven-mode RMS convergence near mode {mode}", rms, 1.0, "%")
        modal_table.append(
            dict(
                target_mode=mode,
                excitation_hz=float(f_exc),
                peak_7=float(p[7]["peak_displacement"]),
                peak_9=float(p[9]["peak_displacement"]),
                peak_11=float(p[11]["peak_displacement"]),
                rms_7=float(p[7]["rms_displacement"]),
                rms_9=float(p[9]["rms_displacement"]),
                rms_11=float(p[11]["rms_displacement"]),
            )
        )
    r.notes["modal_comparison"] = modal_table

    # This combined change tests both tolerances and saved output spacing;
    # it does not independently isolate each source of error.
    numerical_window = 10.0 if extended else 4.0
    for label, ratio, pressure in (
        ("near-first-mode", 0.95, 15.0),
        ("above-first-mode", 1.05, 30.0),
    ):
        common = dict(
            DEFAULT,
            pressure_amplitude=pressure,
            excitation_frequency=ratio * freqs[0],
            t_end=numerical_window,
        )
        normal = simulate_beam(**common, dt=0.002, rtol=1e-8, atol=1e-10)
        intermediate = simulate_beam(**common, dt=0.001, rtol=1e-10, atol=1e-12)
        fine = simulate_beam(**common, dt=0.0005, rtol=1e-10, atol=1e-12)
        peak_max = max(
            percent_difference(x["peak_displacement"], fine["peak_displacement"])
            for x in (normal, intermediate)
        )
        rms_max = max(
            percent_difference(x["rms_displacement"], fine["rms_displacement"])
            for x in (normal, intermediate)
        )
        r.check(f"Peak sensitivity, {label}", peak_max, 1.0, "%")
        r.check(f"RMS sensitivity, {label}", rms_max, 1.0, "%")


# ----- DATASET GEOMETRY CHECKS (EXTENDED RUN ONLY) -----


def dataset_geometry_spotchecks(r):
    """Additional modal checks at the approximate ends of the thesis beam span.

    These use short windows for computational cost, and cannot establish
    convergence for every 10-second dataset simulation or LHS sample.
    """
    print("\nC2. Illustrative dataset-geometry modal checks (short windows)")
    cases = [
        ("L=0.60m, h=0.012m", 0.60, 0.012, 0.40),
        ("L=0.15m, h=0.008m", 0.15, 0.008, 0.03),
    ]
    for label, L, h, window in cases:
        setup = dict(DEFAULT, L=L, h=h, pressure_amplitude=15.0)
        reference = simulate_beam(**setup, t_end=0.00001, dt=0.00001)
        drive = 0.95 * reference["frequencies"][0]
        spacing = min(window / 1000, 1 / (40 * drive))
        outputs = {
            n: simulate_beam(
                **{**setup, "N": n},
                excitation_frequency=drive,
                t_end=window,
                dt=spacing,
                rtol=1e-9,
                atol=1e-12,
            )
            for n in (7, 11)
        }
        pk = percent_difference(
            outputs[7]["peak_displacement"], outputs[11]["peak_displacement"]
        )
        rms = percent_difference(
            outputs[7]["rms_displacement"], outputs[11]["rms_displacement"]
        )
        r.check(f"Dataset-range spotcheck peak: {label}", pk, 1.0, "%")
        r.check(f"Dataset-range spotcheck RMS: {label}", rms, 1.0, "%")
    r.note(
        "Dataset-geometry spotcheck scope",
        "Two illustrative length/thickness cases, short time windows; "
        "not an exhaustive convergence test over the complete LHS dataset",
    )


# ----- DIMENSIONAL AND NONDIMENSIONAL CONSISTENCY -----


def paper_formulation_consistency(r):
    print("\nD. Independent nondimensional formulation consistency")
    r.check(
        "Published modal reconstruction Eq. (42)",
        np.max(np.abs(paper.PHI - paper.PUBLISHED_PHI)),
        1e-8,
        "absolute",
    )

    # Strict mapping: t_nd=tau*t_SI, q_SI=h*q_nd,
    # dq_SI/dt_SI = h*tau*dq_nd/dt_nd, a_nd=a_SI/(h*tau**2).
    y_init_si = np.r_[paper.H * paper.Q0, paper.H * paper.TAU * paper.V0]
    tiny = simulate_beam(
        L=paper.L,
        b=paper.WIDTH,
        h=paper.H,
        E=paper.E,
        rho=paper.RHO,
        zeta=0,
        N=7,
        pressure_amplitude=paper.PRESSURE,
        excitation_frequency=paper.Q_F,
        y0=y_init_si,
        t_end=0.0001,
        dt=0.0001,
        rtol=1e-11,
        atol=1e-13,
    )
    dimensional_rhs = tiny["beam_equations"]
    rng = np.random.default_rng(2026)
    errors = []
    for t_nd in (0.0123, 0.0487, 0.0756):
        q = rng.uniform(-0.2, 0.2, 7)
        v = rng.uniform(-0.2, 0.2, 7)
        y_nd = np.r_[q, v]
        y_si = np.r_[paper.H * q, paper.H * paper.TAU * v]
        rhs_nd = paper.equations(t_nd, y_nd)
        rhs_si = dimensional_rhs(t_nd / paper.TAU, y_si)
        errors.append(
            np.max(np.abs(rhs_nd[7:] - rhs_si[7:] / (paper.H * paper.TAU**2)))
        )
    r.check(
        "SI/nondim modal acceleration (three states)",
        max(errors),
        1e-8,
        "nondimensional absolute",
    )

    # Same nonzero initial state, zero damping, short-time physical load.
    short_end, short_dt = 0.08, 0.00008
    nondim = paper.run_paper_reference(t_end_nd=short_end, dt_nd=short_dt)
    si = simulate_beam(
        L=paper.L,
        b=paper.WIDTH,
        h=paper.H,
        E=paper.E,
        rho=paper.RHO,
        zeta=0.0,
        N=7,
        pressure_amplitude=paper.PRESSURE,
        excitation_frequency=paper.Q_F,
        y0=y_init_si,
        t_end=short_end / paper.TAU,
        dt=short_dt / paper.TAU,
        rtol=1e-10,
        atol=1e-12,
    )
    time_err = np.max(np.abs(nondim["time_nd"] - si["time"] * paper.TAU))
    modal_err = np.max(np.abs(nondim["q_nd"] - si["q_history"] / paper.H))
    response_err = np.max(np.abs(nondim["w_nd"] - si["displacement"] / paper.H))
    r.check("SI/nondim time-grid correspondence", time_err, 1e-12, "nondim absolute")
    r.check(
        "SI/nondim seven-modal history agreement", modal_err, 2e-5, "nondim absolute"
    )
    r.check(
        "SI/nondim observation history agreement", response_err, 2e-5, "nondim absolute"
    )
    r.plot_data["formulations"] = (
        nondim["time_nd"],
        nondim["w_nd"],
        si["displacement"] / paper.H,
    )
    r.note(
        "Base-paper long-time benchmark",
        "NOT ATTEMPTED: excluded by thesis decision; no claim of reproduced long-time chaotic maxima",
    )
    r.note(
        "Base-paper short-time comparison window",
        f"nondimensional t=[0,{short_end}], with {len(nondim['time_nd'])} saved points",
    )


# ----- SAVE VERIFICATION FIGURES -----


def save_plots(data, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for key, values in data.items():
        time, a, b = values
        labels = {
            "linear_comparison": (
                "Linear",
                "Nonlinear",
                "Time (s)",
                "Displacement (m)",
            ),
            "solver": ("DOP853", "Independent RK4", "Time (s)", "Displacement (m)"),
            "formulations": (
                "Nondimensional reference",
                "Original SI model",
                "Nondimensional time",
                "Nondimensional displacement",
            ),
        }
        first, second, xlabel, ylabel = labels[key]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(time, a, label=first, lw=1.2)
        ax.plot(time, b, label=second, lw=1.0, ls="--")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=key.replace("_", " ").title())
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{key}.png", dpi=200)
        plt.close(fig)


# ----- RUN VERIFICATION AND SAVE RESULTS -----

# Run from the project root:
#   python beam_model/verification/Verify_beam.py
#   python beam_model/verification/Verify_beam.py --plots
# Add --extended for longer convergence cases and dataset geometry checks.
# Use --output PATH to choose the JSON results file.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="longer modal and numerical sensitivity cases",
    )
    parser.add_argument(
        "--plots", action="store_true", help="save three verification figures"
    )
    parser.add_argument(
        "--output",
        default="verification_results.json",
        help="where to save numerical results JSON",
    )
    args = parser.parse_args()
    report = Report()
    source = Path(sys.modules[simulate_beam.__module__].__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report.note("Simulator file", str(source))
    report.note("Simulator SHA256", digest)
    report.note("Version", "Unmodified original simulator: no new keywords required")

    # Run the checks in the same order as the printed report.
    analytical_checks(report)
    solver_and_response_checks(report)
    modal_and_numerical_convergence(report, extended=args.extended)
    if args.extended:
        dataset_geometry_spotchecks(report)
    paper_formulation_consistency(report)

    # Summarise the checks and optionally save the three comparison figures.
    total = len(report.checks)
    passed = sum(row["passed"] for row in report.checks)
    report.note("Published long-time replication", "NOT CLAIMED")
    report.note(
        "Numerical verification verdict",
        f"{passed}/{total} checks passed for selected verification conditions",
    )
    if args.plots:
        out_dir = Path(args.output).resolve().parent / "verification_figures"
        save_plots(report.plot_data, out_dir)
        report.note("Verification figures", str(out_dir))

    # Save the numerical report, then return a failing exit code if needed.
    result = {
        "simulator": str(source),
        "simulator_sha256": digest,
        "extended": args.extended,
        "checks": report.checks,
        "notes": report.notes,
        "all_checks_passed": passed == total,
        "scope": "verified numerical implementation under specified test conditions; not experimental validation or long-time paper reproduction",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nNUMERICAL VERIFICATION: {passed}/{total} passed")
    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
