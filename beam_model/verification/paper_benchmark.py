"""This script reproduces the deterministic nondimensional beam benchmark from the base paper to help verify the beam model.

Inputs: The paper geometry, forcing, initial conditions and seven-mode equations.
Outputs: Nondimensional response histories, modal maxima and time-window comparisons.

PURPOSE: Verify dimensional/nondimensional equation consistency and a
well-resolved short-time response against the beam simulator.
This is not a reproduction of the paper's long-time chaotic maxima.
"""

import numpy as np
from scipy.integrate import solve_ivp

# ----- PAPER INPUT PARAMETERS -----

# Paper's deterministic beam configuration (SI values).
E = 206e9  # Young's modulus [Pa]
RHO = 7850.0  # Density [kg/m^3]
L = 2.0  # Beam length [m]
WIDTH = 0.04  # Beam width [m]
H = 0.01  # Beam thickness [m]
PRESSURE = 15.0  # Harmonic pressure amplitude [Pa]
Q_F = 2000.0  # Eq. (41), interpreted as Hz with t_nd = tau*t_SI
N = 7
OBSERVATION_RATIO = 0.65

# ----- NONDIMENSIONAL PARAMETERS AND INITIAL CONDITIONS -----

# Nondimensional time: t_nd = tau*t_SI.
TAU = np.sqrt(E * H**2 / (12.0 * RHO * L**4))
A = H**2 / (12.0 * L**2)
B = E * H**2 / (2.0 * L**4 * RHO * TAU**2)
C = E * H**2 / (12.0 * L**4 * RHO * TAU**2)
MODES = np.arange(1, N + 1)
K = MODES * np.pi
PHI = np.sin(K * OBSERVATION_RATIO)

# Initial modal displacements and velocities in nondimensional units.
Q0 = np.array([0.8, 0.008, 0.0005, 0.002, 0.001, 0.0007, 0.0004])
V0 = np.array([0.8, 0.69, 0.52, 0.43, 0.34, 0.28, 0.15])
Y0 = np.r_[Q0, V0]
FORCING_PERIOD_ND = TAU / Q_F

# Eq. (42) (published 10-digit coefficients): a directly checkable value.
PUBLISHED_PHI = np.array(
    [
        0.8910065241,
        -0.8090169946,
        -0.1564344645,
        0.9510565161,
        -0.7071067782,
        -0.3090169934,
        0.9876883396,
    ]
)

# ----- NONDIMENSIONAL EQUATIONS OF MOTION -----


def equations(t_nd, y_nd, *, forcing=True, nonlinear=True):
    """Nondimensional modal time derivative for seven displacement/velocity pairs.

    y_nd = [q_1..q_7, dq_1/dt_nd..dq_7/dt_nd].
    Assumes no modal damping, consistent with the deterministic paper case.
    """
    # First N entries are displacements; the remaining entries are velocities.
    q = np.asarray(y_nd[:N])
    v = np.asarray(y_nd[N:])

    modal_mass = 0.5 * (1.0 + A * K**2)
    linear = 0.5 * C * K**4 * q
    stretching = np.dot(K**2, q**2)
    nonlinear_force = 0.25 * B * K**2 * q * stretching if nonlinear else 0.0

    # Project the uniform harmonic pressure onto each mode.
    load = (
        (PRESSURE * np.sin(2.0 * np.pi * Q_F * t_nd / TAU) / (RHO * H**2 * TAU**2))
        if forcing
        else 0.0
    )
    modal_load = load * (1 - (-1) ** MODES) / (MODES * np.pi)

    acceleration = (modal_load - linear - nonlinear_force) / modal_mass
    return np.r_[v, acceleration]


# ----- MECHANICAL ENERGY -----


def energy(y):
    """Conserved nondimensional mechanical energy for unforced equations.

    y must have shape (2*N, n_samples) or a single (2*N,) state.
    """
    states = np.asarray(y, dtype=float)
    if states.ndim == 1:
        states = states[:, None]
    q, v = states[:N], states[N:]

    m = 0.5 * (1.0 + A * K**2)
    k_lin = 0.5 * C * K**4
    kinetic = 0.5 * np.sum(m[:, None] * v**2, axis=0)
    bending = 0.5 * np.sum(k_lin[:, None] * q**2, axis=0)
    stretching = (B / 16.0) * np.sum(K[:, None] ** 2 * q**2, axis=0) ** 2
    return kinetic + bending + stretching


# ----- SHORT-TIME REFERENCE SIMULATION -----


def run_paper_reference(
    t_end_nd=0.08,
    dt_nd=0.00008,
    rtol=1e-10,
    atol=1e-12,
    initial_state=None,
    forcing=True,
    nonlinear=True,
):
    """Short-time, well-resolved nondimensional reference simulation.

    This intentionally does not implement the long-time published benchmark.
    The default grid resolves the literal forcing period with >20 samples.
    """
    # Check the requested time grid and initial state before integration.
    if t_end_nd <= 0 or dt_nd <= 0 or rtol <= 0 or atol <= 0:
        raise ValueError("Positive time, sample interval and tolerances required")
    samples = max(1, int(round(t_end_nd / dt_nd)))
    if samples > 1_000_000:
        raise ValueError("Too many samples requested for a short-time reference")
    time = np.linspace(0.0, t_end_nd, samples + 1)
    state = Y0.copy() if initial_state is None else np.asarray(initial_state, float)
    if state.shape != (2 * N,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must have 14 finite entries")

    # Integrate with DOP853; limit the step to resolve the load when forcing is on.
    rhs = lambda t, y: equations(t, y, forcing=forcing, nonlinear=nonlinear)
    solution = solve_ivp(
        rhs,
        (0.0, t_end_nd),
        state,
        method="DOP853",
        t_eval=time,
        rtol=rtol,
        atol=atol,
        max_step=FORCING_PERIOD_ND / 15 if forcing else np.inf,
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"Nondimensional integration failed: {solution.message}")

    # Reconstruct displacement at x/L = 0.65 from the modal histories.
    q = solution.y[:N]
    observed = PHI @ q
    return {
        "time_nd": solution.t,
        "q_nd": q,
        "v_nd": solution.y[N:],
        "w_nd": observed,
        "peak_nd": float(np.max(np.abs(observed))),
        "mode_max_nd": np.max(np.abs(q), axis=1),
        "nfev": solution.nfev,
    }


# ----- RUN THE REFERENCE -----

# From the project root: python beam_model/verification/paper_benchmark.py

if __name__ == "__main__":
    result = run_paper_reference()
    print("Independent nondimensional formulation — short-time diagnostic only")
    print(f"tau = {TAU:.10g} 1/s, forcing period = {FORCING_PERIOD_ND:.10g} nondim")
    print(
        f"max Eq. (42) coefficient discrepancy: {np.max(np.abs(PHI-PUBLISHED_PHI)):.4e}"
    )
    print(f"short-time window 0..{result['time_nd'][-1]:.5f} (nondimensional)")
    print(f"short-time peak |w_nd| = {result['peak_nd']:.8g}")
    print("These short-time values must NOT be compared to published t~1000 maxima.")
