"""Strategic equilibrium and welfare analysis for the manipulator-detector game.

Implements:
    * Manipulator best-response (interior solution of the two first-order
      conditions), with verified second-order conditions.
    * Comparative statics of (r*, c*) with respect to detection sensitivity
      (alpha, beta) and threshold tau.
    * Private-vs-social threshold wedge.
    * Deterrence curves used in Figure 3 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm


@dataclass
class GameParams:
    Q: float = 100.0          # manipulation quantity
    delta: float = 1.0        # price impact of rush orders
    gamma: float = 0.8        # price impact of cancellations
    k: float = 120.0          # quadratic cost on r  (calibrated for interior eqm)
    ell: float = 100.0        # quadratic cost on c  (calibrated for interior eqm)
    L: float = 200.0          # detection penalty
    alpha: float = 1.0        # rush-order detection sensitivity (weight)
    beta: float = 0.8         # cancellation detection sensitivity (weight)
    tau: float = 1.0          # detection threshold
    sigma: float = 0.5        # market noise
    # Welfare parameters
    H: float = 250.0          # social benefit of correct detection
    F: float = 40.0           # social cost of false alarm
    xi: float = 60.0          # marginal social damage from rush-order manipulation
    zeta: float = 45.0        # marginal social damage from cancellation manipulation
    cD: float = 3.0           # per-unit monitoring cost (convex detection tech)


# ---------------------------------------------------------------------------
# Manipulator's problem
# ---------------------------------------------------------------------------
def manipulator_profit(r: float, c: float, p: GameParams) -> float:
    if r < 0 or c < -1e-9 or c > 1 + 1e-9:
        return -1e12
    s = p.alpha * r + p.beta * c
    pdet = norm.cdf((s - p.tau) / p.sigma)
    revenue = (p.delta * r + p.gamma * c) * p.Q
    costs = 0.5 * (p.k * r ** 2 + p.ell * c ** 2)
    penalty = p.L * pdet
    return revenue - costs - penalty


def best_response(p: GameParams, x0: Tuple[float, float] = (1.0, 0.5)) -> Tuple[float, float, float]:
    res = minimize(
        lambda s: -manipulator_profit(s[0], s[1], p),
        x0=np.array(x0),
        bounds=[(0.0, 5.0), (0.0, 1.0)],
        method="L-BFGS-B",
    )
    if not res.success:
        raise RuntimeError(f"best_response failed: {res.message}")
    r_star, c_star = float(res.x[0]), float(res.x[1])
    return r_star, c_star, -float(res.fun)


def focs(r: float, c: float, p: GameParams) -> Tuple[float, float]:
    """Return the two first-order conditions evaluated at (r, c)."""
    s = p.alpha * r + p.beta * c
    phi = norm.pdf((s - p.tau) / p.sigma)
    foc_r = p.delta * p.Q - p.k * r - (p.L * p.alpha / p.sigma) * phi
    foc_c = p.gamma * p.Q - p.ell * c - (p.L * p.beta / p.sigma) * phi
    return foc_r, foc_c


def hessian(r: float, c: float, p: GameParams) -> np.ndarray:
    """Hessian of the manipulator's objective.

    H = [[ -k - (L alpha^2 / sigma^2) phi'(z),  -(L alpha beta / sigma^2) phi'(z) ],
         [ -(L alpha beta / sigma^2) phi'(z), -ell - (L beta^2 / sigma^2) phi'(z) ]]

    where z = (s - tau)/sigma, phi' has sign depending on z.
    Negative definite implies strict concavity at (r, c).
    """
    s = p.alpha * r + p.beta * c
    z = (s - p.tau) / p.sigma
    phi = norm.pdf(z)
    phi_prime = -z * phi
    const = p.L * phi_prime / (p.sigma ** 2)
    Hm = np.array([
        [-p.k - const * p.alpha ** 2, -const * p.alpha * p.beta],
        [-const * p.alpha * p.beta, -p.ell - const * p.beta ** 2],
    ])
    return Hm


# ---------------------------------------------------------------------------
# Comparative statics (implicit function theorem)
# ---------------------------------------------------------------------------
def comparative_statics(p: GameParams) -> Dict[str, float]:
    """Return partial derivatives of the equilibrium (r*, c*) wrt (alpha, beta, tau).

    Uses the Hessian of the manipulator's FOC with respect to (r, c) to apply
    the implicit function theorem:
         d(FOC)/d(x*) @ [dr*/dtheta; dc*/dtheta] = - d(FOC)/dtheta.
    """
    r_star, c_star, _ = best_response(p)
    s = p.alpha * r_star + p.beta * c_star
    z = (s - p.tau) / p.sigma
    phi = norm.pdf(z)
    phi_p = -z * phi

    # Jacobian of FOCs wrt (r, c)
    J = np.array([
        [-p.k - (p.L * p.alpha ** 2 / p.sigma ** 2) * phi_p, -(p.L * p.alpha * p.beta / p.sigma ** 2) * phi_p],
        [-(p.L * p.alpha * p.beta / p.sigma ** 2) * phi_p, -p.ell - (p.L * p.beta ** 2 / p.sigma ** 2) * phi_p],
    ])

    # Partial derivatives of FOCs wrt parameter theta
    # FOC_r = delta Q - k r - (L alpha / sigma) phi(z)
    # dFOC_r/dalpha = -(L/sigma) phi - (L alpha / sigma) phi'(z) * (r/sigma)
    def dfoc_dalpha():
        dfr = -(p.L / p.sigma) * phi - (p.L * p.alpha / p.sigma) * phi_p * (r_star / p.sigma)
        dfc = -(p.L * p.beta / p.sigma) * phi_p * (r_star / p.sigma)
        return np.array([dfr, dfc])

    def dfoc_dbeta():
        dfr = -(p.L * p.alpha / p.sigma) * phi_p * (c_star / p.sigma)
        dfc = -(p.L / p.sigma) * phi - (p.L * p.beta / p.sigma) * phi_p * (c_star / p.sigma)
        return np.array([dfr, dfc])

    def dfoc_dtau():
        dfr = -(p.L * p.alpha / p.sigma) * phi_p * (-1.0 / p.sigma)
        dfc = -(p.L * p.beta / p.sigma) * phi_p * (-1.0 / p.sigma)
        return np.array([dfr, dfc])

    out = {}
    for name, partial in [
        ("alpha", dfoc_dalpha()), ("beta", dfoc_dbeta()), ("tau", dfoc_dtau()),
    ]:
        dxs = -np.linalg.solve(J, partial)
        out[f"dr_d{name}"] = float(dxs[0])
        out[f"dc_d{name}"] = float(dxs[1])
    out["r_star"] = r_star
    out["c_star"] = c_star
    return out


# ---------------------------------------------------------------------------
# Welfare: private vs social threshold
# ---------------------------------------------------------------------------
_PI1_DEFAULT = 0.1  # prior probability of manipulation episode


def _monitoring_cost(tau: float, p: GameParams) -> float:
    """Convex monitoring cost increasing as tau falls (more intense surveillance).

    A smooth convex form: cD * exp(-tau). This makes lowering the threshold
    progressively more expensive without the singular 1/tau behaviour.
    """
    return p.cD * np.exp(-tau)


def private_welfare(tau: float, p: GameParams, pi1: float = _PI1_DEFAULT) -> float:
    """Private detector maximises expected penalty revenue net of monitoring cost."""
    q = GameParams(**{**p.__dict__, "tau": tau})
    r_star, c_star, _ = best_response(q)
    s = q.alpha * r_star + q.beta * c_star
    pdet_mani = norm.cdf((s - tau) / q.sigma)
    pdet_null = norm.cdf((0.0 - tau) / q.sigma)
    rev = pi1 * q.L * pdet_mani - (1 - pi1) * q.F * pdet_null
    return rev - _monitoring_cost(tau, q)


def social_welfare(tau: float, p: GameParams, pi1: float = _PI1_DEFAULT) -> float:
    """Social planner internalises market-quality externality."""
    q = GameParams(**{**p.__dict__, "tau": tau})
    r_star, c_star, _ = best_response(q)
    s = q.alpha * r_star + q.beta * c_star
    pdet_mani = norm.cdf((s - tau) / q.sigma)
    pdet_null = norm.cdf((0.0 - tau) / q.sigma)
    benefit = pi1 * q.H * pdet_mani - (1 - pi1) * q.F * pdet_null
    externality = -pi1 * (q.xi * r_star + q.zeta * c_star)
    return benefit + externality - _monitoring_cost(tau, q)


def optimise_threshold(objective, p: GameParams, grid: np.ndarray | None = None) -> Tuple[float, float]:
    if grid is None:
        grid = np.linspace(-2.0, 4.0, 121)
    vals = np.array([objective(t, p) for t in grid])
    idx = int(np.argmax(vals))
    return float(grid[idx]), float(vals[idx])


# ---------------------------------------------------------------------------
# Deterrence sweeps
# ---------------------------------------------------------------------------
def deterrence_sweep(p: GameParams, alpha_grid: np.ndarray, keep: str = "alpha") -> Dict[str, np.ndarray]:
    """Vary alpha (or beta) and return (alpha, r*, c*)."""
    rs, cs, alphas = [], [], []
    for a in alpha_grid:
        if keep == "alpha":
            q = GameParams(**{**p.__dict__, "alpha": float(a)})
        elif keep == "beta":
            q = GameParams(**{**p.__dict__, "beta": float(a)})
        elif keep == "both":
            q = GameParams(**{**p.__dict__, "alpha": float(a), "beta": float(a) * (p.beta / p.alpha)})
        else:
            raise ValueError(keep)
        r, c, _ = best_response(q)
        alphas.append(float(a)); rs.append(r); cs.append(c)
    return {"alpha": np.array(alphas), "r": np.array(rs), "c": np.array(cs)}


if __name__ == "__main__":
    p = GameParams()
    r, c, pi = best_response(p)
    print(f"Equilibrium: r*={r:.4f}, c*={c:.4f}, profit={pi:.4f}")
    foc_r, foc_c = focs(r, c, p)
    print(f"FOCs: d pi/dr = {foc_r:.2e}, d pi/dc = {foc_c:.2e}")
    Hm = hessian(r, c, p)
    print(f"Hessian eigenvalues: {np.linalg.eigvalsh(Hm)}")
    cs = comparative_statics(p)
    print("Comparative statics:", cs)
    t_priv, v_priv = optimise_threshold(private_welfare, p)
    t_soc, v_soc = optimise_threshold(social_welfare, p)
    print(f"Private optimal tau = {t_priv:.3f} (W={v_priv:.3f})")
    print(f"Social optimal  tau = {t_soc:.3f} (W={v_soc:.3f})")
