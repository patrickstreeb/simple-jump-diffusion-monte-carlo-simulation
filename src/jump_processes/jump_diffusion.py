"""
Jump-diffusion processes for the Market-Making applet suite.

Reimplements (and tidies up) the simulator from the course notes
``0-7-jump-diffusions.html`` (Borodin 2017, Ch. VI).  A *diffusion with
jumps* solves

    dJ_t = mu(J_t) dt + sigma(J_t) dW_t + (rho(J_{t-}, Y) - J_{t-}) dC_t,

where:
  * ``mu``        - drift coefficient mu(x),
  * ``sigma``     - diffusion coefficient sigma(x) (>= 0),
  * ``intensity`` - the (possibly state dependent) jump rate h(x) >= 0,
  * ``rho``       - the jump map rho(x, y) sending the pre-jump state and an
                    innovation y to the post-jump state,
  * ``jump``      - the innovation sampler  rng -> Y,
  * ``C_t``       - the counting process whose rate is h(J_t).

``JumpDiffusion`` bundles those ingredients and simulates sample paths by
Euler-Maruyama with per-step jump thinning (a jump occurs in [t, t+dt) with
probability ``h(J_t) dt``).  The same Brownian increments drive a reference
*pure diffusion* ``X_t`` (no jumps) so the two can be compared on one plot.

Factory helpers build the named families discussed in the notes:
``brownian_additive``, ``merton``, ``inventory``, ``midprice`` and
``ou_reset``.  ``JumpDiffusion.from_choices`` mirrors the drop-downs of the
original web applet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# Type aliases -------------------------------------------------------------
Drift = Callable[[float], float]          # mu(x)
Diffusion = Callable[[float], float]      # sigma(x)
Intensity = Callable[[float], float]      # h(x)
JumpMap = Callable[[float, float], float] # rho(x, y)
JumpSampler = Callable[[np.random.Generator], float]  # rng -> Y


# --- building blocks: drift mu(x) ----------------------------------------
def drift_zero() -> Drift:
    return lambda x: 0.0


def drift_constant(c: float = 0.5) -> Drift:
    return lambda x: c


def drift_ou(theta: float = 0.5) -> Drift:
    """Mean-reverting drift mu(x) = -theta * x."""
    return lambda x: -theta * x


# --- building blocks: intensity h(x) -------------------------------------
def intensity_constant(lam: float = 3.0) -> Intensity:
    return lambda x: lam


def intensity_linear(lam: float = 3.0, scale: float = 0.1) -> Intensity:
    """Rate grows with distance from zero: h(x) = lam * scale * |x|."""
    return lambda x: lam * scale * abs(x)


def intensity_exponential(A: float = 3.0, k: float = 0.3) -> Intensity:
    """Fill-rate style intensity h(x) = A * exp(-k|x|)."""
    return lambda x: A * np.exp(-k * abs(x))


# --- building blocks: jump map rho(x, y) ---------------------------------
def rho_additive(x: float, y: float) -> float:
    return x + y


def rho_multiplicative(x: float, y: float) -> float:
    return x * y


def rho_reset(x: float, y: float) -> float:
    return y


# --- building blocks: innovation samplers --------------------------------
def jump_signed_uniform(lo: float = 0.5, hi: float = 2.0) -> JumpSampler:
    """+/- 1 times a Uniform(lo, hi) magnitude (additive jumps)."""
    return lambda rng: (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(lo, hi)


def jump_uniform(lo: float = 0.85, hi: float = 1.15) -> JumpSampler:
    """Multiplicative factor ~ Uniform(lo, hi)."""
    return lambda rng: rng.uniform(lo, hi)


def jump_lognormal(m: float = -0.05, s: float = 0.1) -> JumpSampler:
    """Multiplicative factor e^Y with Y ~ Normal(m, s) (Merton)."""
    return lambda rng: float(np.exp(rng.normal(m, s)))


def jump_two_sided(up: float, down: float, p_up: float) -> JumpSampler:
    """Returns +up w.p. p_up, else -down (two competing Poisson streams)."""
    return lambda rng: up if rng.random() < p_up else -down


def jump_reset_level(x0: float) -> JumpSampler:
    return lambda rng: x0 * rng.uniform(0.5, 1.5)


# --- simulation result ----------------------------------------------------
@dataclass
class SimResult:
    t: np.ndarray             # time grid, shape (n+1,)
    J: np.ndarray             # jump-diffusion path, shape (n+1,)
    X: np.ndarray             # reference pure diffusion (same dW), shape (n+1,)
    C: np.ndarray             # counting process C_t, shape (n+1,)
    jump_t: np.ndarray        # jump times
    jump_pre: np.ndarray      # value just before each jump
    jump_post: np.ndarray     # value just after each jump
    label: str = "jump diffusion"

    @property
    def n_jumps(self) -> int:
        return int(self.jump_t.size)

    def plot(self, ax=None, show_diffusion: bool = True, color: str = "#d97706"):
        """Plot the path (and, optionally, the reference diffusion + jumps)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        if show_diffusion:
            ax.plot(self.t, self.X, color="#93c5fd", lw=1.0, alpha=0.7,
                    label=r"pure diffusion $X_t$")
        ax.plot(self.t, self.J, color=color, lw=1.6, label=r"jump diffusion $J_t$")
        for k in range(self.n_jumps):
            ax.plot([self.jump_t[k], self.jump_t[k]],
                    [self.jump_pre[k], self.jump_post[k]],
                    color="#ef4444", lw=1.2, ls="--")
            ax.plot(self.jump_t[k], self.jump_post[k], "o", color="#ef4444", ms=3.5)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$J_t$")
        ax.margins(x=0)
        return ax


# --- the model ------------------------------------------------------------
class JumpDiffusion:
    """A diffusion with state-dependent jumps; see module docstring."""

    def __init__(
        self,
        mu: Drift,
        sigma,                       # float or Diffusion callable
        intensity: Intensity,
        rho: JumpMap,
        jump: JumpSampler,
        x0: float = 10.0,
        label: str = "jump diffusion",
    ):
        self.mu = mu
        self.sigma = sigma if callable(sigma) else (lambda x, s=float(sigma): s)
        self.intensity = intensity
        self.rho = rho
        self.jump = jump
        self.x0 = float(x0)
        self.label = label

    def simulate(self, T: float = 5.0, n_steps: int = 2000,
                 rng: Optional[np.random.Generator] = None) -> SimResult:
        """Simulate one sample path on [0, T] with ``n_steps`` Euler steps."""
        if rng is None:
            rng = np.random.default_rng()
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        t = np.linspace(0.0, T, n_steps + 1)
        J = np.empty(n_steps + 1)
        X = np.empty(n_steps + 1)
        C = np.zeros(n_steps + 1)
        J[0] = X[0] = self.x0
        dW = rng.standard_normal(n_steps) * sqrt_dt

        jt: List[float] = []
        jpre: List[float] = []
        jpost: List[float] = []

        for i in range(n_steps):
            # reference pure diffusion (shares the Brownian increment)
            X[i + 1] = X[i] + self.mu(X[i]) * dt + self.sigma(X[i]) * dW[i]
            # jump diffusion: continuous (diffusion) step ...
            j = J[i] + self.mu(J[i]) * dt + self.sigma(J[i]) * dW[i]
            C[i + 1] = C[i]
            # ... then a possible jump, thinned at rate h(J_i)
            rate = max(self.intensity(J[i]), 0.0)
            if rng.random() < rate * dt:
                pre = j
                j = self.rho(j, self.jump(rng))
                jt.append(t[i + 1]); jpre.append(pre); jpost.append(j)
                C[i + 1] = C[i] + 1
            J[i + 1] = j

        return SimResult(
            t=t, J=J, X=X, C=C,
            jump_t=np.asarray(jt), jump_pre=np.asarray(jpre),
            jump_post=np.asarray(jpost), label=self.label,
        )

    def simulate_many(self, T: float = 5.0, n_steps: int = 2000, n_paths: int = 10,
                      rng: Optional[np.random.Generator] = None) -> List[SimResult]:
        if rng is None:
            rng = np.random.default_rng()
        return [self.simulate(T, n_steps, rng) for _ in range(n_paths)]

    # --- applet-style generic constructor --------------------------------
    @classmethod
    def from_choices(cls, rho: str = "additive", h: str = "constant",
                     mu: str = "zero", sigma: float = 1.0, lam: float = 3.0,
                     x0: float = 10.0) -> "JumpDiffusion":
        """Build a model from the original applet's drop-down choices."""
        drift = {"zero": drift_zero(), "constant": drift_constant(),
                 "ou": drift_ou()}[mu]
        inten = {"constant": intensity_constant(lam),
                 "linear": intensity_linear(lam),
                 "exponential": intensity_exponential(lam)}[h]
        rho_fn = {"additive": rho_additive, "multiplicative": rho_multiplicative,
                  "reset": rho_reset}[rho]
        sampler = {"additive": jump_signed_uniform(),
                   "multiplicative": jump_uniform(),
                   "reset": jump_reset_level(x0)}[rho]
        return cls(drift, sigma, inten, rho_fn, sampler, x0=x0,
                   label=f"rho={rho}, h={h}, mu={mu}")


# --- named presets (the worked examples from the notes) ------------------
def preset_brownian_additive(sigma: float = 1.0, lam: float = 3.0,
                             x0: float = 10.0) -> JumpDiffusion:
    """BM with constant drift + additive Poisson jumps: dJ = mu dt + sigma dW + Y dC."""
    return JumpDiffusion(drift_constant(0.3), sigma, intensity_constant(lam),
                         rho_additive, jump_signed_uniform(), x0=x0,
                         label="Brownian + additive Poisson jumps")


def preset_merton(sigma: float = 0.3, lam: float = 3.0,
                  x0: float = 100.0) -> JumpDiffusion:
    """Geometric (Merton) jump diffusion: dS = mu S dt + sigma S dW + S(e^Y-1) dC."""
    return JumpDiffusion(lambda x: 0.05 * x, lambda x: sigma * x,
                         intensity_constant(lam), rho_multiplicative,
                         jump_lognormal(-0.05, 0.12), x0=x0,
                         label="Merton geometric jump diffusion")


def preset_inventory(sigma: float = 1.0, lam: float = 3.0,
                     x0: float = 0.0) -> JumpDiffusion:
    """Pure-jump market-maker inventory q_t = N^b - N^a (+/- 1 fills)."""
    return JumpDiffusion(drift_zero(), 0.0, intensity_constant(2 * lam),
                         rho_additive, jump_two_sided(1.0, 1.0, 0.5), x0=x0,
                         label="Pure-jump inventory (+/-1)")


def preset_midprice(sigma: float = 1.0, lam: float = 3.0, x0: float = 100.0,
                    kappa: float = 0.5) -> JumpDiffusion:
    """Mid-price with market impact: dS = sigma dW + kappa(dN^b - dN^a)."""
    return JumpDiffusion(drift_zero(), sigma, intensity_constant(2 * lam),
                         rho_additive, jump_two_sided(kappa, kappa, 0.5), x0=x0,
                         label="Mid-price with market impact (+/-kappa)")


def preset_ou_reset(sigma: float = 1.0, lam: float = 3.0, x0: float = 5.0,
                    theta: float = 0.5) -> JumpDiffusion:
    """Ornstein-Uhlenbeck with reset-to-zero jumps at rate h(x) = lam|x|."""
    return JumpDiffusion(drift_ou(theta), sigma, intensity_linear(lam),
                         rho_reset, lambda rng: 0.0, x0=x0,
                         label="OU with reset-to-zero jumps")


# name -> (human label, factory(sigma, lam, x0)) for the UI
PRESETS = {
    "brownian_additive": ("Brownian + additive Poisson jumps", preset_brownian_additive),
    "merton": ("Merton geometric jump diffusion", preset_merton),
    "inventory": ("Pure-jump inventory (+/-1)", preset_inventory),
    "midprice": ("Mid-price with market impact (+/-kappa)", preset_midprice),
    "ou_reset": ("OU with reset-to-zero jumps", preset_ou_reset),
}
