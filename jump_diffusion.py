
"""Jump-diffusion model and its Monte Carlo simulation."""

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# The model parameters may depend on the current state of the process, so all of
# them are passed as callable functions evaluated at the state x (rho also at the
# jump value y) rather than as fixed numbers

Num = float | np.ndarray                                       # scalar state or vector of states
Drift = Callable[[Num], Num]                                   # mu(x)
Diffusion = Callable[[Num], Num]                               # sigma(x)
Intensity = Callable[[Num], Num]                               # h(x)
JumpType = Callable[[Num, Num], Num]                           # rho(x, y)
JumpSizeDistribution = Callable[[np.random.Generator], float]  # rng -> Y


class JumpDiffusion:

    def __init__(self, mu: Drift, sigma: Diffusion, intensity: Intensity,
                 rho: JumpType, jump: JumpSizeDistribution, x0: float = 10.0):
        self.mu = mu
        self.sigma = sigma
        self.intensity = intensity
        self.rho = rho
        self.jump = jump
        self.x0 = float(x0)

    @staticmethod
    def brownian_motion(n, T, rng):
        dt = T / n
        dW = np.sqrt(dt) * rng.standard_normal(n)
        return dW

    def simulate(self, T, n, rng) -> "JumpDiffusion.SimResult":
        if T <= 0 or n < 1:
            raise ValueError("need T > 0 and n >= 1")
        dt = T / n
        t = np.linspace(0.0, T, n + 1)
        J = np.empty(n + 1)
        X = np.empty(n + 1)
        C = np.zeros(n + 1)
        J[0] = X[0] = self.x0
        dW = self.brownian_motion(n, T, rng)
        jump_times = []
        value_before_jump = []
        value_after_jump = []
        for i in range(n):
            X[i + 1] = X[i] + self.mu(X[i]) * dt + self.sigma(X[i]) * dW[i]
            j = J[i] + self.mu(J[i]) * dt + self.sigma(J[i]) * dW[i]
            C[i + 1] = C[i]
            if rng.random() < 1 - np.exp(- self.intensity(J[i]) * dt):
                pre = j
                j = self.rho(j, self.jump(rng))
                jump_times.append(t[i + 1])
                value_before_jump.append(pre)
                value_after_jump.append(j)
                C[i + 1] = C[i] + 1
            J[i + 1] = j
        return JumpDiffusion.SimResult(t=t, J=J, X=X, C=C,
                                       jump_t=np.asarray(jump_times),
                                       jump_pre=np.asarray(value_before_jump),
                                       jump_post=np.asarray(value_after_jump),
                                       )

    def simulate_many(self, T, n, n_paths, rng,
                      progress: bool = True) -> list["JumpDiffusion.SimResult"]:
        paths = range(n_paths)
        if progress:
            paths = tqdm(paths, desc="simulating", unit=" path")
        return [self.simulate(T, n, rng) for _ in paths]

    @staticmethod
    def monte_carlo(paths, f=lambda x: x, t=None):
        # index of the evaluation time: the last grid point by default,
        # otherwise the grid point closest to t
        if t is None:
            i = -1
        else:
            grid = paths[0].t
            i = int(np.argmin(np.abs(grid - t)))

        # evaluate f at the path values at that time
        values = []
        for p in paths:
            values.append(f(p.J[i]))
        values = np.array(values)

        # sample mean and its standard error
        estimate = values.mean()
        standard_error = values.std(ddof=1) / np.sqrt(len(values))
        return estimate, standard_error



    class SimResult:

        def __init__(self, t: np.ndarray, J: np.ndarray, X: np.ndarray, C: np.ndarray,
                     jump_t: np.ndarray, jump_pre: np.ndarray, jump_post: np.ndarray):
            self.t = t                  # time grid
            self.J = J                  # jump-diffusion path
            self.X = X                  # same diffusion without jumps
            self.C = C                  # counting process
            self.jump_t = jump_t        # jump times
            self.jump_pre = jump_pre    # value just before each jump
            self.jump_post = jump_post  # value just after each jump

        @property
        def n_jumps(self) -> int:
            return int(self.jump_t.size)

        def jump_sizes(self, t_min=0.0, t_max=None) -> np.ndarray:
            t_max = self.t[-1] if t_max is None else t_max
            mask = (self.jump_t >= t_min) & (self.jump_t <= t_max)
            return self.jump_post[mask] - self.jump_pre[mask]

        def largest_jump(self, t_min=0.0, t_max=None):
            sizes = self.jump_sizes(t_min, t_max)
            return float(sizes[np.argmax(np.abs(sizes))]) if sizes.size else None

        def smallest_jump(self, t_min=0.0, t_max=None):
            sizes = self.jump_sizes(t_min, t_max)
            return float(sizes[np.argmin(np.abs(sizes))]) if sizes.size else None

        def average_jump(self, t_min=0.0, t_max=None):
            sizes = self.jump_sizes(t_min, t_max)
            return float(np.abs(sizes).mean()) if sizes.size else None

        def first_jump(self):
            if self.n_jumps == 0:
                return None
            return float(self.jump_t[0]), float(self.jump_post[0] - self.jump_pre[0])

        def last_jump(self):
            if self.n_jumps == 0:
                return None
            return float(self.jump_t[-1]), float(self.jump_post[-1] - self.jump_pre[-1])

        def combined_plot(self, ax=None, show_diffusion: bool = True):
            if ax is None:
                _, ax = plt.subplots(figsize=(9, 4))
            if show_diffusion:
                ax.plot(self.t,
                        self.X,
                        color="#93c5fd",
                        lw=1.0,
                        alpha=0.7,
                        label=r"pure diffusion $X_t$"
                        )
            ax.plot(self.t,
                    self.J,
                    color="#d97706",
                    lw=1.6,
                    label=r"jump diffusion $J_t$"
                    )
            for k in range(self.n_jumps):
                ax.plot([self.jump_t[k],
                         self.jump_t[k]],
                        [self.jump_pre[k],
                         self.jump_post[k]],
                        color="#ef4444",
                        lw=1.2,
                        ls="--"
                        )
                ax.plot(self.jump_t[k],
                        self.jump_post[k],
                        "o",
                        color="#ef4444",
                        ms=3.5
                        )
            ax.set_xlabel("t")
            ax.set_ylabel(r"$J_t$")
            ax.margins(x=0)
            ax.legend()
