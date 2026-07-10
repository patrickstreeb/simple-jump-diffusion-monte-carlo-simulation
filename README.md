# Simple Jump Diffusion Monte Carlo Simulation

Monte Carlo simulation of jump diffusions (diffusions with jumps, cf. [1, Ch. VI]):

$$\mathrm{d}J_t = \mu(J_t)\, \mathrm{d}t + \sigma(J_t)\, \mathrm{d}W_t + \big(\rho(J_{t^-}, Y_{C_t}) - J_{t^-}\big)\, \mathrm{d}C_t,$$

where $\mu$ is the drift coefficient, $\sigma$ the diffusion coefficient, $C$ a counting process with state-dependent intensity function $h$, $\rho$ the function of jumps and $Y_k$, $k = 1, 2, \ldots$, are i.i.d. random variables which determine the values of the jumps.

## Structure

![Structure diagram](https://github.com/patrickstreeb/simple-jump-diffusion-monte-carlo-simulation/releases/download/v1.0/structure_diagram.png)

## Quick start

```python
import numpy as np
from jump_diffusion import JumpDiffusion

JF = JumpDiffusion(mu=lambda x: 0.3, sigma=lambda x: 1.0, intensity=lambda x: 3.0,
                   rho=lambda x, y: x + y,
                   jump=lambda rng: (1 if rng.random() < 0.5 else -1) * rng.uniform(0.5, 2.0),
                   x0=10.0)

sim = JF.simulate(T=5.0, n=2000, rng=np.random.default_rng(1))
sim.combined_plot()
```

## Reference

[1] Borodin, A. N. (2017). *Stochastic Processes*. Probability and Its Applications. Birkhäuser/Springer International Publishing, Cham. https://doi.org/10.1007/978-3-319-62310-8
