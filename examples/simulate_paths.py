"""Simulate a Merton jump-diffusion path and report the jump count."""
import numpy as np
from jump_processes import preset_merton

res = preset_merton(sigma=0.2, lam=5.0, x0=100.0).simulate(
    T=1.0, n_steps=500, rng=np.random.default_rng(0))
print("jumps:", res.n_jumps, "| final J:", round(float(res.J[-1]), 3))
