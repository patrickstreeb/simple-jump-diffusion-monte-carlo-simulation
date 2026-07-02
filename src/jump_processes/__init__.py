"""Jump-diffusion process simulator.

``JumpDiffusion`` bundles drift/diffusion/intensity/jump ingredients and
simulates sample paths (Euler-Maruyama + per-step jump thinning), with a
reference pure diffusion on the same Brownian path.  ``PRESETS`` / the
``preset_*`` factories build the named families from the course notes.
"""
from .jump_diffusion import (
    JumpDiffusion,
    SimResult,
    PRESETS,
    preset_brownian_additive,
    preset_merton,
    preset_inventory,
    preset_midprice,
    preset_ou_reset,
)

__all__ = [
    "JumpDiffusion", "SimResult", "PRESETS",
    "preset_brownian_additive", "preset_merton", "preset_inventory",
    "preset_midprice", "preset_ou_reset",
]
