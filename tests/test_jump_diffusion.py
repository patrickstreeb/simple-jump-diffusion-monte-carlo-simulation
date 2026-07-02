import numpy as np
from jump_processes import JumpDiffusion, SimResult, PRESETS, preset_brownian_additive


def test_presets_dict():
    assert isinstance(PRESETS, dict) and len(PRESETS) >= 1


def test_no_jumps_when_intensity_zero():
    jd = preset_brownian_additive(sigma=1.0, lam=0.0, x0=0.0)
    res = jd.simulate(T=1.0, n_steps=100, rng=np.random.default_rng(0))
    assert isinstance(res, SimResult)
    assert res.n_jumps == 0
    assert res.J.shape == (101,) and res.t.shape == (101,)


def test_simulate_is_deterministic_with_seed():
    jd = preset_brownian_additive(sigma=1.0, lam=5.0, x0=0.0)
    a = jd.simulate(T=1.0, n_steps=200, rng=np.random.default_rng(42))
    b = jd.simulate(T=1.0, n_steps=200, rng=np.random.default_rng(42))
    assert np.allclose(a.J, b.J) and a.n_jumps == b.n_jumps
