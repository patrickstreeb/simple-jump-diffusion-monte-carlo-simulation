
"""Tests for jump_diffusion.py, run with `python tests.py`."""

import numpy as np
from jump_diffusion import JumpDiffusion

rng = np.random.default_rng

# Test 1: Brownian motion with drift and jumps
Y = lambda r: (1 if r.random() < 0.5 else -1) * r.uniform(0.5, 2.0)
JF = JumpDiffusion(mu=lambda x: 0.3, sigma=lambda x: 1.0, intensity=lambda x: 3.0,
                   rho=lambda x, y: x + y, jump=Y, x0=10.0)

s = JF.simulate(T=5.0, n=2000, rng=rng(1))
assert s.t.shape == s.J.shape == s.X.shape == s.C.shape == (2001,)
assert s.t[0] == 0.0 and s.t[-1] == 5.0 and np.allclose(np.diff(s.t), 5.0 / 2000)
assert s.J[0] == s.X[0] == 10.0

steps = np.diff(s.C)
assert np.all((steps == 0) | (steps == 1))
assert s.C[0] == 0 and s.C[-1] == s.n_jumps
assert len(s.jump_t) == len(s.jump_pre) == len(s.jump_post) == s.n_jumps
assert np.all(s.jump_t > 0) and np.all(s.jump_t <= 5.0) and np.all(np.isin(s.jump_t, s.t))

sizes = s.jump_post - s.jump_pre
assert np.all((np.abs(sizes) >= 0.5) & (np.abs(sizes) <= 2.0))

# jump statistics match the raw jump data
assert np.array_equal(s.jump_sizes(), sizes)
assert s.first_jump() == (s.jump_t[0], sizes[0]) and s.last_jump() == (s.jump_t[-1], sizes[-1])
largest, smallest = s.largest_jump(), s.smallest_jump()
assert largest is not None and smallest is not None
assert abs(largest) == np.abs(sizes).max() and abs(smallest) == np.abs(sizes).min()
assert s.average_jump() == np.abs(sizes).mean()
half = s.jump_t <= 2.5
assert np.array_equal(s.jump_sizes(t_max=2.5), sizes[half])

a = JF.simulate(T=1.0, n=300, rng=rng(7))
b = JF.simulate(T=1.0, n=300, rng=rng(7))
c = JF.simulate(T=1.0, n=300, rng=rng(8))
assert np.array_equal(a.J, b.J) and not np.array_equal(a.J, c.J)

for T, n in [(0.0, 100), (-1.0, 100), (1.0, 0)]:
    try:
        JF.simulate(T=T, n=n, rng=rng())
        assert False, "expected ValueError"
    except ValueError:
        pass
print("example 1 ok: shapes, grid, counting process, jump sizes, reproducibility")

# Test 2: limiting cases, zero intensity and deterministic drift
diff = JumpDiffusion(lambda x: 0.3, lambda x: 1.0, lambda x: 0.0,
                     lambda x, y: x + y, Y, x0=10.0)
s = diff.simulate(T=1.0, n=500, rng=rng(0))
assert s.n_jumps == 0 and np.all(s.C == 0) and np.array_equal(s.J, s.X)
assert s.first_jump() is None and s.last_jump() is None
assert s.largest_jump() is None and s.smallest_jump() is None and s.average_jump() is None

ode = JumpDiffusion(lambda x: 2.0, lambda x: 0.0, lambda x: 0.0,
                    lambda x, y: x + y, Y, x0=1.0)
s = ode.simulate(T=1.0, n=400, rng=rng(0))
assert np.allclose(s.J, 1.0 + 2.0 * s.t)
print("example 2 ok: zero intensity and deterministic drift limits")

# Test 3: other jump types, reset to zero and identity
reset = JumpDiffusion(lambda x: 0.0, lambda x: 1.0, lambda x: 3.0,
                      lambda x, y: y, lambda r: 0.0, x0=5.0)
s = reset.simulate(T=5.0, n=2000, rng=rng(2))
assert s.n_jumps > 0 and np.all(s.jump_post == 0.0)
identity = JumpDiffusion(lambda x: 0.3, lambda x: 1.0, lambda x: 3.0,
                         lambda x, y: x, Y, x0=10.0)
s = identity.simulate(T=5.0, n=2000, rng=rng(1))
assert s.n_jumps > 0 and np.array_equal(s.J, s.X)
print("example 3 ok: reset jumps and identity jumps")


# Test 4: Monte Carlo Simulation
paths = JF.simulate_many(T=1.0, n=200, n_paths=1500, rng=rng(42), progress=False)
assert len(paths) == 1500 and all(isinstance(p, JumpDiffusion.SimResult) for p in paths)
counts = np.array([p.n_jumps for p in paths])
expected = 200 * (1 - np.exp(-3.0 / 200))               # close to h*T = 3
assert abs(counts.mean() - expected) < 0.2
sizes = np.concatenate([p.jump_post - p.jump_pre for p in paths])
assert np.all((np.abs(sizes) >= 0.5) & (np.abs(sizes) <= 2.0))
assert abs(np.mean(sizes > 0) - 0.5) < 0.03             # signs are symmetric
assert abs(np.abs(sizes).mean() - 1.25) < 0.03          # E|Y| of U[0.5, 2]
J_T = np.array([p.J[-1] for p in paths])
assert abs(J_T.mean() - 10.3) < 0.26                    # E[J_T] = x0 + mu*T
assert abs(J_T.var(ddof=1) - 6.25) < 0.9                # sigma^2*T + h*T*E[Y^2]

# monte_carlo reproduces the estimate from the raw terminal values
est, se = JF.monte_carlo(paths)
assert est == J_T.mean() and se == J_T.std(ddof=1) / np.sqrt(len(J_T))
est0, se0 = JF.monte_carlo(paths, t=0.0)                # at t = 0 every path is x0
assert est0 == 10.0 and se0 == 0.0
est2, _ = JF.monte_carlo(paths, f=lambda x: x**2)       # custom f
assert est2 == (J_T**2).mean()

# higher intensity produces more jumps
lo = JumpDiffusion(lambda x: 0.3, lambda x: 1.0, lambda x: 1.0,
                   lambda x, y: x + y, Y, x0=10.0)
hi = JumpDiffusion(lambda x: 0.3, lambda x: 1.0, lambda x: 5.0,
                   lambda x, y: x + y, Y, x0=10.0)
lo_counts = [p.n_jumps for p in lo.simulate_many(1.0, 100, 200, rng(5), progress=False)]
hi_counts = [p.n_jumps for p in hi.simulate_many(1.0, 100, 200, rng(5), progress=False)]
assert np.mean(lo_counts) < np.mean(hi_counts)
print("example 4 ok: jump counts, jump distribution and moments")

print()
print("all tests passed")
