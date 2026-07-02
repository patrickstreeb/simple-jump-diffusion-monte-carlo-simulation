# jump_processes

Jump-diffusion process simulator (independent library).

```python
import numpy as np
from jump_processes import preset_merton
res = preset_merton(sigma=0.2, lam=5.0, x0=100.0).simulate(
    T=1.0, n_steps=500, rng=np.random.default_rng(0))
print(res.n_jumps, res.J[-1])
```

Install: `pip install -e .[dev]`  ·  Test: `pytest`
