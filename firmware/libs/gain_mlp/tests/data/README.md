# Golden fixtures

`GAINMLP.JSN` (network, same format as the SD-card file) and `GOLDEN.JSN`
(JAX-evaluated input/output cases) are generated from the wmr-simulator repo.
The checked-in fixtures use the `problems/pololu_gains.yaml` parametrization
config with a seeded random weight delta (an untrained network would only
produce identity factors), generated with:

```python
# run inside wmr-simulator: JAX_PLATFORMS=cpu uv run python <this snippet>
import numpy as np, jax.numpy as jnp, yaml
from wmr_simulator.gain_parametrization import params_from_cfg
from wmr_simulator.gain_parametrization.error_mlp import with_flat_params
from wmr_simulator.pololu.gain_mlp_exporter import (
    export_gain_mlp, export_gain_mlp_golden, firmware_base_gains)

problem = yaml.safe_load(open("problems/pololu_gains.yaml"))
robot = problem["robot"]
cfg = problem["controller"]["gain_parametrization"]
template = params_from_cfg(cfg, [robot["v_max"], robot["omega_max"]])
rng = np.random.default_rng(42)
num_weights = sum(int(np.asarray(l).size) for l in template.init_layers)
params = with_flat_params(jnp.asarray(0.3 * rng.standard_normal(num_weights).astype(np.float32)), template)
base = firmware_base_gains(problem["controller"]["gains"], robot["max_wheel_speed"])
export_gain_mlp("tests/data/GAINMLP.JSN", params)
export_gain_mlp_golden("tests/data/GOLDEN.JSN", params, base, num_cases=32, seed=0)
```

For a *trained* network, use the CLI instead:

```
uv run python scripts/export_gain_mlp.py GAINMLP.JSN \
    --problem problems/pololu_gains.yaml --tuned models/tuned_gains.yaml \
    --golden GOLDEN.JSN
```
