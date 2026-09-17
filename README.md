# finance-rl

A Gymnasium environment modeling a corporate "CFO agent" that allocates
capital over a multi-year horizon to maximize long-run enterprise value
while avoiding insolvency and debt-covenant breaches.

## Install

```bash
pip install -e .
```

## Usage

```python
from finance_rl import CorporateFinanceEnv, CFOEnvConfig

env = CorporateFinanceEnv(config=CFOEnvConfig(horizon_years=20))
obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # [capex_pct, debt_action, opex_adjustment] in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
```

See `src/finance_rl/cfo_env.py` for the observation/action space layout,
transition dynamics and reward function, and `tests/test_cfo_env.py` for
runnable examples.

## Tests

```bash
pip install -e . pytest
pytest tests/
```
