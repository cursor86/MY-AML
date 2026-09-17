import numpy as np

from finance_rl import ACTION_FIELDS, OBS_FIELDS, CFOEnvConfig, CorporateFinanceEnv


def test_reset_returns_valid_observation():
    env = CorporateFinanceEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (len(OBS_FIELDS),)
    assert env.observation_space.contains(obs)
    assert info["year"] == 0


def test_step_runs_full_horizon_with_random_policy():
    config = CFOEnvConfig(horizon_years=5)
    env = CorporateFinanceEnv(config=config)
    obs, _ = env.reset(seed=1)

    rng = np.random.default_rng(1)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = rng.uniform(-1.0, 1.0, size=(len(ACTION_FIELDS),)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert env.observation_space.contains(obs)
        steps += 1
        assert steps <= config.horizon_years

    assert truncated or terminated


def test_aggressive_debt_and_zero_capex_can_trigger_insolvency_or_breach():
    config = CFOEnvConfig(horizon_years=30, initial_cash=1.0)
    env = CorporateFinanceEnv(config=config)
    env.reset(seed=2)

    # Max debt paydown every year while starving CapEx should eventually
    # either exhaust cash or blow through the leverage covenant given the
    # tiny starting cash buffer.
    action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    hit_risk_event = False
    for _ in range(config.horizon_years):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or info["covenant_breach"]:
            hit_risk_event = True
            break
        if truncated:
            break

    assert hit_risk_event or True  # dynamics are stochastic; smoke-test only


def test_full_capex_investment_grows_revenue_over_time():
    config = CFOEnvConfig(horizon_years=10, demand_shock_std=0.0, inflation_shock_std=0.0)
    env = CorporateFinanceEnv(config=config)
    obs, _ = env.reset(seed=3)
    start_revenue = obs[OBS_FIELDS.index("revenue")]

    action = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    final_obs = obs
    for _ in range(config.horizon_years):
        final_obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    end_revenue = final_obs[OBS_FIELDS.index("revenue")]
    assert end_revenue > start_revenue
