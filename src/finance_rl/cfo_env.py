"""Gymnasium environment modeling a corporate CFO's capital-allocation decisions.

An agent controls capital expenditure, debt financing and operating-cost
adjustments over a multi-year horizon, trying to maximize long-run
enterprise value while avoiding insolvency and debt-covenant breaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class CFOEnvConfig:
    """Tunable parameters for the environment's economics and episode length."""

    horizon_years: int = 20

    # --- initial balance-sheet / income-statement state ---
    initial_cash: float = 20.0          # $M
    initial_revenue: float = 100.0      # $M
    initial_debt: float = 40.0          # $M
    initial_equity: float = 80.0        # $M
    initial_gross_margin: float = 0.55
    initial_ebitda_margin: float = 0.18
    initial_growth_rate: float = 0.04
    initial_wacc: float = 0.09
    initial_demand_index: float = 1.0
    initial_inflation: float = 0.02

    # --- transition-dynamics sensitivities ---
    # How strongly CapEx (as a fraction of revenue) lifts next-period revenue
    # growth. Diminishing returns are applied via sqrt() in `step`.
    capex_growth_sensitivity: float = 0.35
    # Fraction of book CapEx that depreciates (drags EBITDA margin) each year.
    depreciation_rate: float = 0.15
    # How strongly incremental opex (R&D/marketing) improves gross margin,
    # again with diminishing returns.
    opex_margin_sensitivity: float = 0.08
    # Natural mean-reverting decay applied to growth/margins absent investment,
    # reflecting competitive erosion.
    organic_decay: float = 0.05

    # --- financing mechanics ---
    interest_rate_spread_over_wacc: float = -0.02  # debt is cheaper than equity
    max_debt_issue_or_paydown_pct_of_equity: float = 0.25
    tax_rate: float = 0.21

    # --- macro shocks ---
    demand_shock_std: float = 0.08
    inflation_shock_std: float = 0.01
    demand_mean_reversion: float = 0.10

    # --- risk limits ---
    max_leverage_ratio: float = 4.0     # Debt / EBITDA covenant
    insolvency_cash_floor: float = 0.0

    # --- reward shaping ---
    fcf_reward_weight: float = 1.0
    enterprise_value_reward_weight: float = 0.15
    insolvency_penalty: float = 50.0
    covenant_breach_penalty: float = 20.0

    seed_state_noise: float = 0.0  # optional randomization of the initial state


# Observation vector layout, in order. Exposed so downstream code/tests can
# index into the observation without hardcoding magic numbers.
OBS_FIELDS = (
    "cash_balance",
    "revenue",
    "growth_rate",
    "gross_margin",
    "ebitda_margin",
    "debt_to_equity",
    "wacc",
    "market_demand_index",
    "inflation_rate",
)

# Action vector layout.
ACTION_FIELDS = ("capex_pct", "debt_action", "opex_adjustment")


@dataclass
class _FinancialState:
    """Internal mutable balance-sheet / income-statement state for one episode."""

    year: int = 0
    cash: float = 0.0
    revenue: float = 0.0
    growth_rate: float = 0.0
    gross_margin: float = 0.0
    ebitda_margin: float = 0.0
    debt: float = 0.0
    equity: float = 0.0
    wacc: float = 0.0
    demand_index: float = 0.0
    inflation: float = 0.0
    prev_enterprise_value: float = 0.0
    accumulated_capex: float = field(default=0.0)


class CorporateFinanceEnv(gym.Env):
    """An RL environment for multi-year corporate capital-allocation strategy.

    Observation (``Box(9,)``, see :data:`OBS_FIELDS` for field order):
        cash_balance, revenue, growth_rate, gross_margin, ebitda_margin,
        debt_to_equity, wacc, market_demand_index, inflation_rate.

    Action (``Box(3,)``, see :data:`ACTION_FIELDS` for field order), each
    component in ``[-1, 1]``:
        capex_pct        -- growth investment, rescaled to [0, 1] fraction of
                             revenue spent on CapEx.
        debt_action       -- issue (+) or pay down (-) debt, rescaled to
                             ``[-max_pct, +max_pct]`` of current equity.
        opex_adjustment   -- cut (-1) to heavily invest (+1) in R&D/marketing,
                             a relative adjustment to the discretionary-opex
                             baseline.

    Reward combines free-cash-flow generation with the change in estimated
    enterprise value, and applies large penalties for insolvency (negative
    cash) or breaching the leverage covenant (Debt / EBITDA too high).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: CFOEnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or CFOEnvConfig()
        self.render_mode = render_mode

        # Actions are agent-friendly in [-1, 1]; `_decode_action` maps them
        # onto the real financial ranges described in the class docstring.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observations are unbounded in principle (e.g. leverage can spike
        # before termination), so we use a wide but finite Box rather than
        # +/-inf to keep downstream normalization well-behaved.
        obs_low = np.array([-1e3, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -0.5], dtype=np.float32)
        obs_high = np.array([1e5, 1e5, 2.0, 1.0, 1.0, 50.0, 1.0, 5.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._state: _FinancialState | None = None
        self._np_random: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        cfg = self.config
        noise = cfg.seed_state_noise

        def jitter(value: float) -> float:
            if noise <= 0.0:
                return value
            return float(value * (1.0 + self._np_random.normal(0.0, noise)))

        self._state = _FinancialState(
            year=0,
            cash=jitter(cfg.initial_cash),
            revenue=jitter(cfg.initial_revenue),
            growth_rate=cfg.initial_growth_rate,
            gross_margin=cfg.initial_gross_margin,
            ebitda_margin=cfg.initial_ebitda_margin,
            debt=jitter(cfg.initial_debt),
            equity=jitter(cfg.initial_equity),
            wacc=cfg.initial_wacc,
            demand_index=cfg.initial_demand_index,
            inflation=cfg.initial_inflation,
        )
        self._state.prev_enterprise_value = self._enterprise_value(self._state)

        observation = self._get_obs()
        info = self._get_info(fcf=0.0, terminated_reason=None)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: np.ndarray):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        cfg = self.config
        s = self._state
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        capex_pct, debt_action_norm, opex_adj = self._decode_action(action)

        # ---- income statement for the year, before financing decisions ----
        gross_profit = s.revenue * s.gross_margin
        # Discretionary opex baseline is the gap between gross and EBITDA
        # margin; the agent's opex_adjustment scales incremental spend
        # on top of (or cut from) that baseline, funded out of gross profit.
        baseline_opex_margin = s.gross_margin - s.ebitda_margin
        baseline_opex = s.revenue * max(baseline_opex_margin, 0.0)
        discretionary_delta = opex_adj * 0.5 * baseline_opex  # up to +/-50% swing
        opex = max(baseline_opex + discretionary_delta, 0.0)

        capex_spend = capex_pct * s.revenue
        interest_expense = s.debt * (s.wacc + cfg.interest_rate_spread_over_wacc)
        depreciation = cfg.depreciation_rate * s.accumulated_capex

        ebitda = gross_profit - opex
        ebit = ebitda - depreciation
        pretax_income = ebit - interest_expense
        taxes = max(pretax_income, 0.0) * cfg.tax_rate
        net_income = pretax_income - taxes

        # ---- financing: debt issuance / paydown ----
        max_debt_move = cfg.max_debt_issue_or_paydown_pct_of_equity * s.equity
        debt_delta = debt_action_norm * max_debt_move
        # Can't pay down more debt than currently outstanding.
        debt_delta = max(debt_delta, -s.debt)
        new_debt = s.debt + debt_delta

        # ---- cash flow statement ----
        free_cash_flow = net_income + depreciation - capex_spend
        cash_from_financing = debt_delta
        new_cash = s.cash + free_cash_flow + cash_from_financing

        # ---- macro shocks (mean-reverting demand, noisy inflation) ----
        demand_shock = self._np_random.normal(0.0, cfg.demand_shock_std)
        new_demand_index = (
            s.demand_index
            + cfg.demand_mean_reversion * (1.0 - s.demand_index)
            + demand_shock
        )
        new_demand_index = max(new_demand_index, 0.1)

        inflation_shock = self._np_random.normal(0.0, cfg.inflation_shock_std)
        new_inflation = max(s.inflation + inflation_shock, -0.02)

        # ---- strategic dynamics: CapEx compounds growth, opex compounds
        # margin, both with diminishing returns (sqrt) and organic decay ----
        capex_intensity = capex_spend / max(s.revenue, 1e-6)
        growth_boost = cfg.capex_growth_sensitivity * np.sqrt(max(capex_intensity, 0.0))
        new_growth_rate = (
            (s.growth_rate - cfg.organic_decay) * 0.6
            + growth_boost
            + 0.10 * (new_demand_index - 1.0)
        )
        new_growth_rate = float(np.clip(new_growth_rate, -0.5, 1.0))

        opex_intensity = max(discretionary_delta, 0.0) / max(s.revenue, 1e-6)
        margin_boost = cfg.opex_margin_sensitivity * np.sqrt(opex_intensity)
        new_gross_margin = float(np.clip(s.gross_margin + margin_boost - 0.01, 0.05, 0.95))

        # EBITDA margin drifts toward what this year's opex/depreciation imply.
        implied_ebitda_margin = ebitda / max(s.revenue, 1e-6)
        new_ebitda_margin = float(np.clip(implied_ebitda_margin, -0.5, 0.9))

        new_revenue = max(s.revenue * (1.0 + new_growth_rate), 0.0)
        new_equity = s.equity + net_income  # retained-earnings roll-forward
        new_wacc = float(np.clip(s.wacc + 0.02 * demand_shock, 0.03, 0.25))

        next_state = _FinancialState(
            year=s.year + 1,
            cash=new_cash,
            revenue=new_revenue,
            growth_rate=new_growth_rate,
            gross_margin=new_gross_margin,
            ebitda_margin=new_ebitda_margin,
            debt=new_debt,
            equity=max(new_equity, 1e-6),
            wacc=new_wacc,
            demand_index=new_demand_index,
            inflation=new_inflation,
            prev_enterprise_value=s.prev_enterprise_value,
            accumulated_capex=s.accumulated_capex + capex_spend,
        )

        leverage_ratio = new_debt / max(ebitda, 1e-6) if ebitda > 0 else float("inf")
        insolvent = new_cash < cfg.insolvency_cash_floor
        covenant_breach = leverage_ratio > cfg.max_leverage_ratio

        reward, ev = self._compute_reward(
            state=next_state,
            free_cash_flow=free_cash_flow,
            insolvent=insolvent,
            covenant_breach=covenant_breach,
        )
        next_state.prev_enterprise_value = ev
        self._state = next_state

        terminated = bool(insolvent)
        truncated = bool(next_state.year >= cfg.horizon_years)
        terminated_reason = "insolvency" if insolvent else None

        observation = self._get_obs()
        info = self._get_info(fcf=free_cash_flow, terminated_reason=terminated_reason)
        info["leverage_ratio"] = float(leverage_ratio)
        info["covenant_breach"] = bool(covenant_breach)
        info["enterprise_value"] = float(ev)
        info["net_income"] = float(net_income)

        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self._state is None:
            return
        s = self._state
        print(
            f"Year {s.year:2d} | cash={s.cash:8.2f} rev={s.revenue:8.2f} "
            f"growth={s.growth_rate:+.2%} gm={s.gross_margin:.2%} "
            f"ebitda_m={s.ebitda_margin:.2%} debt={s.debt:7.2f} "
            f"equity={s.equity:7.2f}"
        )

    def close(self):
        self._state = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode_action(self, action: np.ndarray) -> tuple[float, float, float]:
        """Map the [-1, 1] action vector onto real financial decision ranges."""
        capex_pct = float((action[0] + 1.0) / 2.0)  # [-1,1] -> [0,1]
        debt_action_norm = float(action[1])          # already [-1,1]
        opex_adjustment = float(action[2])            # already [-1,1]
        return capex_pct, debt_action_norm, opex_adjustment

    def _enterprise_value(self, s: _FinancialState) -> float:
        """A simple perpetuity-growth DCF proxy used purely for reward shaping.

        Enterprise value is not part of the observation (it is not directly
        observable in real time either); it only anchors the reward's
        long-term-value term via period-over-period deltas.
        """
        ebitda = s.revenue * s.ebitda_margin
        # Guard against the perpetuity-growth formula blowing up when the
        # discount rate is close to (or below) the growth rate.
        g = float(np.clip(s.growth_rate, -0.10, s.wacc - 0.01))
        denom = max(s.wacc - g, 0.01)
        return float(ebitda * (1.0 + g) / denom)

    def _compute_reward(
        self,
        state: _FinancialState,
        free_cash_flow: float,
        insolvent: bool,
        covenant_breach: bool,
    ) -> tuple[float, float]:
        cfg = self.config
        ev = self._enterprise_value(state)
        ev_delta = ev - state.prev_enterprise_value

        # Scale by initial revenue so the reward magnitude stays comparable
        # across differently sized companies/configs.
        scale = max(cfg.initial_revenue, 1e-6)
        reward = (
            cfg.fcf_reward_weight * (free_cash_flow / scale)
            + cfg.enterprise_value_reward_weight * (ev_delta / scale)
        )

        if insolvent:
            reward -= cfg.insolvency_penalty
        if covenant_breach:
            reward -= cfg.covenant_breach_penalty

        return reward, ev

    def _get_obs(self) -> np.ndarray:
        s = self._state
        assert s is not None
        debt_to_equity = s.debt / max(s.equity, 1e-6)
        obs = np.array(
            [
                s.cash,
                s.revenue,
                s.growth_rate,
                s.gross_margin,
                s.ebitda_margin,
                debt_to_equity,
                s.wacc,
                s.demand_index,
                s.inflation,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self, fcf: float, terminated_reason: str | None) -> dict:
        s = self._state
        assert s is not None
        return {
            "year": s.year,
            "free_cash_flow": float(fcf),
            "debt": float(s.debt),
            "equity": float(s.equity),
            "terminated_reason": terminated_reason,
        }


def make_env(config: CFOEnvConfig | None = None, render_mode: str | None = None) -> CorporateFinanceEnv:
    """Convenience factory mirroring the ``gym.make``-style construction."""
    return CorporateFinanceEnv(config=config, render_mode=render_mode)
