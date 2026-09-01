import numpy as np
import pandas as pd

from frictions import (
    DEFAULT_SLIPPAGE_BASE_BPS,
    apply_frictions,
    compute_dynamic_slippage_bps,
    total_friction_cost_summary,
)


def _fake_backtest(n=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    signal = rng.integers(0, 2, n)
    strategy_return = rng.normal(0, 0.01, n)
    vol_20d = np.abs(rng.normal(0.02, 0.005, n))
    return pd.DataFrame({
        "date": dates,
        "signal": signal,
        "strategy_return": strategy_return,
        "vol_20d": vol_20d,
    })


def test_dynamic_slippage_increases_with_volatility():
    low_vol = pd.Series([0.0, 0.0, 0.0])
    high_vol = pd.Series([0.05, 0.05, 0.05])
    low = compute_dynamic_slippage_bps(low_vol)
    high = compute_dynamic_slippage_bps(high_vol)
    assert (high > low).all()
    assert (low == DEFAULT_SLIPPAGE_BASE_BPS).all()


def test_apply_frictions_charges_only_on_position_changes():
    bt = _fake_backtest()
    out = apply_frictions(bt)
    # Donde no hay turnover, el costo debe ser exactamente cero.
    no_turnover = out[out["turnover"] == 0]
    assert (no_turnover["cost_taker"] == 0).all()
    assert (no_turnover["cost_maker"] == 0).all()
    # Donde hay turnover, taker (fee mayor) debe costar al menos tanto como maker.
    with_turnover = out[out["turnover"] > 0]
    assert (with_turnover["cost_taker"] >= with_turnover["cost_maker"]).all()


def test_apply_frictions_net_return_never_exceeds_gross():
    bt = _fake_backtest()
    out = apply_frictions(bt)
    assert (out["strategy_return_net_taker"] <= out["strategy_return"] + 1e-12).all()
    assert (out["strategy_return_net_maker"] <= out["strategy_return"] + 1e-12).all()


def test_total_friction_cost_summary_is_nonnegative_and_consistent():
    bt = _fake_backtest()
    out = apply_frictions(bt)
    summary = total_friction_cost_summary(out)
    assert summary["n_position_changes"] >= 0
    assert summary["total_fee_drag_pct"] >= 0
    assert summary["total_slippage_drag_pct"] >= 0
    assert np.isclose(
        summary["total_friction_drag_pct"],
        summary["total_fee_drag_pct"] + summary["total_slippage_drag_pct"],
    )
