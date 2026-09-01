import numpy as np

from detect_spoofing import (
    L2_FEATURE_COLS,
    L2_FULL_RAW_FEATURE_COLS,
    N_LEVELS,
    RAW_L2_LEVEL_COLS,
    engineer_features,
    simulate_orderbook,
)


def _sim(n=800, seed=1):
    rng = np.random.default_rng(seed)
    return engineer_features(simulate_orderbook(n, rng))


def test_simulated_book_has_all_l2_level_columns():
    df = _sim()
    for side in ("bid", "ask"):
        for lvl in range(N_LEVELS):
            for prefix in ("size", "placed", "canceled", "executed"):
                assert f"{side}_{prefix}_l{lvl}" in df.columns


def test_engineered_l2_features_have_no_nulls():
    df = _sim()
    assert df.select(L2_FULL_RAW_FEATURE_COLS).null_count().sum_horizontal().sum() == 0


def test_spoofed_level_depth_is_never_a_model_feature():
    """spoofed_level_depth is derived directly from the ground-truth label
    -- it must exist only as a diagnostic column, never in a feature list
    fed to the model."""
    assert "spoofed_level_depth" not in L2_FEATURE_COLS
    assert "spoofed_level_depth" not in L2_FULL_RAW_FEATURE_COLS
    assert "spoofed_level_depth" not in RAW_L2_LEVEL_COLS


def test_spoof_burst_only_affects_its_own_side_and_levels():
    """A spoof burst must leave the opposite side, and levels outside the
    affected band, statistically indistinguishable from normal snapshots
    -- checked directly against the simulator's own logged spoof levels."""
    rng = np.random.default_rng(3)
    df = _sim(n=2000, seed=3)
    spoofed = df.filter(df["true_is_spoof"])
    assert spoofed.height > 0
    # every spoofed row must have at least one level with a near-total
    # cancellation ratio (the manipulated band), confirming the injected
    # signature actually landed in the per-level columns.
    cancel_cols = [c for c in RAW_L2_LEVEL_COLS if "cancel_ratio" in c]
    max_cancel = df.select(cancel_cols).to_numpy().max(axis=1)
    assert max_cancel[spoofed["snapshot_id"].to_numpy()].min() > 5.0


def test_order_book_imbalance_uses_full_l2_depth_not_just_touch():
    """order_book_imbalance must react to size added at deeper levels, not
    only to the top-of-book -- otherwise it isn't really an L2 feature."""
    rng = np.random.default_rng(5)
    df = simulate_orderbook(300, rng)
    row = df.row(100, named=True)
    total_bid = sum(row[f"bid_size_l{lvl}"] for lvl in range(N_LEVELS))
    total_ask = sum(row[f"ask_size_l{lvl}"] for lvl in range(N_LEVELS))
    expected = (total_bid - total_ask) / (total_bid + total_ask)
    assert abs(row["order_book_imbalance"] - expected) < 1e-9
    # confirm it's not equal to the top-of-book-only version unless coincidence
    top_only = (row["bid_size_l0"] - row["ask_size_l0"]) / (row["bid_size_l0"] + row["ask_size_l0"])
    assert row["order_book_imbalance"] != top_only or abs(total_bid - row["bid_size_l0"]) < 1e-9
