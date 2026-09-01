"""Shared pytest fixtures: synthetic order-book/trade-flow data that
mimics the shape of the real Binance dataset (same column names/ranges)
so unit tests run fast and offline, without requiring the ~550 MB real
download.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

N_ROWS = 400


@pytest.fixture
def synthetic_raw_df() -> pl.DataFrame:
    """A synthetic version of the joined `trades x book` frame produced by
    `xgboost_impact.build_dataset`, with the same schema."""
    rng = np.random.default_rng(0)
    bins = pl.datetime_range(
        start=pl.datetime(2024, 1, 1), end=pl.datetime(2024, 1, 1, 6, 39),
        interval="1m", eager=True,
    )[:N_ROWS]

    price = 60000 + np.cumsum(rng.normal(0, 5, N_ROWS))
    trade_volume = np.abs(rng.normal(10, 3, N_ROWS)) + 0.1
    signed_volume = rng.normal(0, 3, N_ROWS)

    return pl.DataFrame({
        "bin": bins,
        "vwap": price,
        "trade_volume": trade_volume,
        "signed_volume": signed_volume,
        "trade_flow_imbalance": signed_volume / trade_volume,
        "near_depth_imbalance": rng.uniform(-0.3, 0.3, N_ROWS),
        "far_depth_imbalance": rng.uniform(-0.3, 0.3, N_ROWS),
        "total_depth": np.abs(rng.normal(500, 50, N_ROWS)),
    })
