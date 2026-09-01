"""Reproducibly downloads real daily closing prices for a basket of major
crypto assets from Binance Vision (data.binance.vision), a free, public,
no-authentication historical market-data archive published by Binance.

Fetches 24 months of real daily klines (spot, monthly archives) for each
symbol in ASSETS and writes one combined CSV of daily close prices.

Run: venv/Scripts/python.exe data/download_binance_data.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
MONTHS = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13)
          if (y, m) >= (2022, 6) and (y, m) <= (2024, 5)]
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
RAW_DIR = Path(__file__).parent / "raw"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def download_month(symbol: str, month: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{symbol}/1d/{symbol}-1d-{month}.zip"
    try:
        with urlopen(url, timeout=60) as resp:
            raw = resp.read()
    except HTTPError as e:
        print(f"  [{symbol}] {month} unavailable ({e.code}), skipping")
        return None

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
    return df


def fetch_symbol_closes(symbol: str) -> pd.Series:
    frames = []
    for month in MONTHS:
        df = download_month(symbol, month)
        if df is not None:
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["date"] = pd.to_datetime(full["open_time"], unit="ms").dt.date
    series = full.set_index("date")["close"].astype(float)
    series.name = symbol
    return series


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading real daily closes for {ASSETS} ({MONTHS[0]}..{MONTHS[-1]}) from Binance Vision")

    closes = {}
    for symbol in ASSETS:
        print(f"Fetching {symbol}...")
        closes[symbol] = fetch_symbol_closes(symbol)

    prices = pd.DataFrame(closes).sort_index()
    prices.index.name = "date"
    out_path = RAW_DIR / "daily_closes.csv"
    prices.to_csv(out_path)
    print(f"Wrote {len(prices)} rows x {len(ASSETS)} assets -> {out_path}")


if __name__ == "__main__":
    main()
