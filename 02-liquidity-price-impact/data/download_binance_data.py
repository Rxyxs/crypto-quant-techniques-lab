"""Reproducibly downloads real BTCUSDT perpetual-futures order-book depth
and aggregated-trade data from Binance Vision (data.binance.vision), a
free, public, no-authentication historical market-data archive published
directly by Binance.

Two real data series are fetched per day:
  - bookDepth:  cumulative order-book depth/notional at +/-1%..5% from the
                mid price, snapshotted roughly every 30 seconds.
  - aggTrades:  every aggregated real trade (price, quantity, aggressor
                side via `is_buyer_maker`).

Run: venv/Scripts/python.exe data/download_binance_data.py
"""
from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

SYMBOL = "BTCUSDT"
START_DATE = date(2024, 5, 13)
END_DATE = date(2024, 5, 19)  # inclusive, 7 real days
BASE_URL = "https://data.binance.vision/data/futures/um/daily"
RAW_DIR = Path(__file__).parent / "raw"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_and_extract(dataset: str, day: date, out_dir: Path) -> Path | None:
    fname = f"{SYMBOL}-{dataset}-{day.isoformat()}"
    url = f"{BASE_URL}/{dataset}/{SYMBOL}/{fname}.zip"
    out_csv = out_dir / f"{fname}.csv"
    if out_csv.exists():
        print(f"  [{dataset}] {day} already downloaded, skipping")
        return out_csv

    try:
        with urlopen(url, timeout=60) as resp:
            raw = resp.read()
    except HTTPError as e:
        print(f"  [{dataset}] {day} unavailable ({e.code}), skipping")
        return None

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        member = zf.namelist()[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(out_csv, "wb") as dst:
            dst.write(src.read())
    print(f"  [{dataset}] {day} -> {out_csv.name} ({out_csv.stat().st_size / 1e6:.1f} MB)")
    return out_csv


def main() -> None:
    print(f"Downloading real {SYMBOL} perpetual-futures data from Binance Vision")
    print(f"Period: {START_DATE} .. {END_DATE} (source: {BASE_URL})")
    for day in daterange(START_DATE, END_DATE):
        download_and_extract("bookDepth", day, RAW_DIR / "bookDepth")
        download_and_extract("aggTrades", day, RAW_DIR / "aggTrades")
    print("Done.")


if __name__ == "__main__":
    main()
