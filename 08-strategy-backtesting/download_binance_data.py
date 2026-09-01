"""Reproducible downloader for real historical price data from Binance Vision
(`data.binance.vision`) — Binance's free, public, no-authentication archive of
historical market data.

Fetches monthly spot `klines` (candlestick) archives at daily interval for a
single symbol and stitches them into one clean OHLCV DataFrame. No API key
required, no rate limits beyond a normal HTTPS download.

Uso:
    python download_binance_data.py
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START_YEAR_MONTH = (2021, 1)
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_PATH = Path("data/raw") / f"{SYMBOL}_{INTERVAL}_binance_vision.csv"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "n_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def _month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months = []
    y, m = start
    while (y, m) <= end:
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _last_complete_month() -> tuple[int, int]:
    """Binance Vision publishes a monthly archive only once that month is
    fully closed, así que el mes en curso normalmente todavia no existe."""
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    return (y, m)


def _download_month(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame | None:
    filename = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if response.status_code != 200:
        return None

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)

    # Los archivos historicos usan microsegundos en open_time/close_time; los
    # mas recientes (Binance migro el formato) usan milisegundos -- se
    # distingue por la magnitud del numero, no por la fecha del archivo.
    unit = "us" if df["open_time"].iloc[0] > 10**14 else "ms"
    df["date"] = pd.to_datetime(df["open_time"], unit=unit)
    return df[["date", "open", "high", "low", "close", "volume"]]


def download_binance_klines(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    start: tuple[int, int] = START_YEAR_MONTH,
    end: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Descarga y concatena los archivos mensuales de klines reales entre
    `start` y `end` (año, mes), ambos inclusive. `end=None` usa el ultimo mes
    ya cerrado disponible en Binance Vision."""
    end = end or _last_complete_month()
    frames = []
    for year, month in _month_range(start, end):
        df_month = _download_month(symbol, interval, year, month)
        if df_month is None:
            print(f"  {symbol} {year:04d}-{month:02d}: no disponible todavia, se omite.")
            continue
        frames.append(df_month)
        print(f"  {symbol} {year:04d}-{month:02d}: {len(df_month)} velas diarias descargadas.")

    if not frames:
        raise RuntimeError(f"No se pudo descargar ningun mes para {symbol} {interval}.")

    full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    full = full.drop_duplicates(subset="date", keep="last")
    for col in ("open", "high", "low", "close", "volume"):
        full[col] = full[col].astype(float)
    return full


def main() -> pd.DataFrame:
    print(f"Descargando historial real de {SYMBOL} ({INTERVAL}) desde Binance Vision...")
    df = download_binance_klines()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{len(df):,} velas diarias reales guardadas en {OUTPUT_PATH}")
    print(f"Rango: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Precio de cierre actual: ${df['close'].iloc[-1]:,.2f}")
    return df


if __name__ == "__main__":
    main()
