"""
Descarga velas OHLCV reales de BTCUSDT desde la API publica de Binance
(sin API key, endpoint de solo lectura) y las guarda en
data/ohlcv_real_binance.csv con el mismo esquema que el dataset sintetico
de train_classifier.py, para que build_features_and_label() y el resto del
pipeline funcionen sin cambios sobre datos reales.

Uso:
    .\\venv\\Scripts\\python.exe fetch_binance_data.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import polars as pl

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
INTERVAL_MS = 3_600_000
N_CANDLES = 20_000
REQUEST_LIMIT = 1000
REQUEST_DELAY_SECONDS = 0.3  # cortesia con la API publica, evita rate limiting

ROOT = Path(__file__).parent
OUTPUT_PATH = ROOT / "data" / "ohlcv_real_binance.csv"


def fetch_klines_batch(symbol: str, interval: str, start_time_ms: int, limit: int = REQUEST_LIMIT) -> list:
    url = f"{BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}&startTime={start_time_ms}&limit={limit}"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read())


def fetch_real_ohlcv(n_candles: int = N_CANDLES, symbol: str = SYMBOL, interval: str = INTERVAL) -> pl.DataFrame:
    """Pagina hacia atras desde 'ahora' usando startTime, ya que la API publica
    de Binance limita cada respuesta a 1000 velas por llamada."""
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - n_candles * INTERVAL_MS

    rows: list = []
    cursor = start_time_ms
    print(f"Descargando {n_candles:,} velas reales de {symbol} ({interval}) desde la API publica de Binance...")
    while len(rows) < n_candles:
        batch = fetch_klines_batch(symbol, interval, cursor)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + INTERVAL_MS
        print(f"  {len(rows):,}/{n_candles:,} velas descargadas...")
        if len(batch) < REQUEST_LIMIT:
            break
        time.sleep(REQUEST_DELAY_SECONDS)

    rows = rows[:n_candles]

    df = pl.DataFrame(
        rows,
        schema=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ],
        orient="row",
    )
    return df.select(
        pl.col("open_time").cast(pl.Int64).cast(pl.Datetime("ms")).alias("timestamp"),
        pl.lit(symbol).alias("symbol"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_real_ohlcv()
    df.write_csv(OUTPUT_PATH)
    print(f"\n{len(df):,} velas reales guardadas en {OUTPUT_PATH}")
    print(f"Rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
