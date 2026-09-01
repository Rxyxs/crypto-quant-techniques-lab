"""
Motor de pairs trading basado en cointegracion para pares de criptomonedas
(par principal: BTC/ETH), sobre precios diarios reales descargados de la
API publica de Binance.

Implementa el flujo estandar de una estrategia de pairs trading:
    1. test de cointegracion de Engle-Granger sobre los log-precios
    2. estimacion del hedge ratio via regresion OLS
    3. construccion del spread y verificacion de estacionariedad (ADF)
    4. z-score rolling del spread -> señales de entrada/salida
    5. backtest del spread dolar-neutral, con costos de transaccion
       (fee + slippage asimetrico entrada/salida) incluidos por defecto

El hedge ratio OLS de este modulo es **estatico** (un solo numero para toda
la muestra); `kalman_pairs.py` implementa la alternativa **dinamica** (Filtro
de Kalman) que deja que el hedge ratio derive con el tiempo, reutilizando
`generate_signals`/`backtest_spread_strategy` de este mismo modulo.

Uso como script (descarga datos y corre el flujo completo con BTC/ETH):
    .\\venv\\Scripts\\python.exe pairs_trading_engine.py

Tambien se importa desde 01_Cointegration_Analysis.ipynb (OLS estatico) y
02_Kalman_Dynamic_Hedge_Ratio.ipynb (comparacion OLS vs Kalman) para el
analisis paso a paso con graficos.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
N_CANDLES = 1000  # maximo de velas diarias en una sola llamada a la API (~2.7 anos)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
PRIMARY_PAIR = ("BTCUSDT", "ETHUSDT")

ROLLING_ZSCORE_WINDOW = 30
ROLLING_CORR_WINDOW = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5

# Costos de transaccion: fee tipo taker de Binance (10bps es representativo
# para la mayoria de los pares del universo, ver la tabla de fees por
# volumen de Binance) mas slippage, deliberadamente ASIMETRICO entre entrada
# y salida -- no el mismo numero aplicado dos veces. Entrar a una posicion
# ocurre cuando |z| acaba de cruzar el umbral de entrada, es decir en un
# evento de cola (2 sigma): el libro de ordenes es tipicamente mas fino en
# ese instante, y cualquier orden de tamano no trivial mueve mas el precio.
# Salir ocurre cerca de |z| < exit_z, es decir cuando el spread ya volvio a
# un regimen mas calmo -- razonable esperar un libro mas profundo y menos
# slippage. El costo se cobra por las DOS patas del par (se compra/vende Y y
# X simultaneamente), de ahi el factor 2 en `compute_transaction_cost_returns`.
TAKER_FEE_BPS = 10.0
ENTRY_SLIPPAGE_BPS = 5.0
EXIT_SLIPPAGE_BPS = 2.0

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"


def fetch_symbol_closes(symbol: str, interval: str = INTERVAL, n_candles: int = N_CANDLES) -> pd.Series:
    """Descarga velas de la API publica de Binance (sin API key) y devuelve
    el precio de cierre diario como una Serie indexada por fecha. A 1 vela/dia,
    n_candles<=1000 entra en una sola llamada -- no hace falta paginar."""
    url = f"{BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}&limit={n_candles}"
    with urllib.request.urlopen(url, timeout=20) as response:
        rows = json.loads(response.read())
    index = pd.to_datetime([r[0] for r in rows], unit="ms")
    closes = pd.Series([float(r[4]) for r in rows], index=index, name=symbol)
    return closes


def fetch_price_panel(symbols: list[str] = SYMBOLS, interval: str = INTERVAL, n_candles: int = N_CANDLES) -> pd.DataFrame:
    """Descarga y alinea los precios de cierre de varios simbolos en un solo
    DataFrame (columnas = simbolos, filas = fechas)."""
    series = {}
    for symbol in symbols:
        print(f"  descargando {symbol} ({interval}, {n_candles} velas)...")
        series[symbol] = fetch_symbol_closes(symbol, interval, n_candles)
        time.sleep(0.25)  # cortesia con la API publica
    panel = pd.DataFrame(series).dropna()
    return panel


@dataclass
class CointegrationResult:
    test_statistic: float
    p_value: float
    critical_values: dict[str, float]

    @property
    def is_cointegrated_5pct(self) -> bool:
        return self.p_value < 0.05


def engle_granger_test(y: pd.Series, x: pd.Series) -> CointegrationResult:
    """Test de Engle-Granger (statsmodels.tsa.stattools.coint) sobre dos
    series de log-precios. H0: las series NO estan cointegradas."""
    score, p_value, crit_values = coint(y, x)
    return CointegrationResult(
        test_statistic=float(score),
        p_value=float(p_value),
        critical_values={"1%": crit_values[0], "5%": crit_values[1], "10%": crit_values[2]},
    )


@dataclass
class HedgeRatioResult:
    beta: float
    alpha: float
    r_squared: float


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> HedgeRatioResult:
    """Regresion OLS de y sobre x (con intercepto) para estimar el hedge
    ratio (beta) del vector de cointegracion (1, -beta)."""
    x_with_const = sm.add_constant(x.rename("x"))
    model = sm.OLS(y, x_with_const).fit()
    return HedgeRatioResult(
        beta=float(model.params["x"]),
        alpha=float(model.params["const"]),
        r_squared=float(model.rsquared),
    )


def compute_spread(y: pd.Series, x: pd.Series, hedge: HedgeRatioResult) -> pd.Series:
    """spread = y - (alpha + beta*x); si y y x estan cointegradas, el spread
    es estacionario (revierte a su media) aunque y y x individualmente no lo sean."""
    return (y - (hedge.alpha + hedge.beta * x)).rename("spread")


@dataclass
class StationarityResult:
    adf_statistic: float
    p_value: float
    critical_values: dict[str, float]

    @property
    def is_stationary_5pct(self) -> bool:
        return self.p_value < 0.05


def adf_test(series: pd.Series) -> StationarityResult:
    """Test aumentado de Dickey-Fuller. H0: la serie tiene raiz unitaria (no
    es estacionaria)."""
    stat, p_value, _, _, crit_values, _ = adfuller(series.dropna())
    return StationarityResult(
        adf_statistic=float(stat),
        p_value=float(p_value),
        critical_values={k: float(v) for k, v in crit_values.items()},
    )


def rolling_zscore(spread: pd.Series, window: int = ROLLING_ZSCORE_WINDOW) -> pd.Series:
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return ((spread - rolling_mean) / rolling_std).rename("zscore")


def generate_signals(zscore: pd.Series, entry_z: float = ENTRY_Z, exit_z: float = EXIT_Z) -> pd.Series:
    """Posicion del spread en cada dia: +1 = largo el spread (largo y, corto
    x), -1 = corto el spread, 0 = sin posicion. Entra cuando |z| supera
    entry_z, sale cuando |z| cae bajo exit_z, y mantiene la posicion previa
    mientras tanto (logica de banda, no de umbral instantaneo)."""
    position = np.zeros(len(zscore))
    current = 0
    z_values = zscore.to_numpy()
    for i, z in enumerate(z_values):
        if np.isnan(z):
            position[i] = 0
            continue
        if current == 0:
            if z > entry_z:
                current = -1
            elif z < -entry_z:
                current = 1
        else:
            if abs(z) < exit_z:
                current = 0
        position[i] = current
    return pd.Series(position, index=zscore.index, name="position")


def compute_transaction_cost_returns(
    positions: pd.Series,
    taker_fee_bps: float = TAKER_FEE_BPS,
    entry_slippage_bps: float = ENTRY_SLIPPAGE_BPS,
    exit_slippage_bps: float = EXIT_SLIPPAGE_BPS,
) -> pd.Series:
    """Costo de transaccion (fee + slippage), expresado en las mismas
    unidades de retorno fraccional que `strategy_returns`, cobrado el dia en
    que la posicion efectivamente cambia -- no en cada dia que se mantiene
    una posicion abierta. Un cambio se clasifica como:
      - entrada (0 -> +-1): tasa de entrada
      - salida (+-1 -> 0): tasa de salida (mas barata, ver constantes arriba)
      - flip directo (+1 -> -1 o viceversa, |cambio|=2): se cobra salida +
        entrada, porque economicamente es cerrar una posicion y abrir la
        contraria en el mismo dia.
    El factor 2 en cada tasa es por las DOS patas del par (se compra/vende Y
    y X simultaneamente en cada entrada/salida).
    """
    prev_position = positions.shift(1).fillna(0.0)
    is_entry = (prev_position == 0) & (positions != 0)
    is_exit = (prev_position != 0) & (positions == 0)
    is_flip = (prev_position != 0) & (positions != 0) & (prev_position != positions)

    entry_rate = (taker_fee_bps + entry_slippage_bps) / 10_000 * 2
    exit_rate = (taker_fee_bps + exit_slippage_bps) / 10_000 * 2

    cost = (
        is_entry.astype(float) * entry_rate
        + is_exit.astype(float) * exit_rate
        + is_flip.astype(float) * (entry_rate + exit_rate)
    )
    return cost.rename("transaction_cost")


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    equity_curve_gross: pd.Series  # antes de costos de transaccion
    daily_returns: pd.Series
    total_return: float
    total_return_gross: float
    annualized_sharpe: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    total_transaction_cost: float


def backtest_spread_strategy(
    y: pd.Series,
    x: pd.Series,
    beta: float | pd.Series,
    positions: pd.Series,
    include_costs: bool = True,
) -> BacktestResult:
    """Backtest dolar-neutral: cada dia, el PnL bruto es
    position[t-1] * (retorno_y[t] - beta*retorno_x[t]), es decir la posicion
    de AYER capturando el retorno del spread de HOY (sin look-ahead).

    `beta` acepta un float (hedge ratio estatico, OLS) o una `pd.Series`
    alineada al mismo indice que `y`/`x` (hedge ratio dinamico, Kalman) --
    en ambos casos la multiplicacion `beta * x_returns` funciona sin cambios
    porque pandas hace el broadcast correcto en los dos escenarios.

    Si `include_costs=True` (default), se resta `compute_transaction_cost_returns`
    de los retornos brutos antes de acumular el equity curve neto -- `equity_curve_gross`
    siempre queda disponible para medir cuanto "cuesta" operar de verdad.
    """
    y_returns = y.pct_change()
    x_returns = x.pct_change()
    spread_returns = y_returns - beta * x_returns
    gross_returns = (positions.shift(1) * spread_returns).fillna(0.0).rename("gross_returns")

    transaction_costs = compute_transaction_cost_returns(positions) if include_costs else pd.Series(0.0, index=positions.index)
    strategy_returns = (gross_returns - transaction_costs).rename("strategy_returns")

    equity_curve_gross = (1 + gross_returns).cumprod().rename("equity_gross")
    equity_curve = (1 + strategy_returns).cumprod().rename("equity")
    total_return_gross = float(equity_curve_gross.iloc[-1] - 1)
    total_return = float(equity_curve.iloc[-1] - 1)

    daily_mean = strategy_returns.mean()
    daily_std = strategy_returns.std()
    annualized_sharpe = float((daily_mean / daily_std) * np.sqrt(365)) if daily_std > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min())

    position_changes = positions.diff().fillna(0)
    trade_entries = (position_changes != 0) & (positions != 0)
    n_trades = int(trade_entries.sum())

    winning_days = (strategy_returns[positions.shift(1) != 0] > 0).sum()
    active_days = (positions.shift(1) != 0).sum()
    win_rate = float(winning_days / active_days) if active_days > 0 else 0.0

    return BacktestResult(
        equity_curve=equity_curve,
        equity_curve_gross=equity_curve_gross,
        daily_returns=strategy_returns,
        total_return=total_return,
        total_return_gross=total_return_gross,
        annualized_sharpe=annualized_sharpe,
        max_drawdown=max_drawdown,
        n_trades=n_trades,
        win_rate=win_rate,
        total_transaction_cost=float(transaction_costs.sum()),
    )


def screen_pairs_for_cointegration(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Practica estandar de pairs trading: nunca se opera un par solo porque
    'suena correlacionado' (BTC/ETH es la eleccion obvia) -- se testea
    cointegracion en TODO el universo de pares candidatos y solo se opera
    el/los que efectivamente pasan el test. Devuelve una tabla con el
    resultado de Engle-Granger para cada par, ordenada por p-value."""
    symbols = list(price_panel.columns)
    log_prices = np.log(price_panel)
    rows = []
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1 :]:
            eg = engle_granger_test(log_prices[sym_a], log_prices[sym_b])
            rows.append(
                {
                    "pair": f"{sym_a[:-4]}/{sym_b[:-4]}",
                    "y": sym_a,
                    "x": sym_b,
                    "eg_statistic": eg.test_statistic,
                    "eg_pvalue": eg.p_value,
                    "cointegrated_5pct": eg.is_cointegrated_5pct,
                }
            )
    return pd.DataFrame(rows).sort_values("eg_pvalue").reset_index(drop=True)


def rolling_pairwise_correlations(price_panel: pd.DataFrame, window: int = ROLLING_CORR_WINDOW) -> pd.DataFrame:
    """Correlacion rolling (ventana movil) de retornos diarios entre cada par
    de simbolos, en formato ancho: filas = fecha, columnas = 'SIM1-SIM2'."""
    returns = price_panel.pct_change().dropna()
    symbols = list(price_panel.columns)
    pair_corrs = {}
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1 :]:
            pair_name = f"{sym_a[:-4]}-{sym_b[:-4]}"  # quita el sufijo USDT
            pair_corrs[pair_name] = returns[sym_a].rolling(window).corr(returns[sym_b])
    return pd.DataFrame(pair_corrs).dropna()


def run_full_pipeline(y_symbol: str | None = None, x_symbol: str | None = None) -> None:
    """Si no se especifica un par, se descarga el universo completo de
    SYMBOLS, se hace screening de cointegracion en todos los pares (practica
    estandar -- nunca asumir que el par "obvio" cointegra), y se elige
    automaticamente el que efectivamente pasa el test para el analisis en
    profundidad (spread, z-score, backtest)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Descargando precios diarios reales de {len(SYMBOLS)} simbolos desde Binance...")
    panel = fetch_price_panel()
    panel.to_csv(DATA_DIR / "prices_panel.csv")
    print(f"  {len(panel)} dias descargados ({panel.index.min().date()} -> {panel.index.max().date()})")

    print("\nScreening de cointegracion sobre todos los pares candidatos...")
    screening = screen_pairs_for_cointegration(panel)
    screening.to_csv(REPORTS_DIR / "cointegration_screening.csv", index=False)
    for row in screening.itertuples():
        flag = "OK" if row.cointegrated_5pct else "--"
        print(f"  [{flag}] {row.pair:<12} p-value={row.eg_pvalue:.4f}")

    if y_symbol is None or x_symbol is None:
        best = screening.iloc[0]
        y_symbol, x_symbol = best.y, best.x
        print(f"\nPar seleccionado automaticamente (menor p-value): {y_symbol}/{x_symbol}")

    log_y = np.log(panel[y_symbol])
    log_x = np.log(panel[x_symbol])

    print(f"\nTest de cointegracion Engle-Granger ({y_symbol} vs {x_symbol})...")
    eg_result = engle_granger_test(log_y, log_x)
    print(f"  estadistico={eg_result.test_statistic:.3f}  p-value={eg_result.p_value:.4f}  "
          f"cointegrados al 5%: {eg_result.is_cointegrated_5pct}")

    hedge = estimate_hedge_ratio(log_y, log_x)
    print(f"\nHedge ratio (OLS): beta={hedge.beta:.4f}  alpha={hedge.alpha:.4f}  R2={hedge.r_squared:.3f}")

    spread = compute_spread(log_y, log_x, hedge)
    adf_result = adf_test(spread)
    print(f"\nADF test sobre el spread: estadistico={adf_result.adf_statistic:.3f}  "
          f"p-value={adf_result.p_value:.4f}  estacionario al 5%: {adf_result.is_stationary_5pct}")

    zscore = rolling_zscore(spread)
    positions = generate_signals(zscore)
    backtest = backtest_spread_strategy(panel[y_symbol], panel[x_symbol], hedge.beta, positions)

    print(f"\nBacktest ({ENTRY_Z=}, {EXIT_Z=}, ventana z-score={ROLLING_ZSCORE_WINDOW}d):")
    print(f"  retorno total (neto de costos): {backtest.total_return:.1%}  (bruto: {backtest.total_return_gross:.1%})")
    print(f"  Sharpe anualizado: {backtest.annualized_sharpe:.2f}")
    print(f"  max drawdown: {backtest.max_drawdown:.1%}")
    print(f"  trades: {backtest.n_trades}  win rate: {backtest.win_rate:.1%}")
    print(f"  costo total de transaccion: {backtest.total_transaction_cost:.2%} acumulado")

    metrics = {
        "pair": f"{y_symbol}/{x_symbol}",
        "n_days": len(panel),
        "date_range": [str(panel.index.min().date()), str(panel.index.max().date())],
        "engle_granger_pvalue": eg_result.p_value,
        "cointegrated_5pct": eg_result.is_cointegrated_5pct,
        "hedge_ratio_beta": hedge.beta,
        "hedge_ratio_alpha": hedge.alpha,
        "hedge_ratio_r_squared": hedge.r_squared,
        "adf_pvalue": adf_result.p_value,
        "spread_stationary_5pct": adf_result.is_stationary_5pct,
        "total_return": backtest.total_return,
        "total_return_gross": backtest.total_return_gross,
        "total_transaction_cost": backtest.total_transaction_cost,
        "annualized_sharpe": backtest.annualized_sharpe,
        "max_drawdown": backtest.max_drawdown,
        "n_trades": backtest.n_trades,
        "win_rate": backtest.win_rate,
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetricas guardadas en {REPORTS_DIR / 'metrics.json'}")

    try:
        from db_persistence import persist_metrics, persist_screening

        run_id = persist_metrics(metrics)
        persist_screening(screening, run_id=run_id)
        print(f"Metricas y screening tambien persistidos en DuckDB (run_id={run_id})")
    except ImportError:
        pass  # duckdb es opcional -- el pipeline principal no depende de el


if __name__ == "__main__":
    run_full_pipeline()
