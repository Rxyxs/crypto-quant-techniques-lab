"""Modelo de fricciones de trading: comisiones maker/taker + slippage
dinamico por volatilidad, aplicado sobre los retornos brutos de una
estrategia de señal long/flat.

Dos componentes, cada uno con su propia justificacion:

1. **Comisiones maker/taker.** Este motor de backtest actua sobre el cierre
   de una vela ya conocida y necesita entrar/salir en la siguiente apertura
   -- eso es una orden de mercado (taker), no una orden limite que espera a
   que alguien mas cruce el spread (maker). `taker_fee_bps` es el costo real
   que se aplica en cada cambio de posicion; `maker_fee_bps` se deja
   disponible y documentado como el escenario alternativo (una ejecucion mas
   paciente, via orden limite) para comparar cuanto edge se recupera si la
   estrategia pudiera permitirse esperar el fill.
2. **Slippage dinamico por volatilidad.** El slippage no es una constante:
   en mercados cripto, el spread efectivo y el impacto de mercado se amplian
   con la volatilidad realizada (menos liquidez disponible cerca del precio
   medio cuando el mercado se mueve rapido). Se modela como
   `slippage_bps = base_bps + vol_multiplier * volatilidad_diaria_reciente`,
   usando la misma feature de volatilidad rolling (`vol_20d`) ya calculada
   para el modelo -- ningun input nuevo, solo una funcion de costo distinta
   sobre una señal que ya existe.

Ambos costos se cobran solo en los dias en que la posicion realmente cambia
(turnover), proporcional al tamaño del cambio de posicion -- no en cada dia
que la posicion se mantiene sin tocar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Supuestos de costo (basis points = 1/100 de 1%) ---
# Binance spot, tarifa base VIP 0 sin descuento BNB: 10 bps tanto maker como
# taker. Se usa una tarifa taker realista (ligeramente mas alta que la
# minima con todos los descuentos posibles, pero representativa de un
# trader retail sin volumen ni holding de BNB) y una maker de referencia
# para el escenario "que pasaria si pudiera esperar el fill".
DEFAULT_TAKER_FEE_BPS = 10.0
DEFAULT_MAKER_FEE_BPS = 8.0

# Slippage: un piso base (spread efectivo minimo en el par mas liquido de
# Binance) mas un termino proporcional a la volatilidad diaria reciente.
DEFAULT_SLIPPAGE_BASE_BPS = 1.5
DEFAULT_SLIPPAGE_VOL_MULTIPLIER = 40.0


def compute_dynamic_slippage_bps(
    volatility: pd.Series,
    base_bps: float = DEFAULT_SLIPPAGE_BASE_BPS,
    vol_multiplier: float = DEFAULT_SLIPPAGE_VOL_MULTIPLIER,
) -> pd.Series:
    """Slippage esperado (en bps) como funcion de la volatilidad diaria
    reciente (desviacion estandar de retornos, ya rezagada -- no debe
    incluir el retorno del propio dia que se esta costeando)."""
    return base_bps + vol_multiplier * volatility.fillna(volatility.median())


def apply_frictions(
    bt: pd.DataFrame,
    volatility_col: str = "vol_20d",
    taker_fee_bps: float = DEFAULT_TAKER_FEE_BPS,
    maker_fee_bps: float = DEFAULT_MAKER_FEE_BPS,
    slippage_base_bps: float = DEFAULT_SLIPPAGE_BASE_BPS,
    slippage_vol_multiplier: float = DEFAULT_SLIPPAGE_VOL_MULTIPLIER,
) -> pd.DataFrame:
    """Agrega columnas de retorno neto de fricciones a un DataFrame de
    backtest ya construido (con columnas `signal` y `strategy_return`,
    como las que produce `backtest_engine.run_backtest`).

    Agrega:
        turnover              -- |cambio de posicion| en cada fila (0, 1)
        slippage_bps          -- slippage dinamico estimado ese dia
        cost_taker             -- costo (fraccion de retorno) usando ordenes taker
        cost_maker              -- costo alternativo usando ordenes maker
        strategy_return_net_taker -- retorno neto de comisiones taker + slippage
        strategy_return_net_maker -- retorno neto de comisiones maker + slippage
        equity_curve_net_taker / equity_curve_net_maker
    """
    out = bt.copy()
    position_change = out["signal"].diff().abs().fillna(out["signal"].iloc[0])
    out["turnover"] = position_change

    out["slippage_bps"] = compute_dynamic_slippage_bps(
        out[volatility_col], slippage_base_bps, slippage_vol_multiplier
    )

    cost_bps_taker = taker_fee_bps + out["slippage_bps"]
    cost_bps_maker = maker_fee_bps + out["slippage_bps"]

    out["cost_taker"] = out["turnover"] * cost_bps_taker / 10_000.0
    out["cost_maker"] = out["turnover"] * cost_bps_maker / 10_000.0

    out["strategy_return_net_taker"] = out["strategy_return"] - out["cost_taker"]
    out["strategy_return_net_maker"] = out["strategy_return"] - out["cost_maker"]

    out["equity_curve_net_taker"] = (1 + out["strategy_return_net_taker"]).cumprod()
    out["equity_curve_net_maker"] = (1 + out["strategy_return_net_maker"]).cumprod()

    running_max_taker = out["equity_curve_net_taker"].cummax()
    out["drawdown_net_taker"] = out["equity_curve_net_taker"] / running_max_taker - 1

    return out


def total_friction_cost_summary(bt_with_frictions: pd.DataFrame) -> dict[str, float]:
    """Resume el impacto agregado de las fricciones sobre el periodo de
    backtest completo: cuanto retorno se perdio en comisiones vs. slippage."""
    n_trades = int(bt_with_frictions["turnover"].sum())
    total_fee_drag = float((bt_with_frictions["turnover"] * DEFAULT_TAKER_FEE_BPS / 10_000.0).sum())
    total_slippage_drag = float(
        (bt_with_frictions["turnover"] * bt_with_frictions["slippage_bps"] / 10_000.0).sum()
    )
    return {
        "n_position_changes": n_trades,
        "avg_slippage_bps": float(bt_with_frictions.loc[bt_with_frictions["turnover"] > 0, "slippage_bps"].mean()),
        "total_fee_drag_pct": total_fee_drag * 100,
        "total_slippage_drag_pct": total_slippage_drag * 100,
        "total_friction_drag_pct": (total_fee_drag + total_slippage_drag) * 100,
    }
