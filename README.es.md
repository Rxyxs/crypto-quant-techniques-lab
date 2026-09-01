[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# Crypto Quant Techniques Lab

Un solo laboratorio, ocho técnicas independientes aplicadas a datos de mercado cripto — detección de señales, microestructura de mercado, construcción de portafolios y NLP. Cada técnica vive en su propia carpeta numerada, con su propio README, dependencias y tests, y puede correr de forma independiente. Este repo reemplaza ocho repos separados de una sola técnica que antes vivían en este perfil; consolidarlos aquí deja más claro el punto real: son variaciones sobre un mismo toolkit (arbitraje estadístico, detección de anomalías, ML supervisado/no supervisado), no ocho proyectos sin relación.

## Técnicas

| # | Técnica | Carpeta | Qué hace |
|---|---|---|---|
| 01 | Clasificación de dirección (deep learning) | [`01-direction-classification-deep-learning`](01-direction-classification-deep-learning) | Redes densas (ReLU vs. Tanh) clasifican la dirección de la siguiente vela sobre datos OHLCV simulados AR(1)/GARCH y datos reales de Binance. |
| 02 | Liquidez e impacto en precio | [`02-liquidity-price-impact`](02-liquidity-price-impact) | Modelo XGBoost del impacto en precio del order book a partir de features sintéticas de profundidad/desbalance. |
| 03 | Pairs trading (cointegración) | [`03-pairs-trading-cointegration`](03-pairs-trading-cointegration) | Screening estadístico de cointegración + hedge ratio dinámico con filtro de Kalman para pairs trading. |
| 04 | Optimización de portafolio (Markowitz) | [`04-portfolio-markowitz-optimization`](04-portfolio-markowitz-optimization) | Construcción de portafolio por frontera eficiente sobre un universo de activos cripto. |
| 05 | Detección de regímenes y correlación | [`05-regime-detection-correlation`](05-regime-detection-correlation) | Detección de regímenes vía Markov switching y su efecto en la correlación entre activos. |
| 06 | Screening de sentimiento (NLP) | [`06-sentiment-nlp-screening`](06-sentiment-nlp-screening) | Sentimiento FinBERT sobre titulares financieros vs. volumen de trading. |
| 07 | Detección de spoofing en order book | [`07-orderbook-spoofing-detection`](07-orderbook-spoofing-detection) | Isolation Forest + autoencoder sobre datos streaming de order book L2, con calibración de alert-budget. |
| 08 | Backtesting de estrategias | [`08-strategy-backtesting`](08-strategy-backtesting) | Motor de backtesting estadístico comparando señales de trading bajo fricciones realistas. |

## Por qué un repo en vez de ocho

Cada técnica es real, ejecutable y probada de forma independiente (ver el README de cada carpeta para setup y resultados) — esto no es esconder alcance, es representarlo con precisión. Ocho repos con el prefijo `crypto-` se leen como ocho proyectos sin relación; una carpeta de laboratorio con ocho técnicas se lee como lo que realmente es: una exploración sistemática del mismo toolkit — detección de anomalías, ML supervisado/no supervisado, series de tiempo estadísticas — aplicado a distintos problemas del mismo mercado.

## Cómo correr una técnica

Cada carpeta es autocontenida:

```bash
cd 0N-nombre-tecnica
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
python <entry_point>.py
```

Ver el README de cada carpeta para el entry point exacto, resultados reales de una corrida real, y cualquier hallazgo negativo honesto.

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE)
