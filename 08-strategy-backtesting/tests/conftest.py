import sys
from pathlib import Path

# Permite `import backtest_engine`, `import frictions`, etc. al correr
# pytest desde la raiz del repo o desde tests/, sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
