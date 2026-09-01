"""
Arquitectura hibrida Conv1D + Multi-Head Attention para clasificacion de la
direccion de la siguiente vela en 3 clases (Bajista / Neutro / Alcista)
sobre una ventana temporal de velas OHLCV -- a diferencia de
`train_classifier.py`, que consume una sola fila de features tabulares por
prediccion, esta arquitectura consume una secuencia de `WINDOW_LENGTH`
velas consecutivas.

Motivacion de la arquitectura hibrida:
- Las capas **Conv1D** (sobre el eje temporal, con los features tecnicos
  como canales de entrada) extraen patrones locales de corto plazo entre
  velas consecutivas -- el equivalente aprendido de un patron de velas
  japonesas de 2-3 barras, sin tener que definirlo a mano.
- La capa de **Multi-Head Self-Attention**, aplicada sobre la secuencia de
  salida del bloque convolucional, permite a cada posicion temporal
  "mirar" cualquier otra posicion de la ventana con un peso aprendido,
  capturando dependencias de mas largo alcance dentro de la ventana (p.ej.
  una vela de rango amplio 15 barras atras que sigue siendo relevante para
  la prediccion actual) que una CNN pura -- limitada por el tamano de su
  kernel y su campo receptivo -- no puede ver directamente sin apilar
  muchas mas capas convolucionales.

Formalmente, para consultas/llaves/valores Q, K, V (proyecciones lineales
de la misma secuencia -- self-attention, Q=K=V=h):

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

y Multi-Head Attention corre `h` de estas en paralelo sobre subespacios
proyectados distintos y concatena el resultado:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

El escalado por `1/sqrt(d_k)` evita que el producto punto crezca en
magnitud con la dimension `d_k` de cada cabeza y empuje al softmax hacia
una region de gradiente casi nulo.
"""

from __future__ import annotations

import torch
from torch import nn

WINDOW_LENGTH = 24  # 24 velas de 1h = 1 dia de contexto
CONV_CHANNELS = (32, 64)
CONV_KERNEL_SIZE = 3
N_HEADS = 4
FEEDFORWARD_DIM = 128
DROPOUT = 0.2
N_CLASSES = 3

CLASS_NAMES = ["Bajista", "Neutro", "Alcista"]  # indices 0, 1, 2


class Conv1DAttentionClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_length: int = WINDOW_LENGTH,
        conv_channels: tuple[int, int] = CONV_CHANNELS,
        n_heads: int = N_HEADS,
        feedforward_dim: int = FEEDFORWARD_DIM,
        dropout: float = DROPOUT,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
        )

        # Embedding posicional aprendido: la atencion es invariante a
        # permutaciones por construccion, asi que sin esto el modelo no
        # tendria forma de distinguir "vela mas reciente" de "vela mas
        # antigua" dentro de la ventana.
        self.positional_embedding = nn.Parameter(torch.zeros(1, window_length, c2))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=c2, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(c2)

        self.feedforward = nn.Sequential(
            nn.Linear(c2, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, c2),
        )
        self.ff_norm = nn.LayerNorm(c2)

        self.classifier_head = nn.Sequential(
            nn.Linear(c2, c2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c2 // 2, n_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (batch, window_length, n_features)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # -> (batch, window_length, c2)
        h = h + self.positional_embedding

        attn_out, attn_weights = self.attention(
            h, h, h, need_weights=return_attention, average_attn_weights=True
        )
        h = self.attn_norm(h + attn_out)  # residual + LayerNorm, como en un encoder Transformer estandar

        ff_out = self.feedforward(h)
        h = self.ff_norm(h + ff_out)

        pooled = h[:, -1, :]  # ultima posicion temporal = la vela mas reciente de la ventana
        logits = self.classifier_head(pooled)

        if return_attention:
            return logits, attn_weights
        return logits
