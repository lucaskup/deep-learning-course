"""Gera os dados da figura de gradiente estilhaçado (shattered gradients).

Reproduz o experimento da Figura 1/2 de Balduzzi et al. (2017), "The shattered
gradients problem": redes ReLU com batch norm, 200 neurônios por camada, entrada
e saída escalares. Calcula dy/dx em uma grade densa de entradas (modo direto,
propagando a derivada junto com a ativação) e a autocorrelação do gradiente em
função da distância entre duas entradas.

Escreve em slides/data/02-cnn/:
  shattered-grad.csv   x, shallow, deep, res   (gradiente normalizado, 1 rede)
  shattered-acorr.csv  dist, shallow, deep, res (autocorrelação média)

Uso (a partir da raiz do repositório):
    python scripts/export_shattered_gradients.py
"""

import argparse
from pathlib import Path

import numpy as np

WIDTH = 200
CONFIGS = {
    "shallow": dict(depth=1, res=False),   # 1 camada oculta
    "deep": dict(depth=24, res=False),     # 24 camadas
    "res": dict(depth=50, res=True),       # 50 blocos residuais
}


def input_gradient(x, depth, res, rng):
    """dy/dx para cada entrada de x, numa rede inicializada aleatoriamente."""
    w0 = rng.normal(0, 1, (WIDTH, 1))
    b0 = rng.normal(0, 1, (WIDTH, 1))
    h = w0 @ x[None, :] + b0
    dh = np.repeat(w0, len(x), axis=1)
    for _ in range(depth - 1):
        # batch norm com estatísticas do lote (tratadas como constantes)
        mu = h.mean(axis=1, keepdims=True)
        sd = h.std(axis=1, keepdims=True) + 1e-8
        z, dz = (h - mu) / sd, dh / sd
        on = z > 0
        a, da = z * on, dz * on
        w = rng.normal(0, np.sqrt(2 / WIDTH), (WIDTH, WIDTH))
        u, du = w @ a, w @ da
        h, dh = (h + u, dh + du) if res else (u, du)
    v = rng.normal(0, np.sqrt(1 / WIDTH), (1, WIDTH))
    return (v @ (dh * (h > 0)))[0]


def autocorrelation(g, lags):
    g = (g - g.mean()) / g.std()
    return np.array([np.mean(g[: len(g) - k] * g[k:]) for k in lags])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nets", type=int, default=20, help="redes por média")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path("slides/data/02-cnn")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # grade densa para a autocorrelação, distâncias log-espaçadas
    x = np.linspace(-2, 2, 20001)
    dx = x[1] - x[0]
    lags = np.unique(np.round(np.logspace(0, np.log10(5000), 40)).astype(int))
    dist = lags * dx

    grads, acorrs = {}, {}
    for name, cfg in CONFIGS.items():
        gs = [input_gradient(x, rng=rng, **cfg) for _ in range(args.nets)]
        acorrs[name] = np.mean([autocorrelation(g, lags) for g in gs], axis=0)
        g = gs[0][::20]  # 1001 pontos bastam para o gráfico
        grads[name] = g / g.std()

    cols = list(CONFIGS)
    np.savetxt(out / "shattered-grad.csv",
               np.column_stack([x[::20]] + [grads[c] for c in cols]),
               delimiter=",", header="x," + ",".join(cols), comments="",
               fmt="%.5f")
    np.savetxt(out / "shattered-acorr.csv",
               np.column_stack([dist] + [acorrs[c] for c in cols]),
               delimiter=",", header="dist," + ",".join(cols), comments="",
               fmt="%.5f")
    for c in cols:
        print(f"{c:8s} autocorrelação em d={dist[0]:.4f}: {acorrs[c][0]:.2f}, "
              f"d={dist[len(dist)//2]:.3f}: {acorrs[c][len(dist)//2]:.2f}")


if __name__ == "__main__":
    main()
