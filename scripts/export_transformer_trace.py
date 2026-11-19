"""Train a toy encoder-decoder transformer on an 8-sentence translation
corpus and export every intermediate value of one forward pass as LaTeX
macros, consumed by the worked example that closes
``slides/04-transformers/01-transformers.tex``.

Outputs (all committed):

- ``slides/data/04-transformers/toy-weights.json``: the trained weights,
  rounded to one decimal place.
- ``slides/data/04-transformers/toy-trace.tex``: macros read in the deck as
  ``\\tfx{Name}`` (whole matrix), ``\\tfx{Name:i}`` (row ``i``) and
  ``\\tfx{Name:i:j}`` (one entry), all 1-indexed.
- ``docs/transformer-step-by-step/js/weights.js``: the same weights for the
  interactive demo, which recomputes the forward pass in JavaScript.

The model follows the deck: row vectors (``x W``), sinusoidal positional
encoding, one single-head attention per sub-block, post-norm LayerNorm with
gamma = 1 and beta = 0, no bias in the attention projections. The trace is
computed with the *rounded* weights, so every number on the slides can be
reproduced by hand from the matrices shown.

Usage (from the repo root)::

    python scripts/export_transformer_trace.py            # rebuild the trace from the JSON
    python scripts/export_transformer_trace.py --train    # retrain, then rebuild both files
    python scripts/export_transformer_trace.py --train --search 8   # compare seeds, write nothing

Retraining on another machine or PyTorch version can land on different
weights. The deck comments on the numbers of the committed model, so reread
the example frames after a retrain.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "slides" / "data" / "04-transformers"
WEIGHTS = OUT_DIR / "toy-weights.json"
TRACE = OUT_DIR / "toy-trace.tex"
DEMO_WEIGHTS = ROOT / "docs" / "transformer-step-by-step" / "js" / "weights.js"

DETS = [("O", "The"), ("Um", "A")]
NOUNS = [("gato", "cat"), ("cão", "dog")]
ADJS = [("preto", "black"), ("branco", "white")]
SRC_VOCAB = [w for w, _ in DETS + NOUNS + ADJS]
TGT_VOCAB = ["<bos>", "<eos>"] + [w for _, w in DETS + NOUNS + ADJS]
# Portuguese puts the adjective after the noun, English before it, so the
# cross-attention has to reorder: "O gato preto" -> "The black cat".
PAIRS = [([d, n, a], [dt, at, nt])
         for (d, dt), (n, nt), (a, at) in itertools.product(DETS, NOUNS, ADJS)]
EXAMPLE = 0  # "O gato preto" -> "The black cat"
ALIGNMENT = [0, 2, 1]  # source position translated at each decoder step

D, D_FF = 4, 4
ATT = ["enc_Wq", "enc_Wk", "enc_Wv", "dec_Wq", "dec_Wk", "dec_Wv", "x_Wq", "x_Wk", "x_Wv"]
SHAPES = {
    "E_src": (len(SRC_VOCAB), D), "E_tgt": (len(TGT_VOCAB), D),
    **{n: (D, D) for n in ATT},
    "enc_W1": (D, D_FF), "enc_b1": (D_FF,), "enc_W2": (D_FF, D), "enc_b2": (D,),
    "dec_W1": (D, D_FF), "dec_b1": (D_FF,), "dec_W2": (D_FF, D), "dec_b2": (D,),
    "W_out": (D, len(TGT_VOCAB)),
}


# ------------------------------------------------------------------ model

def pos_encoding(m: int, d: int, n: float = 10000.0) -> torch.Tensor:
    i = torch.arange(m, dtype=torch.float32)[:, None]
    k = torch.arange(d)[None, :] // 2
    ang = i / (n ** (2 * k / d))
    return torch.where(torch.arange(d)[None, :] % 2 == 0, torch.sin(ang), torch.cos(ang))


def layer_norm(x: torch.Tensor, t: dict | None = None, key: str = "") -> torch.Tensor:
    mu = x.mean(-1, keepdim=True)
    sigma = torch.sqrt(((x - mu) ** 2).mean(-1, keepdim=True))
    if t is not None:
        t[key + "Mu"], t[key + "Sigma"] = mu.squeeze(-1), sigma.squeeze(-1)
        t[key + "Dev"], t[key + "Var"] = x - mu, (sigma ** 2).squeeze(-1)
    return (x - mu) / sigma


def attention(xq, xkv, Wq, Wk, Wv, mask=None):
    Q, K, V = xq @ Wq, xkv @ Wk, xkv @ Wv
    raw = Q @ K.transpose(-1, -2)
    S = raw / math.sqrt(Q.shape[-1])
    if mask is not None:
        S = S.masked_fill(mask, float("-inf"))
    A = F.softmax(S, dim=-1)
    return A @ V, dict(Q=Q, K=K, V=V, Sraw=raw, S=S, A=A)


def ffn(x, p, side):
    hid = torch.relu(x @ p[f"{side}_W1"] + p[f"{side}_b1"])
    return hid @ p[f"{side}_W2"] + p[f"{side}_b2"], hid


def forward(p: dict, src: torch.Tensor, tgt_in: torch.Tensor, trace: bool = False):
    """src: (B, m), tgt_in: (B, n) index tensors. Returns logits (B, n, |V_tgt|)."""
    t: dict = {}
    emb = p["E_src"][src]
    pe = pos_encoding(src.shape[1], D)
    X = emb + pe
    C, a = attention(X, X, p["enc_Wq"], p["enc_Wk"], p["enc_Wv"])
    H1 = layer_norm(X + C, t, "EncLN1")
    Fo, hid = ffn(H1, p, "enc")
    M = layer_norm(H1 + Fo, t, "EncLN2")
    t.update(EncEmb=emb, EncPE=pe.expand_as(emb), EncIn=X, EncQ=a["Q"], EncK=a["K"], EncV=a["V"], EncSraw=a["Sraw"],
             EncS=a["S"], EncA=a["A"], EncC=C, EncSum1=X + C, EncH1=H1, EncHid=hid, EncF=Fo,
             EncSum2=H1 + Fo, EncOut=M)

    n = tgt_in.shape[1]
    demb = p["E_tgt"][tgt_in]
    dpe = pos_encoding(n, D)
    Y = demb + dpe
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    C1, a1 = attention(Y, Y, p["dec_Wq"], p["dec_Wk"], p["dec_Wv"], mask)
    D1 = layer_norm(Y + C1, t, "DecLN1")
    C2, a2 = attention(D1, M, p["x_Wq"], p["x_Wk"], p["x_Wv"])
    D2 = layer_norm(D1 + C2, t, "DecLN2")
    Fd, dhid = ffn(D2, p, "dec")
    D3 = layer_norm(D2 + Fd, t, "DecLN3")
    logits = D3 @ p["W_out"]
    t.update(DecEmb=demb, DecPE=dpe.expand_as(demb), DecIn=Y, DecSelfA=a1["A"], DecC1=C1, DecSum1=Y + C1, DecH1=D1,
             XQ=a2["Q"], XK=a2["K"], XV=a2["V"], XSraw=a2["Sraw"], XS=a2["S"], XA=a2["A"], XC=C2,
             DecSum2=D1 + C2, DecH2=D2, DecHid=dhid, DecF=Fd, DecSum3=D2 + Fd, DecOut=D3,
             Logits=logits, Probs=F.softmax(logits, -1))
    return (logits, t) if trace else logits


# --------------------------------------------------------------- training

def encode(pairs):
    src = torch.tensor([[SRC_VOCAB.index(w) for w in s] for s, _ in pairs])
    tgt = torch.tensor([[TGT_VOCAB.index(w) for w in t] for _, t in pairs])
    bos = torch.full((len(pairs), 1), TGT_VOCAB.index("<bos>"))
    eos = torch.full((len(pairs), 1), TGT_VOCAB.index("<eos>"))
    return src, torch.cat([bos, tgt], 1), torch.cat([tgt, eos], 1)


def train(seed: int, steps: int, lr: float, smoothing: float) -> dict:
    g = torch.Generator().manual_seed(seed)
    p = {}
    for name, shape in SHAPES.items():
        init = torch.zeros(shape) if len(shape) == 1 else torch.randn(shape, generator=g) * 0.5
        p[name] = init.requires_grad_()
    src, tin, tout = encode(PAIRS)
    opt = torch.optim.Adam(p.values(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(forward(p, src, tin).flatten(0, 1), tout.flatten(), label_smoothing=smoothing)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # one decimal place; "+ 0.0" turns -0.0 into 0.0
    return {k: torch.round(v.detach() * 10) / 10 + 0.0 for k, v in p.items()}


@torch.no_grad()
def greedy(p: dict, src: torch.Tensor, steps: int) -> torch.Tensor:
    cur = torch.full((src.shape[0], 1), TGT_VOCAB.index("<bos>"))
    for _ in range(steps):
        nxt = forward(p, src, cur)[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], 1)
    return cur[:, 1:]


@torch.no_grad()
def summary(p: dict) -> dict:
    src, tin, tout = encode(PAIRS)
    acc = (greedy(p, src, tout.shape[1]) == tout).all(1).float().mean().item()
    e = slice(EXAMPLE, EXAMPLE + 1)
    _, t = forward(p, src[e], tin[e], trace=True)
    probs = t["Probs"][0].gather(-1, tout[EXAMPLE][:, None]).squeeze(-1)
    align = [t["XA"][0][step, pos].item() for step, pos in enumerate(ALIGNMENT)]
    return dict(acc=acc, probs=probs, enc_A=t["EncA"][0], x_A=t["XA"][0], align=align,
                max_w=max(v.abs().max().item() for v in p.values()))


# ----------------------------------------------------------------- export

def num(x: float, dec: int) -> str:
    s = f"{x:.{dec}f}"
    return s[1:] if float(s) == 0 and s.startswith("-") else s


def tex_token(w: str) -> str:
    return r"\texttt{" + w.replace("<", r"\textless ").replace(">", r"\textgreater ") + "}"


class Macros:
    def __init__(self):
        self.lines = [
            "% Gerado por scripts/export_transformer_trace.py. Não editar à mão.",
            r"\providecommand{\tfx}[1]{\csname tfx@#1\endcsname}",
        ]

    def put(self, key: str, value: str):
        self.lines.append(rf"\expandafter\def\csname tfx@{key}\endcsname{{{value}}}")

    def tensor(self, key: str, x: torch.Tensor, dec: int):
        """Vector: Name (row matrix), Name:j. Matrix: Name, Name:i (row), Name:i:j."""
        if x.dim() == 2 and x.shape[0] == 1:
            x = x[0]
        rows = [x] if x.dim() == 1 else list(x)
        body = []
        for i, row in enumerate(rows, 1):
            cells = [num(v, dec) for v in row.tolist()]
            body.append(" & ".join(cells))
            prefix = key if x.dim() == 1 else f"{key}:{i}"
            if x.dim() == 2:
                self.put(prefix, r"\begin{bmatrix} " + body[-1] + r" \end{bmatrix}")
            for j, c in enumerate(cells, 1):
                self.put(f"{prefix}:{j}", c)
        self.put(key, r"\begin{bmatrix} " + r" \\ ".join(body) + r" \end{bmatrix}")


WEIGHT_KEYS = {
    "E_src": "Esrc", "E_tgt": "Etgt", "enc_Wq": "EncWq", "enc_Wk": "EncWk", "enc_Wv": "EncWv",
    "enc_W1": "EncW1", "enc_b1": "EncB1", "enc_W2": "EncW2", "enc_b2": "EncB2",
    "dec_Wq": "DecWq", "dec_Wk": "DecWk", "dec_Wv": "DecWv", "x_Wq": "XWq", "x_Wk": "XWk", "x_Wv": "XWv",
    "dec_W1": "DecW1", "dec_b1": "DecB1", "dec_W2": "DecW2", "dec_b2": "DecB2", "W_out": "Wout",
}
ENCODER_KEYS = ["EncEmb", "EncPE", "EncIn", "EncQ", "EncK", "EncV", "EncSraw", "EncS", "EncA", "EncC",
                "EncSum1", "EncH1", "EncHid", "EncF", "EncSum2", "EncOut", "EncLN1Mu", "EncLN1Sigma",
                "EncLN2Mu", "EncLN2Sigma"]
DECODER_KEYS = ["DecEmb", "DecPE", "DecIn", "DecC1", "DecSum1", "DecH1", "XQ", "XK", "XV", "XSraw", "XS",
                "XA", "XC", "DecSum2", "DecH2", "DecHid", "DecF", "DecSum3", "DecOut", "Logits", "Probs",
                "DecLN1Mu", "DecLN1Sigma", "DecLN2Mu", "DecLN2Sigma", "DecLN3Mu", "DecLN3Sigma"]


@torch.no_grad()
def export(p: dict):
    src, tin, tout = encode(PAIRS)
    e = slice(EXAMPLE, EXAMPLE + 1)
    assert (greedy(p, src, tout.shape[1]) == tout).all(), "rounded model no longer translates the corpus"

    m = Macros()
    m.put("NumParams", str(sum(v.numel() for v in p.values())))
    m.put("NumPairs", str(len(PAIRS)))
    for name, key in WEIGHT_KEYS.items():
        m.tensor(key, p[name], 1)

    # Encoder, and the decoder at its first step (only <bos> written so far).
    _, t = forward(p, src[e], tin[e, :1], trace=True)
    for key in ENCODER_KEYS + DECODER_KEYS:
        m.tensor(key, t[key][0], 2)
    # Every LayerNorm also exports its deviations (x - mu) and variance.
    for key in [k[:-2] for k in ENCODER_KEYS + DECODER_KEYS if k.endswith("Mu")]:
        m.tensor(key + "Dev", t[key + "Dev"][0], 2)
        m.tensor(key + "Var", t[key + "Var"][0], 2)

    # Whole translation with teacher forcing: one row per decoder step.
    _, full = forward(p, src[e], tin[e], trace=True)
    m.tensor("FullXA", full["XA"][0], 2)
    m.tensor("FullSelfA", full["DecSelfA"][0], 2)
    m.tensor("FullProbs", full["Probs"][0], 2)
    for step, row in enumerate(full["Probs"][0], 1):
        best = int(row.argmax())
        m.put(f"Pred:{step}", tex_token(TGT_VOCAB[best]))
        m.put(f"PredProb:{step}", num(row[best].item(), 2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE.write_text("\n".join(m.lines) + "\n", encoding="utf-8")
    print(f"wrote {TRACE.relative_to(ROOT)} ({len(m.lines) - 2} macros)")
    export_demo(p)


def export_demo(p: dict):
    """Same weights for the interactive demo, which redoes the forward pass in JS."""
    data = {
        "srcVocab": SRC_VOCAB, "tgtVocab": TGT_VOCAB,
        "dets": [w for w, _ in DETS], "nouns": [w for w, _ in NOUNS], "adjs": [w for w, _ in ADJS],
        "weights": {k: [[round(x, 1) for x in row] for row in v.tolist()] if v.dim() == 2
                    else [round(x, 1) for x in v.tolist()] for k, v in p.items()},
    }
    js = ("/* Gerado por scripts/export_transformer_trace.py. Não editar à mão. */\n"
          "(function () {\n  'use strict';\n  window.DL = window.DL || {};\n"
          f"  DL.toyWeights = {json.dumps(data, ensure_ascii=False)};\n}})();\n")
    DEMO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    DEMO_WEIGHTS.write_text(js, encoding="utf-8")
    print(f"wrote {DEMO_WEIGHTS.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", action="store_true", help="retrain instead of reading toy-weights.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="keeps the output softmax from saturating at 1.00")
    ap.add_argument("--search", type=int, default=0, metavar="N",
                    help="with --train: report seeds 0..N-1 and write nothing")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)

    if args.train and args.search:
        for seed in range(args.search):
            s = summary(train(seed, args.steps, args.lr, args.label_smoothing))
            print(f"\nseed {seed}: acc={s['acc']:.2f} max|w|={s['max_w']:.1f} "
                  f"p(correct)={[round(v, 2) for v in s['probs'].tolist()]} "
                  f"aligned cross-attention={[round(v, 2) for v in s['align']]}")
            print("encoder self-attention:", s["enc_A"], "cross-attention (rows = decoder steps):", s["x_A"],
                  sep="\n")
        return

    if args.train:
        p = train(args.seed, args.steps, args.lr, args.label_smoothing)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        WEIGHTS.write_text(json.dumps({k: v.tolist() for k, v in p.items()}, indent=1) + "\n")
        print(f"wrote {WEIGHTS.relative_to(ROOT)}")
    else:
        p = {k: torch.tensor(v) for k, v in json.loads(WEIGHTS.read_text()).items()}
    export(p)


if __name__ == "__main__":
    main()
