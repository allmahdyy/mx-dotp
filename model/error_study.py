"""How much error does anchored truncation add, per MX format and width W?

    python model/error_study.py                 # both experiments, default sizes
    python model/error_study.py --rows 200      # quicker

Experiment "block": one K=32 block dot product with an FP32 addend c drawn
at the same magnitude as the dot product.  Reports how often the truncated
result differs from the exact one, and the worst/mean difference in ulps.

Experiment "chain": a GEMM-style row of length L (default 4096) computed as
L/32 chained block dot products, r fed back as c.  Reports the relative
error of the final FP32 value against the true real-valued dot product of
the MX-quantised inputs (so quantisation error is excluded: only the
accumulation policy differs), for exact and for each truncation width.

Data: Gaussian N(0,1), heavy-tailed Student-t (nu=3), and Gaussian with ~1%
outliers 20-100x larger (mimicking LLM activation outlier channels), all
quantised to MX with the OCP scale rule.  Synthetic only: no real model
tensors are used.  Seeds are fixed; results go to results/.
"""
import argparse
import csv
import math
import random
import statistics
import struct
from fractions import Fraction
from pathlib import Path

from mx_formats import FORMATS
from mx_policies import dot_exact, dot_trunc, terms
from mx_quant import Quantiser

ROOT = Path(__file__).resolve().parents[1]
K = 32
WIDTHS = [16, 20, 24, 25, 26, 28, 32]


def ordinal(bits: int) -> int:
    """Map FP32 bits to integers that are monotone in the real value."""
    return -(bits & 0x7FFFFFFF) if bits >> 31 else bits


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def draw(rng, dist, n):
    if dist == "gauss":
        return [rng.gauss(0, 1) for _ in range(n)]
    if dist == "outlier":
        # LLM-activation-like: Gaussian with ~1% outliers 20-100x larger
        return [rng.gauss(0, 1) * (rng.uniform(20, 100) if rng.random() < 0.01 else 1)
                for _ in range(n)]
    # Student-t with 3 degrees of freedom: heavy tails, like LLM activations
    return [rng.gauss(0, 1) / math.sqrt(rng.gammavariate(1.5, 2.0) / 3) for _ in range(n)]


def exact_value(fa, fb, a, b, xa, xb) -> Fraction:
    t = terms(fa, fb, a, b, xa, xb, 0)
    return sum((Fraction(m) * Fraction(2) ** e for m, e, _ in t[1]), Fraction(0))


def block_experiment(fmt, dist, rows, rng):
    q = Quantiser(fmt)
    diff = {w: [] for w in WIDTHS}
    for _ in range(rows):
        a, xa = q.block(draw(rng, dist, K))
        b, xb = q.block(draw(rng, dist, K))
        d = f32(dot_exact(fmt, fmt, a, b, xa, xb, 0))
        c = struct.unpack("<I", struct.pack("<f", d * rng.uniform(-2, 2)))[0]
        ref = dot_exact(fmt, fmt, a, b, xa, xb, c)
        for w in WIDTHS:
            diff[w].append(abs(ordinal(dot_trunc(fmt, fmt, a, b, xa, xb, c, w)) - ordinal(ref)))
    out = []
    for w in WIDTHS:
        dv = diff[w]
        out.append({"exp": "block", "fmt": fmt.name, "dist": dist, "policy": f"trunc{w}",
                    "differs_pct": 100.0 * sum(1 for x in dv if x) / len(dv),
                    "max_ulp": max(dv), "mean_ulp": statistics.fmean(dv)})
    return out


def chain_experiment(fmt, dist, rows, length, rng):
    q = Quantiser(fmt)
    policies = {"exact": None, **{f"trunc{w}": w for w in WIDTHS}}
    rel = {p: [] for p in policies}
    for _ in range(rows):
        blocks = []
        true = Fraction(0)
        for _ in range(length // K):
            a, xa = q.block(draw(rng, dist, K))
            b, xb = q.block(draw(rng, dist, K))
            blocks.append((a, b, xa, xb))
            true += exact_value(fmt, fmt, a, b, xa, xb)
        if true == 0:
            continue
        for p, w in policies.items():
            acc = 0
            for a, b, xa, xb in blocks:
                acc = (dot_exact(fmt, fmt, a, b, xa, xb, acc) if w is None
                       else dot_trunc(fmt, fmt, a, b, xa, xb, acc, w))
            rel[p].append(abs((Fraction(f32(acc)) - true) / true))
    out = []
    for p in policies:
        rv = sorted(float(x) for x in rel[p])
        out.append({"exp": f"chain{length}", "fmt": fmt.name, "dist": dist, "policy": p,
                    "median_rel_err": statistics.median(rv),
                    "p99_rel_err": rv[int(0.99 * (len(rv) - 1))],
                    "max_rel_err": rv[-1]})
    return out


def write(rows, name):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    cols = list(rows[0].keys())
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fmt_v = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
    with open(out / f"{name}.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt_v(r[c]) for c in cols) + " |\n")
    print(f"wrote results/{name}.csv and .md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000, help="blocks for the block experiment")
    ap.add_argument("--chains", type=int, default=100, help="rows for the chain experiment")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--formats", default=",".join(FORMATS))
    args = ap.parse_args()
    fmts = [FORMATS[n] for n in args.formats.split(",")]

    block_rows, chain_rows = [], []
    for fmt in fmts:
        for dist in ("gauss", "t3", "outlier"):
            rng = random.Random(f"{fmt.name}-{dist}")
            block_rows += block_experiment(fmt, dist, args.rows, rng)
            chain_rows += chain_experiment(fmt, dist, args.chains, args.length, rng)
            print(f"done {fmt.name} {dist}", flush=True)
    write(block_rows, "error_block")
    write(chain_rows, f"error_chain{args.length}")


if __name__ == "__main__":
    main()
