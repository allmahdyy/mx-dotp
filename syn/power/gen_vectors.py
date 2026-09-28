"""Stimulus for gate-level power simulation.

Writes GEMM-row-style vectors (realistic MX data, c fed back from the
previous block of the same row) plus the model's expected result, one file
per port, as hex for $readmemh:

    python3 syn/power/gen_vectors.py --fmt e4m3 --k 32 --policy exact --out DIR

Data: Gaussian N(0,1) quantised to MX with the OCP scale rule.  Rows of
`--blocks` blocks each; the first block of a row gets c = 0.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from mx_formats import FORMATS               # noqa: E402
from mx_policies import dot_exact, dot_trunc  # noqa: E402
from mx_quant import Quantiser                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", required=True, help="format of both operands, or A,B")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--policy", default="exact", choices=["exact", "trunc"])
    ap.add_argument("--tw", type=int, default=24)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=32, help="blocks per row")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = args.fmt.split(",")
    fa, fb = FORMATS[names[0]], FORMATS[names[-1]]
    qa, qb = Quantiser(fa), Quantiser(fb)
    rng = random.Random(args.seed)
    model = (dot_exact if args.policy == "exact"
             else lambda *v: dot_trunc(*v, args.tw))

    cols = {n: [] for n in ("a", "b", "xa", "xb", "c", "r")}
    for _ in range(args.rows):
        acc = 0
        for _ in range(args.blocks):
            a, xa = qa.block([rng.gauss(0, 1) for _ in range(args.k)])
            b, xb = qb.block([rng.gauss(0, 1) for _ in range(args.k)])
            r = model(fa, fb, a, b, xa, xb, acc)
            cols["a"].append(sum(x << (i * fa.width) for i, x in enumerate(a)))
            cols["b"].append(sum(x << (i * fb.width) for i, x in enumerate(b)))
            cols["xa"].append(xa)
            cols["xb"].append(xb)
            cols["c"].append(acc)
            cols["r"].append(r)
            acc = r

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, vals in cols.items():
        (out / f"{n}.hex").write_text("".join(f"{v:x}\n" for v in vals))
    (out / "count.txt").write_text(f"{len(cols['r'])}\n")
    print(f"wrote {len(cols['r'])} vectors to {out}")


if __name__ == "__main__":
    main()
