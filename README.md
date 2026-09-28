# mx-dotp

A parameterised, bit-accurate OCP Microscaling (MX) block dot-product unit with
fused FP32 accumulate, verified with cocotb against an exact reference and
pushed through Yosys + OpenROAD on ASAP7 and sky130.

```
r = RNE_fp32( c + 2^(xa-127) · 2^(xb-127) · Σ_{i<K} a_i · b_i )
```

* **Element formats** (OCP MX v1.0): MXFP8 E4M3 / E5M2, MXFP6 E3M2 / E2M3,
  MXFP4 E2M1 and MXINT8, chosen independently for A and B.
* **Shared scales**: E8M0, one per operand block.
* **Exact until the end**: products are exact, the sum is exact and the add
  of `c` is exact.  There is **one** round-to-nearest-even to binary32,
  including subnormal results and overflow to Inf.
* **Parameters**: `FMT_A`, `FMT_B`, `K` (power of two, MX uses 32) and
  `PIPE[3:0]`, which picks any subset of four pipeline cuts.

## Number semantics

The OCP spec leaves dot-product internal precision implementation-defined.
This unit picks the strictest option: the result equals IEEE 754 applied to
the exact expression above.

| Case | Result |
|---|---|
| either scale is `0xFF`, any element or `c` is NaN, Inf × 0, +Inf meets −Inf | `0x7FC00000` |
| any Inf product or Inf `c`, otherwise | ±Inf |
| exact zero | −0 only if every product is −0 and `c` is −0, else +0 |
| finite | single RNE rounding, gradual underflow, overflow → ±Inf |

MXINT8 is 8-bit two's complement × 2⁻⁶, and `0x80` decodes to −2.0.

## Architecture

```
a_i,b_i ─ decode ─ multiply ─ shift onto fixed-point grid ─┐     (K lanes)
                                                           ├─ P0
             signed binary adder tree, +1 bit per level    ├─ P1   exact Σ
   |Σ|, LZC, compare exponents with c, align with sticky, add/sub ─ P2
   LZC normalise, denormalise if tiny, RNE, specials ─ P3 ─ r
```

Every product fits on one fixed-point grid, so the adder tree is exact.  For
example, E5M2 × E5M2 needs 64 bits per product and 69 bits for |Σ| at K=32,
while E2M1 × E2M1 needs 8 and 13 bits.  The final add of `c` uses a window of
`max(|Σ| width, 24) + 3` bits with a sticky LSB, which is enough for one
correct rounding (the argument is in the comments of `model/mx_hw_model.py`).

## Verification

Three layers, each checked against the one above it:

1. **`model/mx_ref.py`** is the golden reference.  It uses exact rational
   arithmetic (`fractions.Fraction`) with no host floating point.  Its element
   decoders are checked against Google's `ml_dtypes` for every code point, and
   its FP32 rounding against numpy's IEEE float64→float32 cast on 200k values
   including forced ties.
2. **`model/mx_hw_model.py`** is a bit-true model of the RTL algorithm with
   the same widths, window, sticky and LZC.  It is checked against the
   reference on directed-random vectors for all formats, mixed pairs and
   K ∈ {1, 2, 4, 32}, and exhaustively for K=1 on the FP4/FP6 formats.
3. **`tb/`** is the cocotb testbench.  It streams vectors into the RTL with
   random `in_valid` gaps and compares every output bit-for-bit with the
   reference, across format, K and pipeline configurations.

Random stimulus is weighted towards corners: special values, extreme and
subnormal elements, sparse blocks, pairwise-cancelling blocks, and `c` chosen
to cancel the dot product almost exactly.

## Running

On Linux or WSL Ubuntu:

```bash
bash scripts/setup_wsl.sh                  # one-time: iverilog, verilator, cocotb, ORFS
python3 -m pytest model -q                 # reference + bit-true model checks
python3 -m pytest tb -q                    # RTL vs reference (Icarus)
SIM=verilator python3 -m pytest tb -q
python3 syn/sweep.py --platform asap7 --set full -j 4
python3 syn/sweep.py --platform sky130hd --set full -j 4
```

The model checks also run on Windows with plain Python.

## Comparison baseline: anchored truncation

`rtl/mx_dotp_trunc.v` is the classic many-term organisation used by
max-exponent-aligned dot-product units: every term is aligned to the
largest nominal exponent, truncated towards zero to a `TW`-bit window,
summed, and rounded once.  Its bit-accurate model is `dot_trunc` in
`model/mx_policies.py`.  `model/error_study.py` measures how far it drifts
from the exact unit (results in `results/error_*.md`); `docs/paper_plan.md`
describes the study this is for.

## PPA

ASAP7, OpenROAD-flow-scripts, after detailed routing.  Method (in
`syn/sweep.py`): zero I/O delay; delay = clock period − worst setup slack.
Power is not reported yet: OpenROAD's vectorless estimate is off by orders
of magnitude on deep arithmetic, so it needs gate-level switching activity.

| Unit | Config | Std-cell area (µm²) | Critical path |
|---|---|---|---|
| exact | E4M3 × E4M3, K=32, combinational | 2809 | 3.80 ns |
| exact | E2M1 × E2M1, K=32, combinational | 951 | 3.04 ns |
| exact | E4M3 × E4M3, K=32, 4 stages | 3247 | 1.66 ns / stage |

Absolute delays are pessimistic: the open flow maps much of the arithmetic
to ripple full-adder chains (the critical path is ~290 cells deep), and the
four pipeline stages are not yet balanced.  Relative comparisons on the same
flow are the meaningful numbers.

## Status

- [x] Exact reference, cross-checked against `ml_dtypes` and numpy
- [x] Bit-true algorithm model, matching the reference
- [x] Exact RTL: 14 cocotb configurations bit-exact on Icarus 12
- [x] Truncating RTL: 17 cocotb configurations bit-exact against its model
- [x] Error study on synthetic MX data (single blocks and 4096-long chains)
- [x] ASAP7 place-and-route of the exact unit (3 configurations)
- [ ] Truncating unit through the same flow; full sweeps; sky130
- [ ] Power from gate-level switching activity
- [ ] Verilator (needs Verilator >= 5.036 for cocotb 2.x) and long random runs
- [ ] Balanced pipeline stages
