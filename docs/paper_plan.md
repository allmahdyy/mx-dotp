# Paper plan

**Working title:** The Cost of Exactness in Microscaling Dot Products: An Open,
Bit-Accurate Study Across All OCP MX Formats

**Target venue:** IEEE ARITH (check the ARITH 2027 call for dates).
Fallbacks: ISCA/MICRO workshops, FPL.

## Question

MX dot-product units make different, hard-wired choices for how precisely the
products are summed:

| Policy | Who | What |
|---|---|---|
| Exact | Arm (Lutz et al., ARITH 2024), this repo | all products on a fixed-point grid, one final rounding |
| Anchored truncation to W bits | NVIDIA tensor cores (Fasi et al.), Ten-Four | align every term to the largest exponent, drop bits below W |
| Reduced-precision accumulation | KU Leuven (Cuyckens et al., 2025) | FP partial sums with a short (16-bit) mantissa |

Nobody measures the error and the hardware cost of these choices side by side,
per format, on the same flow.  This paper does.

**Hypothesis:** exactness is nearly free for the narrow-range formats (FP4,
FP6, INT8: the exact grid for K=32 is 13-30 bits, no wider than a truncated
datapath) and expensive only for E5M2 (69 bits).  If true, there is a
per-format crossover and a simple design rule.

## Contributions

1. One parameterised RTL code base with an accumulation-policy parameter, so
   every policy is compared on identical surroundings.
2. Bit-accurate Python models for every policy, each checked against an exact
   rational reference; RTL checked bit-exact against its model in cocotb.
3. Error measured exactly against the exact reference: single blocks and long
   GEMM-style accumulations, on synthetic data and real LLM tensors quantised
   to MX.
4. Area / delay / power on ASAP7 and sky130 for format x K x policy x W, with
   power from gate-level switching activity (not vectorless).
5. Everything open and reproducible.

## Work items

- [x] Exact reference, exact RTL, cocotb verification
- [x] Policy models (exact fast path, anchored truncation) + tests
- [x] Error harness: synthetic distributions, single block and chained
- [x] Truncated-datapath RTL, verified against its model (17 configs)
- [ ] Exact vs truncated PPA (running: ASAP7 place-and-route + synth sweep)

## First error results (synthetic data, results/error_*.md)

- Stress test (single block, c at the dot product's magnitude): W=24 changes
  28-37% of FP results and 79-86% of INT8 results; W=32 changes <= 0.35%
  (INT8 1.5%).  INT8 is the most truncation-sensitive format.
- GEMM rows of 4096: W >= 24 matches exact to within FP32 rounding for all
  formats; W = 16-20 loses 1e-3..1e-2 relative accuracy (one FP6 heavy-tail
  case 33%).  FP6 / FP4 / INT8 exact accumulation is bit-for-bit the true
  value (the whole sum fits in FP32).
- Outlier data (Gaussian with ~1% elements 20-100x, mimicking LLM activation
  outlier channels; synthetic, no real model tensors): FP formats stay at the
  FP32 floor with W=24, but INT8 does not - chain p99 relative error 2.1e-4
  at W=24 vs 5e-8 exact (~4000x), and needs W~28 to match.  INT8 is the
  format where exact is both cheaper (area 0.75x, energy 0.83x) and
  necessary for accuracy.
- Consequence: accuracy alone does not force exactness at W >= 24; the case
  for exact rests on worst-case cancellation, order-independent
  (bit-reproducible) results, and whether it is cheap, which the PPA
  comparison has to settle.
- [ ] Gate-level-activity power flow
- [ ] Full PPA sweep, both PDKs
- [ ] Real LLM tensors (needs a model checkpoint download)
- [ ] Read closest papers in full: Lutz 2024, Desrentes 2023, Cuyckens 2025,
      Ten-Four, van Baalen 2023 (hardware-cost figure), Fasi et al. on tensor
      cores
- [ ] Write

## Method decisions (and why)

- **Area:** from synthesis (ORFS/Yosys+ABC).  Exact/trunc ratios after
  synthesis matched full place-and-route within ~1% (FP4 0.55 vs 0.54, FP8
  E4M3 1.18 vs 1.17).  Re-synthesis noise is up to ~3%; only claim ratios
  that clear it.
- **Timing and power:** stop after CTS, wires estimated from placement.
  Placement-only power omitted the clock network (pipelined FP8: 3.64 pJ vs
  7.11 pJ routed); after CTS it was 7.18 pJ (1% off).
- **Power:** gate-level simulation of the netlist with realistic MX data
  (GEMM rows, c fed back), VCD into OpenROAD; 100% of pins annotated; every
  run also re-checks the netlist bit-exactly against the model.  Zero-delay
  simulation: glitch power is not included.  Vectorless power was ~76x high
  and is never reported.
- **Primary speed/energy metric:** single-cycle (combinational) units, so no
  register placement is involved.  Pipelined results are secondary, with the
  same four cut positions in both units (P0 before alignment).
- **Retiming rejected:** ABC retiming (SYNTH_RETIME_MODULES) made the exact
  unit slower (1.65 -> 1.85 ns) and ~1.4x the energy, and nearly tripled the
  truncating unit's flip-flops (632 -> 1751, energy ~2x).  Unpredictable,
  so not a fair balancer.

## Known weaknesses to address

- Open-source flow maps adders poorly (deep FA chains), so absolute delays
  are pessimistic.  Either improve the mapping or show conclusions are
  insensitive to it (relative comparisons on one flow).
- Novelty is a careful comparison, not a new circuit: the takeaway (crossover
  and design rule) must be crisp.

## Related work (found so far)

- Lutz et al., Fused FP8 4-Way Dot Product With Scaling and FP32 Accumulation, ARITH 2024
- Desrentes, Dupont de Dinechin, Le Maire, Exact Dot Product Accumulate Operators for 8-bit FP Deep Learning, DSD 2023
- Cuyckens et al., Precision-Scalable Microscaling Datapaths with Optimized Reduction Tree, arXiv 2511.06313
- Rout, Tine, Ten-Four: An Open-Source Fused Dot Product Unit, arXiv 2512.00053
- Islamoglu et al., MXDOTP, ASAP 2025
- Samson et al., Exploring FPGA designs for MX and beyond, FPL 2024
- Mekhemer et al., Jack of All Scales, arXiv 2607.13898
- van Baalen et al., FP8 versus INT8 for efficient deep learning inference, arXiv 2303.17951
- Kaul et al., Optimized Fused FP Many-Term Dot-Product Hardware, ARITH 2019
- Hickmann et al., Intel NNP-T Fused FP Many-Term Dot Product, ARITH 2020
- Uguen, de Dinechin, Design-space exploration for the Kulisch accumulator
- Rouhani et al., OCP Microscaling Formats (MX) v1.0 Specification, 2023
