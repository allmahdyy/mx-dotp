"""cocotb testbench for rtl/mx_dotp.v (exact) and rtl/mx_dotp_trunc.v.

Every output is compared bit-for-bit against a model: model/mx_ref.py
(exact rational arithmetic, one RNE rounding) for the exact unit, and
model/mx_policies.py dot_trunc(w=TW) for the truncating unit.
Configuration comes from the environment, set by tb/test_runner.py:
FMT_A, FMT_B, K, PIPE, POLICY (exact|trunc), TW, NVEC, SEED.
"""
import os
import random
import sys
from collections import deque
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from mx_formats import BY_CODE           # noqa: E402
from mx_policies import dot_trunc        # noqa: E402
from mx_ref import mx_dot                # noqa: E402
from stimulus import Stimulus            # noqa: E402

FA = BY_CODE[int(os.environ.get("FMT_A", 0))]
FB = BY_CODE[int(os.environ.get("FMT_B", 0))]
K = int(os.environ.get("K", 32))
NVEC = int(os.environ.get("NVEC", 2000))
SEED = int(os.environ.get("SEED", 1))
POLICY = os.environ.get("POLICY", "exact")
TW = int(os.environ.get("TW", 24))


def model(a, b, xa, xb, c):
    if POLICY == "trunc":
        return dot_trunc(FA, FB, a, b, xa, xb, c, TW)
    return mx_dot(FA, FB, a, b, xa, xb, c)


def pack(codes, width):
    return sum(x << (i * width) for i, x in enumerate(codes))


def directed_vectors():
    """Hand-picked corners the random stimulus might take a while to hit."""
    z = [0] * K
    one_a = {"e4m3": 0x38, "e5m2": 0x3C, "e3m2": 0x0C, "e2m3": 0x08, "e2m1": 0x2, "int8": 0x40}
    oa, ob = one_a[FA.name], one_a[FB.name]
    nz_a, nz_b = 1 << (FA.width - 1), 1 << (FB.width - 1)
    return [
        (z, z, 127, 127, 0x00000000),                  # +0
        ([nz_a] * K, z, 127, 127, 0x80000000),         # -0 * +0 + -0 = -0
        ([nz_a] * K, z, 127, 127, 0x00000000),         # -0 + +0 = +0
        ([oa] * K, [ob] * K, 127, 127, 0x00000000),    # K
        ([oa] * K, [ob] * K, 254, 254, 0x00000000),    # overflow -> +Inf
        ([oa] * K, [ob] * K, 0, 0, 0x00000000),        # underflow -> 0
        ([oa] * K, [ob] * K, 0, 1, 0x00000001),        # tiny + min subnormal
        ([oa] * K, [ob] * K, 127, 127, 0x7F7FFFFF),    # max normal c
        ([oa] * K, [ob] * K, 255, 127, 0x00000000),    # NaN scale
        ([oa] + z[1:], [ob] + z[1:], 127, 127, 0xBF800000),  # 1 - 1 = +0
        ([oa] + z[1:], [ob] + z[1:], 127, 127, 0xFF800000),  # -Inf
        ([oa] + z[1:], [ob] + z[1:], 127, 127, 0x7FC00001),  # NaN c
    ]


@cocotb.test()
async def mx_dotp_vs_reference(dut):
    try:
        clk = Clock(dut.clk, 10, unit="ns")
    except TypeError:                           # cocotb < 2.0
        clk = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clk.start())

    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.xa.value = 0
    dut.xb.value = 0
    dut.c.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    stim = Stimulus(FA, FB, K, seed=SEED)
    rng = random.Random(SEED + 1)
    todo = deque(directed_vectors())
    todo.extend(stim.vector() for _ in range(NVEC))
    expected = deque()
    checked = errors = 0

    while todo or expected:
        await FallingEdge(dut.clk)
        if todo and rng.random() < 0.85:
            a, b, xa, xb, c = vec = todo.popleft()
            dut.a.value = pack(a, FA.width)
            dut.b.value = pack(b, FB.width)
            dut.xa.value, dut.xb.value, dut.c.value = xa, xb, c
            dut.in_valid.value = 1
            expected.append((vec, model(a, b, xa, xb, c)))
        else:
            dut.in_valid.value = 0
        await ReadOnly()
        if int(dut.out_valid.value):
            assert expected, "out_valid with nothing in flight"
            vec, ref = expected.popleft()
            got = int(dut.r.value)
            checked += 1
            if got != ref:
                errors += 1
                if errors <= 10:
                    dut._log.error("mismatch: got %08x ref %08x for %s", got, ref, vec)
        await RisingEdge(dut.clk)

    dut._log.info("%s x %s K=%d %s: %d vectors checked, %d mismatches",
                  FA.name, FB.name, K, POLICY if POLICY == "exact" else f"trunc{TW}",
                  checked, errors)
    assert errors == 0
