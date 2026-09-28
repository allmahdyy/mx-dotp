"""Build and run the cocotb testbench across configurations.

    python -m pytest tb -q                    # default sweep
    SIM=verilator python -m pytest tb -q      # icarus is the default
    NVEC=20000 python -m pytest tb -q -k e4m3
    python -m pytest tb -q -k trunc           # only the truncating unit
"""
import os
from pathlib import Path

import pytest

try:
    from cocotb_tools.runner import get_results, get_runner   # cocotb >= 2.0
except ImportError:
    from cocotb.runner import get_results, get_runner         # cocotb 1.8 / 1.9

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "rtl").glob("*.v"))
CODES = {"e4m3": 0, "e5m2": 1, "e3m2": 2, "e2m3": 3, "e2m1": 4, "int8": 5}
TOP = {"exact": "mx_dotp", "trunc": "mx_dotp_trunc"}

# (policy, TW, fmt_a, fmt_b, K, PIPE)
CONFIGS = (
    # exact unit
    [("exact", 0, f, f, 32, 0b0000) for f in CODES]
    + [("exact", 0, f, f, 32, 0b1111) for f in ("e4m3", "e5m2", "e2m1")]
    + [("exact", 0, "e4m3", "e4m3", k, p) for k, p in ((1, 0b1000), (2, 0b0101), (8, 0b0110))]
    + [("exact", 0, "e4m3", "e5m2", 32, 0b0011), ("exact", 0, "e2m1", "int8", 16, 0b1001)]
    # truncating unit
    + [("trunc", 24, f, f, 32, 0b0000) for f in CODES]
    + [("trunc", w, "e4m3", "e4m3", 32, 0b0000) for w in (12, 16, 20, 28, 40)]
    + [("trunc", 24, "e5m2", "e5m2", 32, 0b1111), ("trunc", 16, "e2m1", "e2m1", 32, 0b0110)]
    + [("trunc", 24, "e4m3", "e4m3", k, p) for k, p in ((1, 0b1000), (8, 0b0101))]
    + [("trunc", 26, "e4m3", "e5m2", 32, 0b0011), ("trunc", 24, "e2m1", "int8", 16, 0b1001)]
)


def cfg_id(c):
    pol, tw, a, b, k, p = c
    return f"{pol}{tw or ''}-{a}x{b}-k{k}-p{p:04b}"


@pytest.mark.parametrize("policy,tw,fa,fb,k,pipe", CONFIGS, ids=[cfg_id(c) for c in CONFIGS])
def test_mx_dotp(policy, tw, fa, fb, k, pipe):
    sim = os.environ.get("SIM", "icarus")
    params = {"FMT_A": CODES[fa], "FMT_B": CODES[fb], "K": k, "PIPE": pipe}
    if policy == "trunc":
        params["TW"] = tw
    tag = cfg_id((policy, tw, fa, fb, k, pipe))
    top = TOP[policy]
    runner = get_runner(sim)
    build_args = ["-Wno-WIDTH", "-Wno-UNOPTFLAT"] if sim == "verilator" else []
    runner.build(sources=SOURCES, hdl_toplevel=top, parameters=params,
                 build_dir=ROOT / "sim_build" / sim / tag, build_args=build_args,
                 timescale=("1ns", "1ps"), always=True)
    env = {"FMT_A": str(CODES[fa]), "FMT_B": str(CODES[fb]), "K": str(k),
           "POLICY": policy, "TW": str(tw),
           "NVEC": os.environ.get("NVEC", "2000"), "SEED": os.environ.get("SEED", "1")}
    xml = runner.test(hdl_toplevel=top, test_module="test_mx_dotp",
                      test_dir=Path(__file__).parent, extra_env=env,
                      build_dir=ROOT / "sim_build" / sim / tag)
    ntests, nfail = get_results(xml)
    assert ntests > 0 and nfail == 0
