"""Checks for the accumulation-policy models.

    python -m pytest model/test_policies.py -q
"""
import random
import zlib

import pytest

from mx_formats import FORMATS
from mx_policies import dot_exact, dot_trunc, rne_fp32
from mx_ref import mx_dot
from mx_formats import round_to_fp32
from fractions import Fraction
from stimulus import Stimulus

PAIRS = [(n, n) for n in FORMATS] + [("e4m3", "e5m2"), ("e2m1", "e4m3"), ("int8", "e2m3")]


@pytest.mark.parametrize("name", ["e4m3", "e5m2", "e3m2", "e2m3", "e2m1"])
def test_quantiser_matches_ml_dtypes(name):
    import ml_dtypes
    import numpy as np
    from mx_formats import decode_elem
    from mx_quant import Quantiser
    dt = {"e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2,
          "e3m2": ml_dtypes.float6_e3m2fn, "e2m3": ml_dtypes.float6_e2m3fn,
          "e2m1": ml_dtypes.float4_e2m1fn}[name]
    f = FORMATS[name]
    q = Quantiser(f)
    rng = random.Random(5)
    for _ in range(20000):
        x = rng.uniform(-1, 1) * q.vals[-1]                 # inside the finite range
        code = q._elem(x)
        _, s, mag = decode_elem(f, code)
        mine = float(-mag if s else mag)
        assert mine == float(np.array(x, dtype=np.float64).astype(dt)), (x, code)


def test_rne_fp32_matches_reference_rounding():
    rng = random.Random(11)
    for _ in range(100_000):
        n = rng.getrandbits(rng.randint(1, 80)) * rng.choice([-1, 1]) or 1
        lsb = rng.randint(-260, 150)
        assert rne_fp32(n, lsb) == round_to_fp32(Fraction(n) * Fraction(2) ** lsb)


@pytest.mark.parametrize("fa,fb", PAIRS)
@pytest.mark.parametrize("k", [1, 4, 32])
def test_exact_policy_matches_reference(fa, fb, k):
    fa, fb = FORMATS[fa], FORMATS[fb]
    stim = Stimulus(fa, fb, k, seed=zlib.crc32(f"pol{fa.name}{fb.name}{k}".encode()))
    for _ in range(2000):
        v = stim.vector()
        assert dot_exact(fa, fb, *v) == mx_dot(fa, fb, *v), v


@pytest.mark.parametrize("fa,fb", PAIRS)
def test_wide_truncation_is_exact(fa, fb):
    """With W wider than the whole exponent range nothing is dropped."""
    fa, fb = FORMATS[fa], FORMATS[fb]
    stim = Stimulus(fa, fb, 32, seed=zlib.crc32(f"wide{fa.name}{fb.name}".encode()))
    for _ in range(1000):
        v = stim.vector()
        assert dot_trunc(fa, fb, *v, w=700) == dot_exact(fa, fb, *v), v


def test_truncation_drops_small_terms():
    """c = 1.0 plus the product 1.5 * 2^-24 = 0.75 ulp(1.0).

    Exact: 1 + 0.75 ulp rounds up to the next float after 1.0.
    W=24 anchored at 1.0 keeps nothing below 2^-23, so the product vanishes."""
    f = FORMATS["e4m3"]
    a, b = [0x3C], [0x38]                        # 1.5 * 1.0
    c = 0x3F800000                               # 1.0
    xa = xb = 115                                # scale 2^(115+115-254) = 2^-24
    assert dot_exact(f, f, a, b, xa, xb, c) == 0x3F800001
    assert dot_trunc(f, f, a, b, xa, xb, c, w=24) == 0x3F800000
    assert dot_trunc(f, f, a, b, xa, xb, c, w=26) == 0x3F800001


def test_exact_cancellation_gives_positive_zero():
    """1 * -1 + 1.0 cancels exactly: +0 under every policy."""
    f = FORMATS["e4m3"]
    a, b = [0x38], [0xB8]                        # 1 * -1
    assert dot_exact(f, f, a, b, 127, 127, 0x3F800000) == 0
    assert dot_trunc(f, f, a, b, 127, 127, 0x3F800000, w=24) == 0
