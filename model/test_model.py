"""Self-checks for the reference and the bit-true hardware model.

    python -m pytest model -q
"""
import itertools
import random
import struct
import zlib
from fractions import Fraction

import ml_dtypes
import numpy as np
import pytest

from mx_formats import FORMATS, NAN, INF, decode_elem, decode_fp32, round_to_fp32
from mx_hw_model import mx_dot_hw
from mx_ref import mx_dot
from stimulus import Stimulus

ML_DTYPES = {
    "e4m3": ml_dtypes.float8_e4m3fn,
    "e5m2": ml_dtypes.float8_e5m2,
    "e3m2": ml_dtypes.float6_e3m2fn,
    "e2m3": ml_dtypes.float6_e2m3fn,
    "e2m1": ml_dtypes.float4_e2m1fn,
}


@pytest.mark.parametrize("name", sorted(ML_DTYPES))
def test_decode_matches_ml_dtypes(name):
    f = FORMATS[name]
    codes = np.arange(1 << f.width, dtype=np.uint8)
    ref = codes.view(ML_DTYPES[name]).astype(np.float64)
    for x in range(1 << f.width):
        kind, sign, mag = decode_elem(f, x)
        r = ref[x]
        if kind == NAN:
            assert np.isnan(r), (name, x)
        elif kind == INF:
            assert np.isinf(r) and (r < 0) == bool(sign), (name, x)
        else:
            assert float(-mag if sign else mag) == r, (name, x)
            assert np.signbit(r) == bool(sign), (name, x)


def test_decode_int8():
    f = FORMATS["int8"]
    assert decode_elem(f, 0x7F)[2] == Fraction(127, 64)
    assert decode_elem(f, 0x80)[1:] == (1, Fraction(2))
    assert decode_elem(f, 0xFF)[1:] == (1, Fraction(1, 64))


def test_scale_e8m0_matches_ml_dtypes():
    e8 = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    assert np.isnan(e8[255])
    for x in range(255):
        assert e8[x] == 2.0 ** (x - 127)


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(v)))[0]


def test_round_to_fp32_matches_numpy():
    """numpy's float64 -> float32 cast is IEEE RNE; exact doubles give a single rounding."""
    rng = random.Random(7)
    with np.errstate(over="ignore"):
        for _ in range(200_000):
            e = rng.randint(-160, 135)
            v = (1 + rng.random()) * 2.0 ** e * rng.choice([-1, 1])
            if rng.random() < 0.3:                   # force ties and near-ties
                bits = _f32_bits(v)
                v = float(np.float32(struct.unpack("<f", struct.pack("<I", bits))[0]))
                v = v + rng.choice([-1, 1]) * abs(v) * 2.0 ** -25
            if v == 0 or not np.isfinite(v):
                continue
            assert round_to_fp32(Fraction(v)) == _f32_bits(v), v


def test_decode_fp32_roundtrip():
    rng = random.Random(3)
    for _ in range(50_000):
        bits = rng.getrandbits(32)
        kind, sign, mag = decode_fp32(bits)
        if kind == "num" and mag != 0:
            assert round_to_fp32(-mag if sign else mag) == bits


PAIRS = [(n, n) for n in FORMATS] + [("e4m3", "e5m2"), ("e2m1", "e4m3"), ("int8", "e2m3")]


@pytest.mark.parametrize("fa,fb", PAIRS)
@pytest.mark.parametrize("k", [1, 2, 4, 32])
def test_hw_model_matches_reference(fa, fb, k):
    fa, fb = FORMATS[fa], FORMATS[fb]
    stim = Stimulus(fa, fb, k, seed=zlib.crc32(f"{fa.name}{fb.name}{k}".encode()))
    n = 3000 if k == 32 else 6000
    for _ in range(n):
        a, b, xa, xb, c = stim.vector()
        ref = mx_dot(fa, fb, a, b, xa, xb, c)
        hw = mx_dot_hw(fa, fb, a, b, xa, xb, c)
        assert hw == ref, (fa.name, fb.name, a, b, xa, xb, hex(c), hex(ref), hex(hw))


@pytest.mark.parametrize("name", ["e2m1", "e3m2", "e2m3"])
def test_hw_model_exhaustive_k1(name):
    """Every element pair for the small formats, across scale and C corners."""
    f = FORMATS[name]
    cs = [0, 0x80000000, 0x00000001, 0x80400000, 0x3F800000, 0xBF800001, 0x7F7FFFFF]
    for a, b, xa, xb, c in itertools.product(range(1 << f.width), range(1 << f.width),
                                             [0, 1, 127, 253, 254], [0, 127, 254], cs):
        assert mx_dot_hw(f, f, [a], [b], xa, xb, c) == mx_dot(f, f, [a], [b], xa, xb, c)
