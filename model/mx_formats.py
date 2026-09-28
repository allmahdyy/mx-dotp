"""OCP Microscaling (MX) v1.0 element and scale formats.

Every element decodes to an exact rational (``fractions.Fraction``) plus a
sign bit (kept separately so that -0 survives) or to an Inf/NaN marker.
Nothing here uses host floating point, so the model is bit-accurate by
construction.
"""
from dataclasses import dataclass
from fractions import Fraction

NUM, INF, NAN = "num", "inf", "nan"


@dataclass(frozen=True)
class ElemFormat:
    name: str
    code: int     # value of the FMT_A / FMT_B RTL parameter
    width: int    # bits per element
    ebits: int    # exponent bits (0 for MXINT8)
    mbits: int    # stored mantissa bits (MXINT8: magnitude bits)
    bias: int

    @property
    def is_int(self) -> bool:
        return self.ebits == 0


FORMATS = {
    f.name: f
    for f in (
        ElemFormat("e4m3", 0, 8, 4, 3, 7),
        ElemFormat("e5m2", 1, 8, 5, 2, 15),
        ElemFormat("e3m2", 2, 6, 3, 2, 3),
        ElemFormat("e2m3", 3, 6, 2, 3, 1),
        ElemFormat("e2m1", 4, 4, 2, 1, 1),
        ElemFormat("int8", 5, 8, 0, 7, 0),
    )
}
BY_CODE = {f.code: f for f in FORMATS.values()}

SCALE_NAN = 0xFF  # E8M0: 2^(X-127), X=0xFF is NaN
SCALE_BIAS = 127


def decode_elem(fmt: ElemFormat, x: int):
    """Return (kind, sign, magnitude) with magnitude an exact Fraction."""
    assert 0 <= x < (1 << fmt.width)
    sign = x >> (fmt.width - 1)
    if fmt.is_int:
        # MXINT8: two's complement with an implicit scale of 2^-6.
        v = x - 256 if sign else x
        return NUM, sign, Fraction(abs(v), 64)
    e = (x >> fmt.mbits) & ((1 << fmt.ebits) - 1)
    m = x & ((1 << fmt.mbits) - 1)
    emax_field = (1 << fmt.ebits) - 1
    if fmt.name == "e4m3" and e == emax_field and m == (1 << fmt.mbits) - 1:
        return NAN, sign, None
    if fmt.name == "e5m2" and e == emax_field:
        return (INF if m == 0 else NAN), sign, None
    if e == 0:
        mag = Fraction(m) * Fraction(2) ** (1 - fmt.bias - fmt.mbits)
    else:
        mag = Fraction((1 << fmt.mbits) | m) * Fraction(2) ** (e - fmt.bias - fmt.mbits)
    return NUM, sign, mag


def decode_fp32(bits: int):
    sign = bits >> 31
    e = (bits >> 23) & 0xFF
    f = bits & 0x7FFFFF
    if e == 0xFF:
        return (INF if f == 0 else NAN), sign, None
    if e == 0:
        return NUM, sign, Fraction(f) * Fraction(2) ** -149
    return NUM, sign, Fraction(f | (1 << 23)) * Fraction(2) ** (e - 150)


FP32_QNAN = 0x7FC00000
FP32_INF = 0x7F800000


def round_to_fp32(x: Fraction) -> int:
    """Round a non-zero exact value to binary32, round-to-nearest-even.

    Handles subnormal results, and overflow to Inf, exactly as IEEE 754 says.
    """
    assert x != 0
    sign = 1 if x < 0 else 0
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** e:
        e -= 1                      # now 2^e <= a < 2^(e+1)
    q = max(e, -126) - 23           # exponent of the result's ulp
    n = a / Fraction(2) ** q
    f = n.numerator // n.denominator
    rem = n - f
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and f & 1):
        f += 1
    if f < (1 << 23):               # subnormal (only possible when q == -149)
        return (sign << 31) | f
    if f == (1 << 24):              # rounding carried into the next binade
        f >>= 1
        q += 1
    field = q + 150
    if field >= 0xFF:
        return (sign << 31) | FP32_INF
    return (sign << 31) | (field << 23) | (f & 0x7FFFFF)
