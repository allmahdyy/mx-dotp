"""Golden reference for the MX block dot product with FP32 accumulate.

    R = round_fp32( C + X_A * X_B * sum_i A_i * B_i )

The sum and the add are exact (rational arithmetic) and there is exactly one
rounding, to binary32 with round-to-nearest-even.  Special values follow
IEEE 754 applied to that exact expression:

* NaN if either shared scale is 0xFF, any element or C is NaN, any product is
  Inf * 0, or infinities of opposite sign meet.  NaN output is 0x7FC00000.
* Inf if any product or C is Inf (and no NaN condition holds).
* An exact zero is -0 only if every product is -0 and C is -0, else +0.
"""
from fractions import Fraction

from mx_formats import (
    FP32_INF, FP32_QNAN, INF, NAN, SCALE_BIAS, SCALE_NAN, ElemFormat,
    decode_elem, decode_fp32, round_to_fp32,
)


def mx_dot(fa: ElemFormat, fb: ElemFormat, a, b, xa: int, xb: int, c: int) -> int:
    assert len(a) == len(b)
    if xa == SCALE_NAN or xb == SCALE_NAN:
        return FP32_QNAN
    inf_signs = set()
    nan = False
    total = Fraction(0)
    all_neg_zero = True
    for ai, bi in zip(a, b):
        ka, sa, va = decode_elem(fa, ai)
        kb, sb, vb = decode_elem(fb, bi)
        s = sa ^ sb
        if NAN in (ka, kb):
            nan = True
        elif INF in (ka, kb):
            if (ka == INF or va != 0) and (kb == INF or vb != 0):
                inf_signs.add(s)
            else:
                nan = True                      # Inf * 0
        else:
            p = va * vb
            total += -p if s else p
            all_neg_zero &= (p == 0 and s == 1)
    kc, sc, vc = decode_fp32(c)
    if kc == NAN:
        nan = True
    elif kc == INF:
        inf_signs.add(sc)
    if nan or len(inf_signs) == 2:
        return FP32_QNAN
    if inf_signs:
        return (inf_signs.pop() << 31) | FP32_INF

    s_prod = total * Fraction(2) ** (xa + xb - 2 * SCALE_BIAS)
    result = s_prod + (-vc if sc else vc)
    if result == 0:
        neg = s_prod == 0 and all_neg_zero and vc == 0 and sc == 1
        return 0x80000000 if neg else 0
    return round_to_fp32(result)
