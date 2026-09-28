"""Fast, bit-accurate models of MX dot-product accumulation policies.

    r = rnd_fp32( policy_sum( c, 2^(xa+xb-254) * a_i * b_i ) )

Policies
--------
exact
    Every term is kept exactly; one RNE rounding at the end.  Same result as
    mx_ref.mx_dot (checked in test_policies.py) but uses plain integers, so it
    is fast enough for error studies.

trunc(W)
    Anchored truncation, as in max-exponent-aligned many-term adders (NVIDIA
    tensor cores per Fasi et al., Ten-Four, Kaul et al.).  Each non-zero term
    has a *nominal* top exponent, the weight of the highest bit its
    significand can occupy: products use ea+eb with PSIGW = SIGW_A + SIGW_B
    bits (two integer bits for FP formats), c uses its 24-bit significand.
    Subnormals are not normalised, as in hardware.  The anchor E is the
    largest nominal top; every term's magnitude is truncated towards zero to
    the grid 2^(E-W+1) (W bits below and including the anchor), the terms
    are summed exactly and the sum is rounded once (RNE).

Special values (NaN, Inf, signed zero) follow the same rules as mx_ref for
every policy; policies only differ on finite inputs.
"""
from mx_formats import FP32_INF, FP32_QNAN, SCALE_NAN, ElemFormat
from mx_hw_model import dec_hw, fmt_params


def rne_fp32(n: int, lsb: int) -> int:
    """binary32 bits of the exact value n * 2^lsb (n != 0), round to nearest even."""
    sign = 1 if n < 0 else 0
    a = -n if sign else n
    top = lsb + a.bit_length() - 1
    q = max(top, -126) - 23                    # exponent of the result ulp
    sh = q - lsb
    if sh <= 0:
        f = a << -sh
    else:
        f = a >> sh
        rem = a & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and f & 1):
            f += 1
    if f < (1 << 23):
        return (sign << 31) | f
    if f == (1 << 24):
        f >>= 1
        q += 1
    field = q + 150
    if field >= 0xFF:
        return (sign << 31) | FP32_INF
    return (sign << 31) | (field << 23) | (f & 0x7FFFFF)


def terms(fa: ElemFormat, fb: ElemFormat, a, b, xa: int, xb: int, c: int):
    """Decode a block.

    Returns ("special", bits) or
    ("terms", [(signed_int, lsb_exp, nominal_top_exp), ...], zero_sign)
    where zero_sign is the sign an exact zero result must carry.
    """
    sigw_a, emin_a, _ = fmt_params(fa)
    sigw_b, emin_b, _ = fmt_params(fb)
    psigw = sigw_a + sigw_b
    scale = xa + xb - 254
    out = []
    nan = pos_inf = neg_inf = False
    all_neg_zero = True
    for ai, bi in zip(a, b):
        sa, ma, ea, za, ia, na = dec_hw(fa, ai)
        sb, mb, eb, zb, ib, nb = dec_hw(fb, bi)
        s = sa ^ sb
        nan |= na or nb or (ia and zb) or (za and ib)
        if ia or ib:
            pos_inf |= not s
            neg_inf |= bool(s)
        all_neg_zero &= (za or zb) and s == 1
        m = ma * mb
        if m:
            lsb = emin_a + emin_b + ea + eb + scale
            out.append((-m if s else m, lsb, lsb + psigw - 1))
    sc, ec, fc = c >> 31, (c >> 23) & 0xFF, c & 0x7FFFFF
    if ec == 0xFF:
        if fc:
            nan = True
        elif sc:
            neg_inf = True
        else:
            pos_inf = True
    if xa == SCALE_NAN or xb == SCALE_NAN or nan or (pos_inf and neg_inf):
        return ("special", FP32_QNAN)
    if pos_inf or neg_inf:
        return ("special", (int(neg_inf) << 31) | FP32_INF)
    mc = ((1 << 23) | fc) if ec else fc
    if mc:
        lsb = max(ec, 1) - 150
        out.append((-mc if sc else mc, lsb, lsb + 23))
    zero_sign = 1 if (all_neg_zero and sc and not out) else 0
    return ("terms", out, zero_sign)


def _finish(total: int, lsb: int, zero_sign: int) -> int:
    if total == 0:
        return zero_sign << 31
    return rne_fp32(total, lsb)


def dot_exact(fa, fb, a, b, xa, xb, c) -> int:
    t = terms(fa, fb, a, b, xa, xb, c)
    if t[0] == "special":
        return t[1]
    _, ts, zs = t
    if not ts:
        return zs << 31
    lsb = min(e for _, e, _ in ts)
    return _finish(sum(m << (e - lsb) for m, e, _ in ts), lsb, zs)


def dot_trunc(fa, fb, a, b, xa, xb, c, w: int) -> int:
    t = terms(fa, fb, a, b, xa, xb, c)
    if t[0] == "special":
        return t[1]
    _, ts, zs = t
    if not ts:
        return zs << 31
    anchor = max(top for _, _, top in ts)        # largest nominal top exponent
    lsb = anchor - w + 1
    total = 0
    for m, e, _ in ts:
        mag = abs(m)
        mag = mag << (e - lsb) if e >= lsb else mag >> (lsb - e)  # truncate towards zero
        total += -mag if m < 0 else mag
    return _finish(total, lsb, zs)


POLICIES = {
    "exact": dot_exact,
    **{f"trunc{w}": (lambda w: lambda *args: dot_trunc(*args, w))(w)
       for w in (16, 20, 24, 25, 26, 28, 32, 40, 48)},
}
