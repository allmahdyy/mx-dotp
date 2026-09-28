"""Bit-true model of the RTL datapath in rtl/mx_dotp.v.

Mirrors the hardware step by step (fixed-point product alignment, integer
adder tree, a finite alignment window with a sticky bit, LZC normalisation,
subnormal denormalisation and RNE rounding) using only bounded-width integer
operations.  Checking it against mx_ref.mx_dot validates the algorithm and
its bit widths independently of any simulator.
"""
from mx_formats import FP32_INF, FP32_QNAN, SCALE_NAN, ElemFormat


def fmt_params(f: ElemFormat):
    """(SIGW, EMIN, EMAX): significand width and LSB exponents of a decoded element."""
    if f.is_int:
        return 8, -6, -6
    emin = 1 - f.bias - f.mbits
    max_field = (1 << f.ebits) - 2 if f.name == "e5m2" else (1 << f.ebits) - 1
    return f.mbits + 1, emin, max_field - f.bias - f.mbits


def clog2(n: int) -> int:
    return (n - 1).bit_length()


class HwParams:
    def __init__(self, fa: ElemFormat, fb: ElemFormat, k: int):
        sa, emin_a, emax_a = fmt_params(fa)
        sb, emin_b, emax_b = fmt_params(fb)
        self.PSIGW = sa + sb
        self.EPMIN = emin_a + emin_b
        self.FXW = self.PSIGW + (emax_a + emax_b) - self.EPMIN
        self.LOGK = clog2(k)
        self.MW = self.FXW + self.LOGK          # |sum| magnitude width
        self.SW = self.MW + 1                   # signed sum width
        self.DW = max(self.MW, 24)
        self.NW = self.DW + 4                   # carry | DW data | G R S


def dec_hw(f: ElemFormat, x: int):
    """-> (sign, sig, eoff, zero, inf, nan) exactly as rtl/mx_dec.v produces."""
    sign = x >> (f.width - 1)
    if f.is_int:
        mag = (-x) & 0xFF if sign else x
        return sign, mag, 0, x == 0, False, False
    e = (x >> f.mbits) & ((1 << f.ebits) - 1)
    m = x & ((1 << f.mbits) - 1)
    allones = e == (1 << f.ebits) - 1
    nan = inf = False
    if f.name == "e4m3":
        nan = allones and m == 7
    elif f.name == "e5m2":
        inf, nan = allones and m == 0, allones and m != 0
    sig = ((1 << f.mbits) | m) if e else m
    zero = sig == 0
    if inf or nan:                   # keep specials off the fixed-point grid
        sig, e = 0, 0
    return sign, sig, max(e - 1, 0), zero, inf, nan


def sticky_shr(x: int, width: int, d: int):
    """Logical right shift of a width-bit value; returns (shifted, any_bit_lost)."""
    d = min(d, width)
    return x >> d, (x & ((1 << d) - 1)) != 0


def lzc(x: int, width: int) -> int:
    return width - x.bit_length()


def mx_dot_hw(fa: ElemFormat, fb: ElemFormat, a, b, xa: int, xb: int, c: int) -> int:
    p = HwParams(fa, fb, len(a))

    # ---- stage 1: decode, multiply, place products on a common fixed-point grid
    s_sum = 0
    any_nan = pos_inf = neg_inf = False
    all_neg_zero = True
    for ai, bi in zip(a, b):
        sa, ma, ea, za, ia, na = dec_hw(fa, ai)
        sb, mb, eb, zb, ib, nb = dec_hw(fb, bi)
        ps = sa ^ sb
        any_nan |= na or nb or (ia and zb) or (za and ib)
        pinf = ia or ib
        pos_inf |= pinf and not ps
        neg_inf |= pinf and ps
        all_neg_zero &= (za or zb) and ps == 1
        term = (ma * mb) << (ea + eb)            # < 2^FXW
        assert term < (1 << p.FXW)
        # ---- stage 2: signed adder tree (sum is exact in SW bits)
        s_sum += -term if ps else term
    assert -(1 << p.MW) < s_sum < (1 << p.MW)

    sc, ec, fc = c >> 31, (c >> 23) & 0xFF, c & 0x7FFFFF
    c_nan, c_inf = ec == 0xFF and fc != 0, ec == 0xFF and fc == 0
    pos_inf |= c_inf and not sc
    neg_inf |= c_inf and sc
    if xa == SCALE_NAN or xb == SCALE_NAN or any_nan or c_nan or (pos_inf and neg_inf):
        return FP32_QNAN
    if pos_inf or neg_inf:
        return (int(neg_inf) << 31) | FP32_INF

    # ---- stage 3: exact add of S * 2^ES and C, inside a finite window
    sS = 1 if s_sum < 0 else 0
    mS = abs(s_sum)
    ES = p.EPMIN + xa + xb - 254                 # exponent of mS's LSB
    mC = ((1 << 23) | fc) if ec else fc
    EC = max(ec, 1) - 150
    zS, zC = mS == 0, mC == 0
    if zS and zC:
        return 0x80000000 if (all_neg_zero and sc) else 0

    topS = ES + p.MW - 1 - lzc(mS, p.MW)         # exponent of the MSB
    topC = EC + 23 - lzc(mC, 24)
    alS = (mS << lzc(mS, p.MW)) << (p.DW - p.MW)  # left-aligned in DW bits
    alC = (mC << lzc(mC, 24)) << (p.DW - 24)
    if zS or (not zC and topC > topS):
        s_big, al_big, top_big, s_sml, al_sml, top_sml, sml_zero = sc, alC, topC, sS, alS, topS, zS
    else:
        s_big, al_big, top_big, s_sml, al_sml, top_sml, sml_zero = sS, alS, topS, sc, alC, topC, zC

    ww = p.DW + 3                                # window below the carry bit
    big_w = al_big << 3
    d = 0 if sml_zero else top_big - top_sml
    assert d >= 0
    sml_w, lost = sticky_shr(al_sml << 3, ww, d)
    sml_w |= int(lost)                           # sticky folded into the LSB
    eff_sub = s_big ^ s_sml
    raw = (big_w - sml_w) if eff_sub else (big_w + sml_w)
    raw &= (1 << p.NW) - 1
    neg = eff_sub and (raw >> (p.NW - 1)) & 1
    mag = (-raw) & ((1 << p.NW) - 1) if neg else raw
    r_sign = s_big ^ int(bool(neg))
    if mag == 0:
        return 0                                 # exact cancellation: +0

    # ---- stage 4: normalise, denormalise if tiny, round to nearest even
    lz = lzc(mag, p.NW)
    er = top_big + 1 - lz                        # exponent of the result MSB
    norm = mag << lz
    rs = min(max(-126 - er, 0), p.NW)
    den, lost2 = sticky_shr(norm, p.NW, rs)
    kept = den >> (p.NW - 24)
    g = (den >> (p.NW - 25)) & 1
    st = (den & ((1 << (p.NW - 25)) - 1)) != 0 or lost2
    m24 = kept + (g & (int(st) | (kept & 1)))
    biased = max(er, -126) + 126
    tot = (biased << 23) + m24
    if tot >= FP32_INF:
        tot = FP32_INF
    return (r_sign << 31) | tot
