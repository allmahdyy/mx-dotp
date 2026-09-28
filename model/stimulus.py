"""Directed-random stimulus for the MX dot-product unit.

Shared by the Python model tests and the cocotb testbench.  Plain uniform
bits would make most vectors NaN (for K=32 E4M3) and would almost never hit
cancellation, subnormal outputs or overflow, so each vector picks a "flavour"
that pushes towards one of those corners.
"""
import random

from mx_formats import ElemFormat, decode_elem, NUM
from mx_ref import mx_dot


def _finite_codes(f: ElemFormat):
    return [x for x in range(1 << f.width) if decode_elem(f, x)[0] == NUM]


def _special_codes(f: ElemFormat):
    return [x for x in range(1 << f.width) if decode_elem(f, x)[0] != NUM]


def _extreme_codes(f: ElemFormat):
    """Zeros, smallest subnormals and largest finite values of both signs."""
    fin = sorted(_finite_codes(f), key=lambda x: decode_elem(f, x)[2])
    picks = fin[:3] + fin[-3:]
    return picks + [x ^ (1 << (f.width - 1)) for x in picks]


class Stimulus:
    def __init__(self, fa: ElemFormat, fb: ElemFormat, k: int, seed: int = 1):
        self.fa, self.fb, self.k = fa, fb, k
        self.rng = random.Random(seed)
        self.fin = (_finite_codes(fa), _finite_codes(fb))
        self.spc = (_special_codes(fa), _special_codes(fb))
        self.ext = (_extreme_codes(fa), _extreme_codes(fb))

    def _elems(self, side: int, flavour: str):
        r, f = self.rng, (self.fa, self.fb)[side]
        out = []
        for _ in range(self.k):
            u = r.random()
            if flavour == "special" and self.spc[side] and u < 0.08:
                out.append(r.choice(self.spc[side]))
            elif flavour in ("extreme", "special") and u < 0.35:
                out.append(r.choice(self.ext[side]))
            elif flavour == "sparse" and u < 0.8:
                out.append(0)
            else:
                out.append(r.choice(self.fin[side]))
        return out

    def _scale(self):
        r = self.rng
        u = r.random()
        if u < 0.01:
            return 0xFF
        if u < 0.25:
            return r.choice([0, 1, 2, 126, 127, 128, 252, 253, 254])
        if u < 0.6:
            return r.randint(100, 154)
        return r.randint(0, 254)

    def _fp32(self):
        r = self.rng
        u = r.random()
        if u < 0.25:
            return 0 if r.random() < 0.5 else 0x80000000
        if u < 0.3:
            return r.choice([0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFFFFFFF])
        if u < 0.4:
            return (r.getrandbits(1) << 31) | r.getrandbits(23)          # subnormal
        if u < 0.7:
            e = r.randint(97, 157)                                       # near unit scale
            return (r.getrandbits(1) << 31) | (e << 23) | r.getrandbits(23)
        return r.getrandbits(32)

    def vector(self):
        """-> (a, b, xa, xb, c)"""
        r = self.rng
        flavour = r.choice(["plain", "plain", "extreme", "special", "sparse", "cancel", "cancel_c"])
        a = self._elems(0, flavour)
        b = self._elems(1, flavour)
        xa, xb = self._scale(), self._scale()
        c = self._fp32()
        if flavour == "cancel" and self.k >= 2:
            # Make the products cancel pairwise, then perturb one element.
            half = self.k // 2
            for i in range(half):
                a[half + i] = a[i]
                b[half + i] = b[i] ^ (1 << (self.fb.width - 1))
            if not self.fb.is_int and r.random() < 0.5:
                b[-1] = r.choice(self.fin[1])
        if flavour == "cancel_c":
            # Choose C close to -dot so that the final add cancels heavily.
            xa, xb = r.randint(110, 144), r.randint(110, 144)
            d = mx_dot(self.fa, self.fb, a, b, xa, xb, 0)
            if (d >> 23) & 0xFF != 0xFF:
                c = (d ^ 0x80000000) + r.choice([-2, -1, 0, 0, 1, 2])
                c &= 0xFFFFFFFF
        return a, b, xa, xb, c
