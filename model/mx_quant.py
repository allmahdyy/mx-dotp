"""Quantise real-valued blocks to MX (OCP MX v1.0, section 6.3).

Shared scale X = floor(log2(max|v|)) - emax_elem, then each v / 2^X is
rounded to the nearest element value (ties to even) and saturated to the
largest finite magnitude.  All-zero blocks get X = 0 (scale 2^-127).
"""
import bisect
import math

from mx_formats import NUM, decode_elem

EMAX_ELEM = {"e4m3": 8, "e5m2": 15, "e3m2": 4, "e2m3": 2, "e2m1": 2, "int8": 0}


class Quantiser:
    def __init__(self, fmt):
        self.fmt = fmt
        pos = {}
        for code in range(1 << (fmt.width - 1)):          # sign bit clear
            kind, _, mag = decode_elem(fmt, code)
            if kind == NUM:
                pos.setdefault(float(mag), code)
        if fmt.is_int:                                   # 0x80 = -2.0 has no positive twin
            pos.pop(2.0, None)
        self.vals = sorted(pos)
        self.codes = [pos[v] for v in self.vals]
        self.sign = 1 << (fmt.width - 1)

    def _elem(self, x: float) -> int:
        m = abs(x)
        i = bisect.bisect_left(self.vals, m)
        if i >= len(self.vals):
            code = self.codes[-1]                        # saturate
        elif i == 0 or self.vals[i] == m:
            code = self.codes[i]
        else:
            lo, hi = self.vals[i - 1], self.vals[i]
            if m - lo < hi - m:
                code = self.codes[i - 1]
            elif m - lo > hi - m:
                code = self.codes[i]
            else:                                        # tie: even code (LSB of mantissa 0)
                code = self.codes[i - 1] if self.codes[i - 1] % 2 == 0 else self.codes[i]
        if x < 0 and code:
            code |= self.sign
            if self.fmt.is_int:                          # two's complement, not sign-magnitude
                code = (256 - (code & 0x7F)) & 0xFF
        return code

    def block(self, vals):
        """-> (element codes, E8M0 scale code)"""
        amax = max(abs(v) for v in vals)
        if amax == 0:
            return [0] * len(vals), 0
        x = math.floor(math.log2(amax)) - EMAX_ELEM[self.fmt.name]
        x = max(-127, min(127, x))
        s = 2.0 ** -x
        return [self._elem(v * s) for v in vals], x + 127
