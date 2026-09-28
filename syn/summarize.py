#!/usr/bin/env python3
"""Merge the synthesis sweeps into one exact-vs-truncated area table.

    python syn/summarize.py            # writes results/area_summary.md

Reads results/ppa_<platform>_synth[_truncW]_syn.csv for asap7 and sky130hd
and reports, per configuration, the exact area and the ratio
exact / truncated for every truncation width found.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PLATFORMS = ["asap7", "sky130hd"]
WIDTHS = [24, 32]


def load(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["config"].removesuffix("_syn"): float(r["area"])
                for r in csv.DictReader(f) if r["area"] not in ("", "None")}


def main():
    lines = ["# Area: exact vs anchored truncation (synthesis)", "",
             "Ratio < 1 means the exact unit is smaller.  Areas in um^2.", ""]
    for plat in PLATFORMS:
        exact = load(RES / f"ppa_{plat}_synth_syn.csv")
        trunc = {w: {k.removeprefix(f"trunc{w}_"): v
                     for k, v in load(RES / f"ppa_{plat}_synth_trunc{w}_syn.csv").items()}
                 for w in WIDTHS}
        if not exact:
            continue
        lines += [f"## {plat}", "",
                  "| config | exact | " + " | ".join(f"trunc{w}" for w in WIDTHS)
                  + " | " + " | ".join(f"exact/trunc{w}" for w in WIDTHS) + " |",
                  "|---|---:|" + "---:|" * (2 * len(WIDTHS))]
        for cfg, a in exact.items():
            t = [trunc[w].get(cfg) for w in WIDTHS]
            cells = [f"{a:,.0f}"] + [f"{x:,.0f}" if x else "-" for x in t] \
                + [f"**{a / x:.2f}**" if x else "-" for x in t]
            lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        lines.append("")
    out = RES / "area_summary.md"
    out.write_text("\n".join(lines))
    print(out.read_text())


if __name__ == "__main__":
    main()
