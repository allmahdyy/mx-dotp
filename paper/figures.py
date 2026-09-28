#!/usr/bin/env python3
"""Paper figures from results/.   python paper/figures.py  ->  paper/figs/*.pdf|png

Colours: categorical slots 1-6 of the reference palette, checked with the
dataviz validator (light surface: all checks pass; three slots are below 3:1
contrast, so every series also has its own marker and line style and the
exact numbers are in the paper's tables).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figs"
sys.path.insert(0, str(ROOT / "syn"))
from final_tables import FMTS, NAME, csv_map, energy, cfg  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W = 3.5                                   # IEEE single column, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote paper/figs/{name}.pdf/.png")


def fig_crossover():
    """Exact / truncated (W=24) ratio per format: area on both PDKs and energy."""
    a7 = csv_map("ppa_asap7_synth_syn.csv"); a7t = csv_map("ppa_asap7_synth_trunc24_syn.csv")
    s1 = csv_map("ppa_sky130hd_synth_syn.csv"); s1t = csv_map("ppa_sky130hd_synth_trunc24_syn.csv")
    en = energy("asap7")
    series = [
        ("Area, ASAP7 (synthesis)", lambda c: a7[c] / a7t["trunc24_" + c]),
        ("Area, sky130 (synthesis)", lambda c: s1[c] / s1t["trunc24_" + c]),
        ("Energy/op, ASAP7 (gate-level)", lambda c: en[c] / en["trunc24_" + c]),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ys = range(len(FMTS))
    ax.axvline(1.0, color=INK2, lw=0.8, zorder=1)
    for i, (label, fn) in enumerate(series):
        xs = [fn(cfg(f)) for f in FMTS]
        ax.scatter(xs, [y + (i - 1) * 0.22 for y in ys], s=22, marker=MARK[i], color=C[i],
                   edgecolor="white", linewidth=0.6, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0])
    ax.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"])
    ax.minorticks_off()
    ax.set_xlim(0.42, 2.3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([NAME[f] for f in FMTS])
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("exact / truncated (W = 24), K = 32, single cycle")
    ax.text(0.45, -0.95, "exact cheaper", color=INK2, fontsize=7, ha="left")
    ax.text(2.2, -0.95, "truncated cheaper", color=INK2, fontsize=7, ha="right")
    ax.set_ylim(len(FMTS) - 0.5, -1.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    save(fig, "crossover")


def fig_accuracy():
    """Share of single-block results changed by truncation, vs window width."""
    with open(RES / "error_block.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["dist"] == "gauss"]
    widths = [16, 20, 24, 25, 26, 28, 32]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for i, fmt in enumerate(FMTS):
        ys = []
        for w in widths:
            v = float(next(r for r in rows if r["fmt"] == fmt and r["policy"] == f"trunc{w}")["differs_pct"])
            ys.append(max(v, 0.03))                       # 0 % drawn at the floor
        ax.plot(widths, ys, color=C[i], marker=MARK[i], ms=3.5, ls=LS[i], label=NAME[fmt],
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylim(0.025, 150)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1%", "1%", "10%", "100%"])
    ax.set_xticks(widths)
    ax.set_xlabel("truncation window W (bits)")
    ax.set_ylabel("results changed vs. exact")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.text(32.3, 0.03, "0 %", color=INK2, fontsize=6, va="center")
    ax.legend(ncol=2, frameon=False, loc="lower left", handlelength=2.2, fontsize=6.5)
    save(fig, "accuracy_vs_w")


def fig_blocksize():
    """E4M3 4-stage area ratio vs block size, both PDKs."""
    ks = [8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    for i, (plat, label) in enumerate((("asap7", "ASAP7"), ("sky130hd", "sky130"))):
        ex = csv_map(f"ppa_{plat}_synth_syn.csv")
        tr = csv_map(f"ppa_{plat}_synth_trunc24_syn.csv")
        ys = [ex[cfg("e4m3", "1111", k)] / tr["trunc24_" + cfg("e4m3", "1111", k)] for k in ks]
        ax.plot(ks, ys, color=C[i], marker=MARK[i], ms=4, ls=LS[i], label=label,
                markeredgecolor="white", markeredgewidth=0.4)
        ax.annotate(f"{ys[-1]:.2f}", (ks[-1], ys[-1]), xytext=(4, -2 if i else 3),
                    textcoords="offset points", fontsize=6.5, color=INK2)
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.text(8, 1.005, "break-even", color=INK2, fontsize=6.5, va="bottom")
    ax.set_ylim(0.95, 1.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("block size K (MXFP8 E4M3, 4-stage)")
    ax.set_ylabel("area exact / trunc24")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "blocksize")


if __name__ == "__main__":
    fig_crossover()
    fig_accuracy()
    fig_blocksize()
