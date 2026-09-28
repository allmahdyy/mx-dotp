#!/usr/bin/env python3
"""PPA sweep of mx_dotp through OpenROAD-flow-scripts (Yosys + OpenROAD).

    export ORFS_HOME=~/OpenROAD-flow-scripts && source $ORFS_HOME/env.sh
    python3 syn/sweep.py --platform asap7 --set quick -j 4
    python3 syn/sweep.py --platform sky130hd --set full -j 4
    python3 syn/sweep.py --platform asap7 --set full --stage synth    # area only, fast

Each configuration gets its own ORFS FLOW_VARIANT.  Results are written to
results/ppa_<platform>_<stage>.csv and results/ppa_<platform>_<stage>.md.

Method (keep this honest in any write-up):
* I/O delays are 0, so reported delay is the worst of in->reg, reg->reg,
  reg->out and in->out paths.  Combinational configs (PIPE=0) therefore
  report the full input-to-output delay.
* "delay" = clock period - worst setup slack, after the chosen stage
  (post-route by default).  --stage place stops after detailed placement
  (estimated wires, ideal clock; delay tracked the routed value within 2% on
  the pipelined FP8 unit).  --stage cts adds the clock tree and hold fixing,
  which power needs: placement-stage power omits the clock network and
  under-estimated the pipelined FP8 unit by ~2x.
* Target period: an unreachable target makes timing repair upsize and
  buffer heavily (+30-70% area seen), differently per design.  Final numbers
  use --period-from <earlier results.csv> so each configuration is repaired
  against 1.1x (--period-margin) the delay it actually achieved.  A positive slack means the design is faster
  than the target.  Timing-driven steps see only the target, so a design
  run at a loose target may be larger/slower than one run near its limit.
* power_vectorless is OpenROAD's report_power with default (vectorless)
  switching activity.  Through deep arithmetic it is off by orders of
  magnitude (it implies ~27 toggles per net per cycle on the smoke design),
  so it is recorded but must not be reported.  Real power needs switching
  activity from gate-level simulation.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODES = {"e4m3": 0, "e5m2": 1, "e3m2": 2, "e2m3": 3, "e2m1": 4, "int8": 5}

# Target clock periods in the platform's time unit (ps for asap7, ns for sky130).
TARGET = {
    "asap7":    {"comb": 1500, "pipe": 500},
    "sky130hd": {"comb": 15.0, "pipe": 5.0},
}

SETS = {
    "smoke": [("e2m1", "e2m1", 8, 0b0000)],
    "pipe8": [("e4m3", "e4m3", 32, 0b1111)],
    "comb6": [(f, f, 32, 0b0000) for f in CODES],
    "comb2": [("e2m1", "e2m1", 32, 0b0000), ("e5m2", "e5m2", 32, 0b0000)],
    "quick": [("e4m3", "e4m3", 32, 0b0000), ("e2m1", "e2m1", 32, 0b0000),
              ("e4m3", "e4m3", 32, 0b1111)],
    "full": (
        [(f, f, 32, p) for f in CODES for p in (0b0000, 0b1111)]
        + [("e4m3", "e4m3", k, 0b1111) for k in (8, 16, 64)]
        + [("e4m3", "e5m2", 32, 0b1111), ("e2m1", "e4m3", 32, 0b1111)]
    ),
}

SDC = """\
set clk_name  core_clock
set clk_port  [get_ports clk]
set clk_period {period}
create_clock -name $clk_name -period $clk_period $clk_port
set_input_delay  0 -clock $clk_name [all_inputs -no_clocks]
set_output_delay 0 -clock $clk_name [all_outputs]
"""

TOP = {"exact": "mx_dotp", "trunc": "mx_dotp_trunc"}

CONFIG_MK = """\
export PLATFORM         = {platform}
export DESIGN_NAME      = {top}
export DESIGN_NICKNAME  = mx_dotp
export VERILOG_FILES    = {verilog}
export VERILOG_TOP_PARAMS = FMT_A {fa} FMT_B {fb} K {k} PIPE {pipe}{extra_params}
export SDC_FILE         = {sdc}
export CORE_UTILIZATION = {util}
export PLACE_DENSITY    = {density}
"""


def tag(fa, fb, k, pipe, suffix="", policy="exact", tw=0):
    pol = "" if policy == "exact" else f"trunc{tw}_"
    return f"{pol}{fa}x{fb}_k{k}_p{pipe:04b}{suffix}"


def find(metrics: dict, *suffixes):
    for s in suffixes:
        for key, v in metrics.items():
            if key.endswith(s) and v not in ("N/A", None):
                return v
    return None


def run_one(args, cfg):
    fa, fb, k, pipe = cfg
    t = tag(*cfg, suffix=f"_{args.tag}" if args.tag else "", policy=args.policy, tw=args.tw)
    rundir = ROOT / "syn" / "runs" / args.platform / t
    rundir.mkdir(parents=True, exist_ok=True)
    base = tag(*cfg, policy=args.policy, tw=args.tw)
    if args.period_from:
        # target = margin x the delay this configuration achieved before, so
        # every design is repaired against an equally reachable target
        period = round(args.period_from[base] * args.period_margin, 3)
    else:
        period = args.period or TARGET[args.platform]["pipe" if pipe else "comb"]
    (rundir / "constraint.sdc").write_text(SDC.format(period=period))
    util, density = (30, 0.60) if args.platform == "asap7" else (35, 0.60)
    (rundir / "config.mk").write_text(CONFIG_MK.format(
        platform=args.platform, top=TOP[args.policy],
        verilog=" ".join(str(p) for p in sorted((ROOT / "rtl").glob("*.v"))),
        fa=CODES[fa], fb=CODES[fb], k=k, pipe=pipe,
        extra_params=f" TW {args.tw}" if args.policy == "trunc" else "",
        sdc=rundir / "constraint.sdc", util=util, density=density)
        + "".join(f"export {kv.split('=', 1)[0]} = {kv.split('=', 1)[1]}\n" for kv in args.set_var))

    flow = Path(args.orfs) / "flow"
    target = {"synth": "synth", "place": "place", "cts": "cts", "route": "finish"}[args.stage]
    cmd = ["make", "-C", str(flow), f"DESIGN_CONFIG={rundir / 'config.mk'}",
           f"FLOW_VARIANT={t}", target]
    with open(rundir / "flow.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    logs = flow / "logs" / args.platform / "mx_dotp" / t
    reports = flow / "reports" / args.platform / "mx_dotp" / t

    row = {"config": t, "fmt_a": fa, "fmt_b": fb, "K": k, "pipe": f"{pipe:04b}",
           "stages": bin(pipe).count("1"), "target": period, "ok": rc == 0}
    metrics = {}
    for js in sorted(logs.glob("*.json")):
        try:
            metrics.update(json.loads(js.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    prefix = {"synth": "synth__", "place": "detailedplace__", "cts": "cts__",
              "route": "finish__"}[args.stage]
    stage_m = {k2: v for k2, v in metrics.items() if k2.startswith(prefix)} or metrics
    row["area"] = find(stage_m, "design__instance__area__stdcell", "design__instance__area")
    row["cells"] = find(stage_m, "design__instance__count__stdcell", "design__instance__count")
    ws = find(stage_m, "timing__setup__ws")
    row["slack"] = ws
    row["delay"] = round(period - ws, 3) if isinstance(ws, (int, float)) else None
    # Vectorless: activity propagation through deep arithmetic grossly
    # overestimates switching.  Kept for reference only; do not report.
    row["power_vectorless"] = find(stage_m, "power__total")
    if args.stage == "synth" and row["area"] is None:
        stat = reports / "synth_stat.txt"
        if stat.exists():
            for line in stat.read_text().splitlines():
                if "Chip area" in line:
                    row["area"] = float(line.split()[-1])
    print(f"[{'ok' if rc == 0 else 'FAIL'}] {t}: area={row['area']} delay={row['delay']} "
          f"cells={row['cells']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--platform", default="asap7", choices=sorted(TARGET))
    ap.add_argument("--set", default="quick", choices=sorted(SETS))
    ap.add_argument("--stage", default="route", choices=["synth", "place", "cts", "route"])
    ap.add_argument("--period", type=float, help="override target clock period")
    ap.add_argument("--period-from", metavar="CSV",
                    help="results CSV of an earlier run: target = margin x its delay per config")
    ap.add_argument("--period-margin", type=float, default=1.1)
    ap.add_argument("--orfs", default=os.environ.get("ORFS_HOME"))
    ap.add_argument("--policy", default="exact", choices=sorted(TOP))
    ap.add_argument("--tw", type=int, default=24, help="window width for --policy trunc")
    ap.add_argument("--set-var", action="append", default=[], metavar="KEY=VALUE",
                    help="extra ORFS variable for config.mk (repeatable)")
    ap.add_argument("--tag", default="", help="suffix for run names and result files")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    args = ap.parse_args()
    if not args.orfs:
        sys.exit("set ORFS_HOME or pass --orfs")
    if args.period_from:
        with open(args.period_from) as f:
            args.period_from = {r["config"]: float(r["delay"]) for r in csv.DictReader(f)
                                if r["delay"] not in ("", "None")}

    with ThreadPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(lambda c: run_one(args, c), SETS[args.set]))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pol = "" if args.policy == "exact" else f"_trunc{args.tw}"
    stem = out / f"ppa_{args.platform}_{args.stage}{pol}{'_' + args.tag if args.tag else ''}"
    cols = list(rows[0].keys())
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(f"{stem}.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"wrote {stem}.csv and {stem}.md")


if __name__ == "__main__":
    main()
