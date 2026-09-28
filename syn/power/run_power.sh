#!/usr/bin/env bash
# Gate-level, vector-based power for one configuration.
#
#   bash syn/power/run_power.sh PLATFORM FMT K PIPE POLICY TW [SUFFIX]
#   e.g. bash syn/power/run_power.sh asap7 e4m3 32 0 exact 0
#   PWR_STAGE=cts bash syn/power/run_power.sh ...   # after CTS, estimated wires
#
# Needs a finished route (or cts) run from syn/sweep.py for the same
# configuration (same run name), and ORFS_HOME / OPENROAD_EXE / YOSYS_EXE set.
# Prints one line:
#   POWER <run> stage=S vectors=N mismatches=M power_W=P energy_pJ_per_op=E
set -euo pipefail
PLATFORM=$1 FMT=$2 K=$3 PIPE=$4 POLICY=$5 TW=$6 SUFFIX=${7:-}
PERIOD_NS=${PERIOD_NS:-1.0}           # one vector per clock; energy/op = P * period
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
FLOW=$ORFS_HOME/flow

declare -A EW=([e4m3]=8 [e5m2]=8 [e3m2]=6 [e2m3]=6 [e2m1]=4 [int8]=8)
PB=$(printf '%04d' "$(echo "obase=2; $PIPE" | bc)")
LAT=$(echo -n "${PB//0/}" | wc -c)
if [ "$POLICY" = exact ]; then TOP=mx_dotp; PRE=""; else TOP=mx_dotp_trunc; PRE="trunc${TW}_"; fi
RUN="${PRE}${FMT}x${FMT}_k${K}_p${PB}${SUFFIX}"
RES=$FLOW/results/$PLATFORM/mx_dotp/$RUN
PWR_STAGE=${PWR_STAGE:-route}
if [ "$PWR_STAGE" = route ]; then
  NETLIST=$RES/6_final.v
elif [ "$PWR_STAGE" = cts ]; then
  # ORFS writes no Verilog after CTS: dump one from the odb
  NETLIST=$RES/4_cts.v
  [ "$NETLIST" -nt "$RES/4_cts.odb" ] || make -C "$FLOW" \
      DESIGN_CONFIG="$REPO/syn/runs/$PLATFORM/$RUN/config.mk" FLOW_VARIANT="$RUN" \
      run RUN_SCRIPT="$REPO/syn/power/write_cts_v.tcl" RUN_LOG_NAME_STEM=7_write_v \
      > /dev/null 2>&1
else
  echo "PWR_STAGE must be route or cts" >&2; exit 1
fi
[ -f "$NETLIST" ] || { echo "no $PWR_STAGE netlist for $RUN" >&2; exit 1; }

VEC=$HOME/power_runs/$PLATFORM/$PWR_STAGE/$RUN
rm -rf "$VEC"; mkdir -p "$VEC"
python3 "$REPO/syn/power/gen_vectors.py" --fmt "$FMT" --k "$K" --policy "$POLICY" \
    --tw "$TW" --out "$VEC" >/dev/null
N=$(cat "$VEC/count.txt")

case $PLATFORM in
  asap7) CELLS=$(ls $FLOW/platforms/asap7/verilog/stdcell/asap7sc7p5t_*.v $FLOW/platforms/asap7/verilog/stdcell/empty.v) ;;
  sky130hd)
    # ORFS ships no sky130 cell simulation models: derive functional ones from
    # the Liberty file (combinational cells only, enough for PIPE=0 designs)
    CELLS=$HOME/power_runs/sky130hd_cells.v
    [ -f "$CELLS" ] || python3 "$REPO/syn/power/lib2v.py" \
        "$(ls $FLOW/platforms/sky130hd/lib/sky130_fd_sc_hd__tt*.lib* | head -1)" \
        sky130_fd_sc_hd__tapvpwrvgnd_1 > "$CELLS" ;;
esac

iverilog -g2012 -o "$VEC/sim" \
    -DTOP=$TOP -DAW=$((K * EW[$FMT])) -DBW=$((K * EW[$FMT])) -DLAT=$LAT -DN=$N \
    -DPERIOD=$PERIOD_NS -DVEC=\"$VEC\" -DFUNCTIONAL -DUNIT_DELAY= \
    "$REPO/syn/power/tb_gl.v" "$NETLIST" $CELLS 2> "$VEC/iverilog.log" \
    || { tail -20 "$VEC/iverilog.log" >&2; exit 1; }
CHECK=$(vvp -n "$VEC/sim" | tee "$VEC/sim.log" | grep GL-CHECK)
MISM=$(echo "$CHECK" | sed -E 's/.*mismatches=([0-9]+).*/\1/')

make -C "$FLOW" DESIGN_CONFIG="$REPO/syn/runs/$PLATFORM/$RUN/config.mk" FLOW_VARIANT="$RUN" \
    run RUN_SCRIPT="$REPO/syn/power/power.tcl" RUN_LOG_NAME_STEM=7_power \
    VCD_FILE="$VEC/activity.vcd" VCD_SCOPE=tb/dut PWR_STAGE="$PWR_STAGE" > "$VEC/power.log" 2>&1 \
    || { tail -30 "$VEC/power.log" >&2; exit 1; }
P=$(awk '/^Total/ {print $5; exit}' "$VEC/power.log")
E=$(python3 -c "print(f'{$P * $PERIOD_NS * 1e3:.4g}')")
echo "POWER $RUN stage=$PWR_STAGE vectors=$N mismatches=$MISM power_W=$P energy_pJ_per_op=$E"
