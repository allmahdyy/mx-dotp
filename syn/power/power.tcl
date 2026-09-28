# Vector-based power of a design after clock-tree synthesis or routing.  Run
# through ORFS so the platform, libraries and results directory come from the
# design config:
#   make -C $ORFS/flow DESIGN_CONFIG=... FLOW_VARIANT=... run \
#        RUN_SCRIPT=syn/power/power.tcl VCD_FILE=... VCD_SCOPE=tb/dut PWR_STAGE=route|cts
# route: routed design with extracted parasitics (SPEF).
# cts:   clock tree and hold buffers present, wires estimated from placement.
source $::env(SCRIPTS_DIR)/load.tcl
if { $::env(PWR_STAGE) == "route" } {
  load_design 6_final.odb 6_final.sdc
  read_spef $::env(RESULTS_DIR)/6_final.spef
} else {
  load_design 4_cts.odb 4_cts.sdc
  estimate_parasitics -placement
}
read_vcd -scope $::env(VCD_SCOPE) $::env(VCD_FILE)
report_activity_annotation
report_power
