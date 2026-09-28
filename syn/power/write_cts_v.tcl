# Write a Verilog netlist of the design after clock-tree synthesis
# (4_cts.odb -> 4_cts.v) for gate-level simulation.
source $::env(SCRIPTS_DIR)/load.tcl
load_design 4_cts.odb 4_cts.sdc
write_verilog $::env(RESULTS_DIR)/4_cts.v
