// Gate-level testbench for power: drives the routed netlist with the vectors
// from gen_vectors.py, one per clock, checks every result against the model
// and dumps a VCD of the measurement window for OpenROAD's read_vcd.
//
// Defines (set by run_power.sh):
//   TOP   netlist module (mx_dotp / mx_dotp_trunc)
//   AW BW element-vector widths (K*EWA, K*EWB)
//   LAT   pipeline latency in cycles (popcount of PIPE)
//   N     number of vectors
//   PERIOD clock period in ns
//   VEC   directory with the .hex files, VCD output path
`timescale 1ns/1ps
module tb;
    reg clk = 0, rst_n = 0, in_valid = 0;
    reg  [`AW-1:0] a;
    reg  [`BW-1:0] b;
    reg  [7:0]  xa, xb;
    reg  [31:0] c;
    wire [31:0] r;
    wire out_valid;

    `TOP dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .a(a), .b(b),
              .xa(xa), .xb(xb), .c(c), .out_valid(out_valid), .r(r));

    reg [`AW-1:0] va  [0:`N-1];
    reg [`BW-1:0] vb  [0:`N-1];
    reg [7:0]     vxa [0:`N-1];
    reg [7:0]     vxb [0:`N-1];
    reg [31:0]    vc  [0:`N-1];
    reg [31:0]    vr  [0:`N-1];

    always #(`PERIOD / 2.0) clk = ~clk;

    integer i, nout, errors;
    initial begin
        $readmemh({`VEC, "/a.hex"},  va);
        $readmemh({`VEC, "/b.hex"},  vb);
        $readmemh({`VEC, "/xa.hex"}, vxa);
        $readmemh({`VEC, "/xb.hex"}, vxb);
        $readmemh({`VEC, "/c.hex"},  vc);
        $readmemh({`VEC, "/r.hex"},  vr);
        a = 0; b = 0; xa = 0; xb = 0; c = 0;
        nout = 0; errors = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);
        $dumpfile({`VEC, "/activity.vcd"});
        $dumpvars(0, dut);
        for (i = 0; i < `N; i = i + 1) begin
            a = va[i]; b = vb[i]; xa = vxa[i]; xb = vxb[i]; c = vc[i];
            in_valid = 1;
            @(negedge clk);
        end
        in_valid = 0;
        repeat (`LAT + 1) @(negedge clk);
        $dumpoff;
        $display("GL-CHECK vectors=%0d outputs=%0d mismatches=%0d", `N, nout, errors);
        $finish;
    end

    // outputs appear LAT cycles after their inputs; sample just before the
    // next input change (a combinational design settles within the cycle)
    always @(posedge clk) begin
        #(`PERIOD / 4.0);
        if (rst_n && out_valid) begin
            if (r !== vr[nout]) begin
                errors = errors + 1;
                if (errors <= 5) $display("mismatch %0d: got %h want %h", nout, r, vr[nout]);
            end
            nout = nout + 1;
        end
    end
endmodule
