// Optional pipeline register: a plain wire when EN = 0.
module mx_pipe #(
    parameter W  = 1,
    parameter EN = 1
) (
    input  wire         clk,
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (EN) begin : g_reg
            reg [W-1:0] r;
            always @(posedge clk) r <= d;
            assign q = r;
        end else begin : g_wire
            assign q = d;
        end
    endgenerate
endmodule
