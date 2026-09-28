// Decode one MX element into sign, integer significand and exponent offset.
//
//   value = (-1)^sign * sig * 2^(EMIN + eoff)
//
// EMIN is the exponent of the significand LSB for the smallest binade
// (see mx_dotp.v).  Inf/NaN lanes get sig = 0 and eoff = 0 so they stay on
// the fixed-point grid; their flags drive the special-value path instead.
//
// FMT: 0 E4M3, 1 E5M2, 2 E3M2, 3 E2M3, 4 E2M1, 5 INT8 (OCP MX v1.0)
module mx_dec #(
    parameter FMT  = 0,
    parameter EW   = 8,   // element width
    parameter SIGW = 4,   // significand width
    parameter EOW  = 4    // eoff width
) (
    input  wire [EW-1:0]   x,
    output wire            sign,
    output wire [SIGW-1:0] sig,
    output wire [EOW-1:0]  eoff,
    output wire            zero,
    output wire            inf,
    output wire            nan
);
    assign sign = x[EW-1];

    generate
        if (FMT == 5) begin : g_int
            // MXINT8: two's complement, implicit scale 2^-6.  |-128| = 128 fits in 8 bits.
            assign sig  = sign ? (~x + 8'd1) : x;
            assign eoff = {EOW{1'b0}};
            assign zero = ~|x;
            assign inf  = 1'b0;
            assign nan  = 1'b0;
        end else begin : g_fp
            localparam MB = SIGW - 1;
            localparam EB = EW - 1 - MB;
            wire [EB-1:0] e = x[EW-2 -: EB];
            wire [MB-1:0] m = x[MB-1:0];
            wire allones = &e;
            wire normal  = |e;
            wire special;
            if (FMT == 0) begin : g_e4m3       // NaN = S.1111.111, no Inf
                assign nan = allones & (&m);
                assign inf = 1'b0;
            end else if (FMT == 1) begin : g_e5m2
                assign nan = allones & (|m);
                assign inf = allones & ~(|m);
            end else begin : g_nospecial        // FP6 / FP4: no Inf or NaN
                assign nan = 1'b0;
                assign inf = 1'b0;
            end
            assign special = inf | nan;
            assign zero = ~normal & ~(|m);
            assign sig  = special ? {SIGW{1'b0}} : {normal, m};
            assign eoff = (special | ~normal) ? {EOW{1'b0}} : e - 1'b1;
        end
    endgenerate
endmodule
