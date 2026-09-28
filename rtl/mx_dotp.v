// OCP MX block dot product with fused FP32 accumulate.
//
//   r = RNE_fp32( c + 2^(xa-127) * 2^(xb-127) * sum_{i<K} a_i * b_i )
//
// The sum of products and the add of c are exact; there is one rounding.
// Special values follow IEEE 754 on that exact expression (see
// model/mx_ref.py, the golden reference, and model/mx_hw_model.py, a
// bit-true model of this datapath).
//
// Datapath, with optional pipeline cuts P0..P3 selected by PIPE[3:0]:
//   decode + multiply                                  | P0
//   align onto a fixed-point grid, signed binary adder
//   tree (exact, grows 1 bit/level)                    | P1
//   sign-magnitude, LZC, align with c, add/sub         | P2
//   normalise, denormalise, round, specials            | P3 -> r
//
// FMT_A / FMT_B: 0 E4M3, 1 E5M2, 2 E3M2, 3 E2M3, 4 E2M1, 5 INT8.
// K must be a power of two (MX uses K = 32).  EWA / EWB are derived: do not override.
module mx_dotp #(
    parameter FMT_A = 0,
    parameter FMT_B = 0,
    parameter K     = 32,
    parameter PIPE  = 4'b0000,
    parameter EWA   = (FMT_A == 2 || FMT_A == 3) ? 6 : (FMT_A == 4) ? 4 : 8,
    parameter EWB   = (FMT_B == 2 || FMT_B == 3) ? 6 : (FMT_B == 4) ? 4 : 8
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             in_valid,
    input  wire [K*EWA-1:0] a,        // element i at a[i*EWA +: EWA]
    input  wire [K*EWB-1:0] b,
    input  wire [7:0]       xa,       // E8M0 shared scales
    input  wire [7:0]       xb,
    input  wire [31:0]      c,        // FP32 addend
    output wire             out_valid,
    output wire [31:0]      r
);
    // ------------------------------------------------------------------ params
    function integer clog2(input integer n);
        integer v;
        begin
            clog2 = 0;
            for (v = n - 1; v > 0; v = v >> 1) clog2 = clog2 + 1;
        end
    endfunction
    // significand width / LSB exponent of the smallest binade / max eoff
    function integer f_sigw(input integer f);
        f_sigw = (f == 5) ? 8 : (f == 0 || f == 3) ? 4 : (f == 4) ? 2 : 3;
    endfunction
    function integer f_emin(input integer f);
        f_emin = (f == 0) ? -9 : (f == 1) ? -16 : (f == 2) ? -4 :
                 (f == 3) ? -3 : (f == 4) ? -1 : -6;
    endfunction
    function integer f_eoffmax(input integer f);
        f_eoffmax = (f == 0) ? 14 : (f == 1) ? 29 : (f == 2) ? 6 :
                    (f == 3) ? 2  : (f == 4) ? 2  : 0;
    endfunction
    function integer f_eow(input integer f);
        f_eow = (f == 1) ? 5 : (f == 0) ? 4 : (f == 2) ? 3 : (f == 5) ? 1 : 2;
    endfunction
    function integer imax(input integer x, input integer y);
        imax = (x > y) ? x : y;
    endfunction

    localparam SIGWA = f_sigw(FMT_A), SIGWB = f_sigw(FMT_B);
    localparam EOWA  = f_eow(FMT_A),  EOWB  = f_eow(FMT_B);
    localparam PSIGW = SIGWA + SIGWB;
    localparam EPMIN = f_emin(FMT_A) + f_emin(FMT_B);
    localparam EOFFM = f_eoffmax(FMT_A) + f_eoffmax(FMT_B);
    localparam SHW   = clog2(EOFFM + 1) + 1;
    localparam FXW   = PSIGW + EOFFM;       // product magnitude on the grid
    localparam LOGK  = clog2(K);
    localparam MW    = FXW + LOGK;          // |sum| width
    localparam SW    = MW + 1;              // signed sum width
    localparam DW    = imax(MW, 24);         // alignment data width
    localparam WW    = DW + 3;              // + guard, round, sticky
    localparam NW    = WW + 1;              // + carry
    localparam EXW   = 12;                  // signed exponent width
    localparam LZSW  = clog2(MW + 1);
    localparam LZNW  = clog2(NW + 1);

    // adder-tree level L holds K>>L signed values of width FXW+1+L
    function integer lw(input integer l);
        lw = FXW + 1 + l;
    endfunction
    function integer loff(input integer l);
        integer j;
        begin
            loff = 0;
            for (j = 0; j < l; j = j + 1) loff = loff + (K >> j) * lw(j);
        end
    endfunction
    localparam TREEW = loff(LOGK + 1);
    localparam TW0   = FXW + 1;             // signed product width (level 0)

    // flags: {nan, pos_inf, neg_inf, all_products_neg_zero}
    localparam FLW = 4;

    // ============================================== stage A: decode + multiply
    // P0 registers the narrow {product, ea+eb, sign} per lane (same cut as
    // mx_dotp_trunc.v); the shift onto the wide grid happens in stage B.
    wire [K*PSIGW-1:0] pm_a;
    wire [K*SHW-1:0]   sh_a;
    wire [K-1:0]       ps_a;
    wire [K-1:0] l_nan, l_pinf, l_ninf, l_nz;

    genvar i, l, j;
    generate
        for (i = 0; i < K; i = i + 1) begin : g_lane
            wire sa, sb, za, zb, ia, ib, na, nb;
            wire [SIGWA-1:0] ma;
            wire [SIGWB-1:0] mb;
            wire [EOWA-1:0]  ea;
            wire [EOWB-1:0]  eb;
            mx_dec #(.FMT(FMT_A), .EW(EWA), .SIGW(SIGWA), .EOW(EOWA)) u_da (
                .x(a[i*EWA +: EWA]), .sign(sa), .sig(ma), .eoff(ea),
                .zero(za), .inf(ia), .nan(na));
            mx_dec #(.FMT(FMT_B), .EW(EWB), .SIGW(SIGWB), .EOW(EOWB)) u_db (
                .x(b[i*EWB +: EWB]), .sign(sb), .sig(mb), .eoff(eb),
                .zero(zb), .inf(ib), .nan(nb));
            wire ps = sa ^ sb;
            wire [SHW-1:0] sh = ea + eb;
            assign pm_a[i*PSIGW +: PSIGW] = ma * mb;
            assign sh_a[i*SHW +: SHW]     = sh;
            assign ps_a[i]   = ps;
            assign l_nan[i]  = na | nb | (ia & zb) | (za & ib);
            assign l_pinf[i] = (ia | ib) & ~ps;
            assign l_ninf[i] = (ia | ib) & ps;
            assign l_nz[i]   = (za | zb) & ps;
        end
    endgenerate

    wire c_nan = (&c[30:23]) & (|c[22:0]);
    wire c_inf = (&c[30:23]) & ~(|c[22:0]);
    wire [FLW-1:0] fl_a = {
        (&xa) | (&xb) | (|l_nan) | c_nan,
        (|l_pinf) | (c_inf & ~c[31]),
        (|l_ninf) | (c_inf & c[31]),
        &l_nz
    };

    // ---- P0
    localparam P0W = K * PSIGW + K * SHW + K + 16 + 32 + FLW;
    wire [K*PSIGW-1:0] pm_b;
    wire [K*SHW-1:0]   sh_b;
    wire [K-1:0]       ps_b;
    wire [7:0]  xa_b, xb_b;
    wire [31:0] c_b;
    wire [FLW-1:0] fl_b;
    mx_pipe #(.W(P0W), .EN(PIPE[0])) u_p0 (.clk(clk),
        .d({pm_a, sh_a, ps_a, xa, xb, c, fl_a}),
        .q({pm_b, sh_b, ps_b, xa_b, xb_b, c_b, fl_b}));

    // ============================================== stage B: align + adder tree
    wire [TREEW-1:0] tv;
    generate
        for (i = 0; i < K; i = i + 1) begin : g_align
            wire [FXW-1:0] pmx = pm_b[i*PSIGW +: PSIGW];
            wire [FXW-1:0] t   = pmx << sh_b[i*SHW +: SHW];
            wire [FXW:0]   tz  = {1'b0, t};
            assign tv[i*TW0 +: TW0] = ps_b[i] ? (~tz + 1'b1) : tz;
        end
    endgenerate
    generate
        for (l = 0; l < LOGK; l = l + 1) begin : g_lvl
            localparam IW = lw(l), IO = loff(l), OO = loff(l + 1);
            for (j = 0; j < (K >> (l + 1)); j = j + 1) begin : g_add
                wire [IW-1:0] x0 = tv[IO + (2*j)   * IW +: IW];
                wire [IW-1:0] x1 = tv[IO + (2*j+1) * IW +: IW];
                assign tv[OO + j*(IW+1) +: IW+1] = {x0[IW-1], x0} + {x1[IW-1], x1};
            end
        end
    endgenerate
    localparam ROOT = loff(LOGK);
    wire [SW-1:0] sum_b = tv[ROOT +: SW];

    // ---- P1
    wire [SW-1:0] sum_c;
    wire [7:0]  xa_c, xb_c;
    wire [31:0] c_c;
    wire [FLW-1:0] fl_c;
    mx_pipe #(.W(SW + 16 + 32 + FLW), .EN(PIPE[1])) u_p1 (.clk(clk),
        .d({sum_b, xa_b, xb_b, c_b, fl_b}),
        .q({sum_c, xa_c, xb_c, c_c, fl_c}));

    // ============================================== stage C: exact add with c
    wire          sS = sum_c[SW-1];
    wire [MW-1:0] mS = sS ? (~sum_c[MW-1:0] + 1'b1) : sum_c[MW-1:0];
    wire          sC = c_c[31];
    wire [7:0]    ec = c_c[30:23];
    wire [23:0]   mC = {|ec, c_c[22:0]};
    wire zS = ~|mS;
    wire zC = ~|mC;

    wire [LZSW-1:0] lzS;
    wire [4:0]      lzC;
    mx_lzc #(.W(MW), .CW(LZSW)) u_lzs (.in(mS), .cnt(lzS));
    mx_lzc #(.W(24), .CW(5))    u_lzc (.in(mC), .cnt(lzC));

    // exponents of the MSBs
    wire signed [EXW-1:0] topS = $signed({{(EXW-8){1'b0}}, xa_c})
                               + $signed({{(EXW-8){1'b0}}, xb_c})
                               + (EPMIN - 254 + MW - 1)
                               - $signed({{(EXW-LZSW){1'b0}}, lzS});
    wire signed [EXW-1:0] topC = $signed({{(EXW-8){1'b0}}, (|ec) ? ec : 8'd1})
                               + (23 - 150)
                               - $signed({{(EXW-5){1'b0}}, lzC});

    // left-align both in DW bits
    wire [DW-1:0] mSx = mS;
    wire [DW-1:0] mCx = mC;
    wire [DW-1:0] alS = (mSx << lzS) << (DW - MW);
    wire [DW-1:0] alC = (mCx << lzC) << (DW - 24);

    wire swap = zS | (~zC & (topC > topS));        // c is the larger operand
    wire                  s_big   = swap ? sC   : sS;
    wire                  s_sml   = swap ? sS   : sC;
    wire [DW-1:0]         al_big  = swap ? alC  : alS;
    wire [DW-1:0]         al_sml  = swap ? alS  : alC;
    wire signed [EXW-1:0] top_big = swap ? topC : topS;
    wire signed [EXW-1:0] top_sml = swap ? topS : topC;
    wire                  z_sml   = swap ? zS   : zC;

    wire signed [EXW-1:0] dd = top_big - top_sml;
    wire [EXW-1:0] d = z_sml ? {EXW{1'b0}} : (dd > WW) ? WW : dd;

    wire [2*WW-1:0] sml_sh = {al_sml, 3'b000, {WW{1'b0}}} >> d;
    wire [WW-1:0]   sml_w  = {sml_sh[2*WW-1:WW+1], sml_sh[WW] | (|sml_sh[WW-1:0])};
    wire [NW-1:0]   big_n  = {1'b0, al_big, 3'b000};
    wire [NW-1:0]   sml_n  = {1'b0, sml_w};
    wire            eff_sub = s_big ^ s_sml;
    wire [NW-1:0]   raw    = eff_sub ? (big_n - sml_n) : (big_n + sml_n);
    wire            neg    = eff_sub & raw[NW-1];
    wire [NW-1:0]   mag_c  = neg ? (~raw + 1'b1) : raw;
    wire            rsgn_c = s_big ^ neg;
    wire            zz_c   = zS & zC;              // both operands exactly zero
    wire            nzs_c  = fl_c[0] & sC;         // sign of that zero

    // ---- P2
    wire [NW-1:0] mag_d;
    wire signed [EXW-1:0] top_d;
    wire rsgn_d, zz_d, nzs_d;
    wire [2:0] fl_d;
    mx_pipe #(.W(NW + EXW + 3 + 3), .EN(PIPE[2])) u_p2 (.clk(clk),
        .d({mag_c, top_big, rsgn_c, zz_c, nzs_c, fl_c[3:1]}),
        .q({mag_d, top_d,   rsgn_d, zz_d, nzs_d, fl_d}));

    // ============================================== stage D: normalise + round
    wire [LZNW-1:0] lz;
    mx_lzc #(.W(NW), .CW(LZNW)) u_lzn (.in(mag_d), .cnt(lz));
    wire [NW-1:0] norm = mag_d << lz;
    wire signed [EXW-1:0] er = top_d + 1 - $signed({{(EXW-LZNW){1'b0}}, lz});
    wire tiny = er < -126;
    wire signed [EXW-1:0] rs_raw = -126 - er;
    wire [EXW-1:0] rs = ~tiny ? {EXW{1'b0}} : (rs_raw > NW) ? NW : rs_raw;
    wire [2*NW-1:0] den = {norm, {NW{1'b0}}} >> rs;
    wire [23:0] kept = den[2*NW-1 -: 24];
    wire        g    = den[2*NW-25];
    wire        st   = |den[2*NW-26:0];
    wire [24:0] m24  = kept + (g & (st | kept[0]));
    wire [EXW-1:0] biased = tiny ? {EXW{1'b0}} : er + 126;
    wire [EXW+23:0] tot = {biased, 23'd0} + m24;
    wire ovf = tot >= 32'h7F80_0000;

    wire nan_d = fl_d[2], pinf_d = fl_d[1], ninf_d = fl_d[0];
    wire [31:0] r_d =
        (nan_d | (pinf_d & ninf_d)) ? 32'h7FC0_0000 :
        (pinf_d | ninf_d)           ? {ninf_d, 31'h7F80_0000} :
        zz_d                        ? {nzs_d, 31'd0} :
        ~|mag_d                     ? 32'd0 :
        ovf                         ? {rsgn_d, 31'h7F80_0000} :
                                      {rsgn_d, tot[30:0]};

    // ---- P3
    mx_pipe #(.W(32), .EN(PIPE[3])) u_p3 (.clk(clk), .d(r_d), .q(r));

    // ---- valid: the only reset flops (datapath registers carry no reset)
    wire [4:0] vchain;
    assign vchain[0] = in_valid;
    genvar s;
    generate
        for (s = 0; s < 4; s = s + 1) begin : g_v
            if (PIPE[s]) begin : g_reg
                reg v;
                always @(posedge clk or negedge rst_n)
                    if (!rst_n) v <= 1'b0;
                    else        v <= vchain[s];
                assign vchain[s+1] = v;
            end else begin : g_wire
                assign vchain[s+1] = vchain[s];
            end
        end
    endgenerate
    assign out_valid = vchain[4];
endmodule
