// OCP MX block dot product with FP32 accumulate, anchored-truncation policy.
//
//   r = RNE_fp32( sum of trunc_TW(term) ),  terms = c and 2^(xa+xb-254) * a_i * b_i
//
// The comparison baseline for mx_dotp.v (exact).  This is the classic
// many-term dot-product organisation (Kaul et al. ARITH'19, NVIDIA tensor
// cores per Fasi et al.): every term is aligned to the largest *nominal*
// exponent (the anchor) and truncated towards zero to a TW-bit window, the
// window is summed, and the sum is rounded once.  Nominal exponents come
// straight from the operand exponents (products: ea+eb, PSIGW significand
// bits; c: its 24-bit significand); subnormals are not normalised.
// Bit-accurate model: model/mx_policies.py dot_trunc(..., w=TW).
//
// Pipeline cuts selected by PIPE[3:0]:
//   decode + multiply                                  | P0
//   anchor (max tree), align + truncate, adder tree, +c| P1
//   sign-magnitude                                     | P2
//   normalise, denormalise, round, specials            | P3 -> r
module mx_dotp_trunc #(
    parameter FMT_A = 0,
    parameter FMT_B = 0,
    parameter K     = 32,
    parameter PIPE  = 4'b0000,
    parameter TW    = 24,
    parameter EWA   = (FMT_A == 2 || FMT_A == 3) ? 6 : (FMT_A == 4) ? 4 : 8,
    parameter EWB   = (FMT_B == 2 || FMT_B == 3) ? 6 : (FMT_B == 4) ? 4 : 8
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             in_valid,
    input  wire [K*EWA-1:0] a,
    input  wire [K*EWB-1:0] b,
    input  wire [7:0]       xa,
    input  wire [7:0]       xb,
    input  wire [31:0]      c,
    output wire             out_valid,
    output wire [31:0]      r
);
    // ------------------------------------------------------------------ params
    // (same format tables as mx_dotp.v)
    function integer clog2(input integer n);
        integer v;
        begin
            clog2 = 0;
            for (v = n - 1; v > 0; v = v >> 1) clog2 = clog2 + 1;
        end
    endfunction
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
    localparam SHW   = clog2(EOFFM + 1) + 1;   // width of ea+eb
    localparam LOGK  = clog2(K);
    localparam EXW   = 12;
    localparam TPW   = TW + 1;                 // signed truncated term
    localparam SPW   = TPW + LOGK;             // signed sum of the K products
    localparam SWT   = SPW + 1;                // + c term
    localparam MWT   = SWT - 1;                // |sum|
    localparam NWT   = imax(MWT, 26);          // normalisation width (>= 24 + G + S)
    localparam LZNW  = clog2(NWT + 1);
    localparam DW    = clog2(TW + 1);          // alignment shift width (clamped to TW)
    localparam FLW   = 4;                      // {nan, pos_inf, neg_inf, all_products_neg_zero}

    // ============================================== stage A: decode + multiply
    wire [K*PSIGW-1:0] pm_a;
    wire [K*SHW-1:0]   ee_a;
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
            wire [SHW-1:0] ee = ea + eb;
            assign pm_a[i*PSIGW +: PSIGW] = ma * mb;
            assign ee_a[i*SHW +: SHW]     = ee;
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
    wire [K*PSIGW-1:0] pm_b;
    wire [K*SHW-1:0]   ee_b;
    wire [K-1:0]       ps_b;
    wire [7:0]  xa_b, xb_b;
    wire [31:0] c_b;
    wire [FLW-1:0] fl_b;
    mx_pipe #(.W(K*PSIGW + K*SHW + K + 16 + 32 + FLW), .EN(PIPE[0])) u_p0 (.clk(clk),
        .d({pm_a, ee_a, ps_a, xa, xb, c, fl_a}),
        .q({pm_b, ee_b, ps_b, xa_b, xb_b, c_b, fl_b}));

    // ============================================== stage B: anchor, align, sum
    // largest ea+eb over non-zero products: balanced max tree
    wire [K-1:0] nzp;
    wire [(2*K-1)*SHW-1:0] mx;                // heap: node n has children 2n+1, 2n+2
    generate
        for (i = 0; i < K; i = i + 1) begin : g_nzp
            assign nzp[i] = |pm_b[i*PSIGW +: PSIGW];
            assign mx[(K-1+i)*SHW +: SHW] = nzp[i] ? ee_b[i*SHW +: SHW] : {SHW{1'b0}};
        end
        for (i = 0; i < K - 1; i = i + 1) begin : g_max
            wire [SHW-1:0] m0 = mx[(2*i+1)*SHW +: SHW];
            wire [SHW-1:0] m1 = mx[(2*i+2)*SHW +: SHW];
            assign mx[i*SHW +: SHW] = (m0 > m1) ? m0 : m1;
        end
    endgenerate
    wire [SHW-1:0] ee_max = mx[0 +: SHW];
    wire any_p = |nzp;

    // nominal top exponents
    wire signed [EXW-1:0] top_pbase = $signed({{(EXW-8){1'b0}}, xa_b})
                                    + $signed({{(EXW-8){1'b0}}, xb_b})
                                    + (EPMIN - 254 + PSIGW - 1);
    wire signed [EXW-1:0] top_pmax  = top_pbase + $signed({{(EXW-SHW){1'b0}}, ee_max});
    wire [7:0]  ec = c_b[30:23];
    wire [23:0] mC = {|ec, c_b[22:0]};
    wire nz_c = |mC;
    wire signed [EXW-1:0] top_c = $signed({{(EXW-8){1'b0}}, (|ec) ? ec : 8'd1}) - 127;

    wire signed [EXW-1:0] anchor = ~any_p ? top_c :
                                   ~nz_c  ? top_pmax :
                                   (top_c > top_pmax) ? top_c : top_pmax;

    // product shifts: d_i = (anchor - top_pbase) - ee_i, clamped to TW
    wire signed [EXW-1:0] ap_raw = anchor - top_pbase;
    wire [EXW-1:0] ap = (ap_raw < 0) ? {EXW{1'b0}} :
                        (ap_raw > TW + EOFFM) ? TW + EOFFM : ap_raw;
    wire signed [EXW-1:0] dc_raw = anchor - top_c;
    wire [DW-1:0] dC = (dc_raw < 0) ? {DW{1'b0}} : (dc_raw > TW) ? TW : dc_raw[DW-1:0];

    // aligned, truncated, signed product terms -> adder tree level 0
    function integer lw(input integer lv);
        lw = TPW + lv;
    endfunction
    function integer loff(input integer lv);
        integer jj;
        begin
            loff = 0;
            for (jj = 0; jj < lv; jj = jj + 1) loff = loff + (K >> jj) * lw(jj);
        end
    endfunction
    localparam TREEW = loff(LOGK + 1);
    wire [TREEW-1:0] tv;

    generate
        for (i = 0; i < K; i = i + 1) begin : g_align
            wire [PSIGW-1:0] pm = pm_b[i*PSIGW +: PSIGW];
            wire [TW-1:0] win;
            if (TW >= PSIGW) begin : g_wide
                wire [TW-1:0] pmx = pm;
                assign win = pmx << (TW - PSIGW);
            end else begin : g_narrow
                assign win = pm >> (PSIGW - TW);
            end
            wire [EXW-1:0] d_raw = ap - {{(EXW-SHW){1'b0}}, ee_b[i*SHW +: SHW]};
            wire [DW-1:0]  d     = (d_raw > TW) ? TW : d_raw[DW-1:0];
            wire [TW-1:0]  sh    = win >> d;
            wire [TPW-1:0] tz    = {1'b0, sh};
            assign tv[i*TPW +: TPW] = ps_b[i] ? (~tz + 1'b1) : tz;
        end
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
    wire [SPW-1:0] sum_p = tv[ROOT +: SPW];

    // c term
    wire [TW-1:0] win_c;
    generate
        if (TW >= 24) begin : g_cwide
            wire [TW-1:0] mcx = mC;
            assign win_c = mcx << (TW - 24);
        end else begin : g_cnarrow
            assign win_c = mC >> (24 - TW);
        end
    endgenerate
    wire [TW-1:0]  sh_c = win_c >> dC;
    wire [TPW-1:0] tz_c = {1'b0, sh_c};
    wire [TPW-1:0] t_c  = c_b[31] ? (~tz_c + 1'b1) : tz_c;
    wire [SWT-1:0] sum_b = {sum_p[SPW-1], sum_p} + {{(SWT-TPW){t_c[TPW-1]}}, t_c};

    wire zz_b  = ~any_p & ~nz_c;               // every term exactly zero
    wire nzs_b = fl_b[0] & c_b[31];            // sign of that zero

    // ---- P1
    wire [SWT-1:0] sum_c;
    wire signed [EXW-1:0] anc_c;
    wire zz_c, nzs_c;
    wire [2:0] fl_c;
    mx_pipe #(.W(SWT + EXW + 2 + 3), .EN(PIPE[1])) u_p1 (.clk(clk),
        .d({sum_b, anchor, zz_b, nzs_b, fl_b[3:1]}),
        .q({sum_c, anc_c,  zz_c, nzs_c, fl_c}));

    // ============================================== stage C: sign-magnitude
    wire           rsgn_c = sum_c[SWT-1];
    wire [MWT-1:0] mag_c  = rsgn_c ? (~sum_c[MWT-1:0] + 1'b1) : sum_c[MWT-1:0];

    // ---- P2
    wire [MWT-1:0] mag_d;
    wire signed [EXW-1:0] anc_d;
    wire rsgn_d, zz_d, nzs_d;
    wire [2:0] fl_d;
    mx_pipe #(.W(MWT + EXW + 3 + 3), .EN(PIPE[2])) u_p2 (.clk(clk),
        .d({mag_c, anc_c, rsgn_c, zz_c, nzs_c, fl_c}),
        .q({mag_d, anc_d, rsgn_d, zz_d, nzs_d, fl_d}));

    // ============================================== stage D: normalise + round
    // value = mag * 2^(anchor - TW + 1); with magx NWT bits wide the MSB
    // exponent is anchor - TW + NWT - lz.
    wire [NWT-1:0] magx = mag_d;
    wire [LZNW-1:0] lz;
    mx_lzc #(.W(NWT), .CW(LZNW)) u_lzn (.in(magx), .cnt(lz));
    wire [NWT-1:0] norm = magx << lz;
    wire signed [EXW-1:0] er = anc_d + (NWT - TW)
                             - $signed({{(EXW-LZNW){1'b0}}, lz});
    wire tiny = er < -126;
    wire signed [EXW-1:0] rs_raw = -126 - er;
    wire [EXW-1:0] rs = ~tiny ? {EXW{1'b0}} : (rs_raw > NWT) ? NWT : rs_raw;
    wire [2*NWT-1:0] den = {norm, {NWT{1'b0}}} >> rs;
    wire [23:0] kept = den[2*NWT-1 -: 24];
    wire        g    = den[2*NWT-25];
    wire        st   = |den[2*NWT-26:0];
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

    // ---- valid: the only reset flops
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
