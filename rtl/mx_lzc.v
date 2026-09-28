// Leading-zero count, cnt = W when the input is zero.
//
// Log-depth tree (Oklobdzija, IEEE TVLSI 1994): the input is padded to a
// power of two P with ones below the LSB, so an all-zero input counts to W.
// Level l holds P>>l nodes, each covering 2^l bits, with an all-zero flag z
// and an l-bit count c.  Two siblings merge as
//     z = z_hi & z_lo,    c = z_hi ? {1, c_lo} : {0, c_hi}.
module mx_lzc #(
    parameter W  = 32,
    parameter CW = 6
) (
    input  wire [W-1:0]  in,
    output wire [CW-1:0] cnt
);
    function integer clog2(input integer n);
        integer v;
        begin
            clog2 = 0;
            for (v = n - 1; v > 0; v = v >> 1) clog2 = clog2 + 1;
        end
    endfunction

    localparam LG = clog2(W);
    localparam P  = 1 << LG;

    genvar l, j;
    generate
        if (LG == 0) begin : g_one
            assign cnt = ~in;
        end else begin : g_tree
            wire [P-1:0] x;
            if (P > W) begin : g_pad
                assign x = {in, {(P-W){1'b1}}};
            end else begin : g_nopad
                assign x = in;
            end

            for (l = 0; l <= LG; l = l + 1) begin : lv
                localparam CWL = (l > 0) ? l : 1;
                wire [(P>>l)-1:0]     z;
                wire [(P>>l)*CWL-1:0] c;
            end
            assign lv[0].z = ~x;
            assign lv[0].c = {P{1'b0}};

            for (l = 0; l < LG; l = l + 1) begin : merge
                for (j = 0; j < (P >> (l + 1)); j = j + 1) begin : node
                    wire zh = lv[l].z[2*j+1];
                    wire zl = lv[l].z[2*j];
                    assign lv[l+1].z[j] = zh & zl;
                    if (l == 0) begin : g_leaf
                        assign lv[1].c[j] = zh;
                    end else begin : g_inner
                        wire [l-1:0] ch = lv[l].c[(2*j+1)*l +: l];
                        wire [l-1:0] cl = lv[l].c[(2*j)*l   +: l];
                        assign lv[l+1].c[j*(l+1) +: l+1] = zh ? {1'b1, cl} : {1'b0, ch};
                    end
                end
            end

            wire [CW-1:0] c_root = lv[LG].c[LG-1:0];
            assign cnt = lv[LG].z[0] ? P : c_root;
        end
    endgenerate
endmodule
