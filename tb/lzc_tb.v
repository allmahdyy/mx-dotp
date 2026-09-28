// Standalone check of rtl/mx_lzc.v against a behavioural count.
//   iverilog -g2012 -o lzc_tb tb/lzc_tb.v rtl/mx_lzc.v && vvp lzc_tb
module lzc_check #(parameter W = 8) ();
    localparam CW = $clog2(W + 1);
    reg  [W-1:0]  x;
    wire [CW-1:0] got;
    integer i, k, ref_cnt, errors;
    mx_lzc #(.W(W), .CW(CW)) dut (.in(x), .cnt(got));

    task check;
        begin
            ref_cnt = W;
            for (k = 0; k < W; k = k + 1) if (x[k]) ref_cnt = W - 1 - k;
            #1;
            if (got !== ref_cnt) begin
                errors = errors + 1;
                if (errors < 5) $display("W=%0d x=%h got %0d want %0d", W, x, got, ref_cnt);
            end
        end
    endtask

    initial begin
        errors = 0;
        if (W <= 16) begin
            for (i = 0; i < (1 << W); i = i + 1) begin x = i; check; end
        end else begin
            for (i = 0; i < 20000; i = i + 1) begin
                // random value with a random number of leading zeros
                x = {$urandom, $urandom, $urandom} >> ($urandom % (W + 1));
                check;
            end
            x = 0; check;
        end
        $display("W=%0d: %0d errors", W, errors);
    end
endmodule

module lzc_tb;
    lzc_check #(1)  w1 ();
    lzc_check #(5)  w5 ();
    lzc_check #(8)  w8 ();
    lzc_check #(13) w13 ();
    lzc_check #(16) w16 ();
    lzc_check #(24) w24 ();
    lzc_check #(28) w28 ();
    lzc_check #(73) w73 ();
endmodule
