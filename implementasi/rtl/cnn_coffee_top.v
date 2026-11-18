module cnn_coffee_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        wr_en,
    input  wire [11:0] wr_addr,
    input  wire [7:0]  wr_data,
    input  wire        start,
    output reg         done,
    output reg  [1:0]  pred_class
);
    `include "cnn_weights.vh"

    reg [7:0] img_mem [0:3071];
    reg signed [7:0] c1_w [0:215];
    reg signed [7:0] c2_w [0:575];
    reg signed [7:0] fc_w [0:31];

    initial begin
        $readmemh("../rtl/c1_weights.mem", c1_w);
        $readmemh("../rtl/c2_weights.mem", c2_w);
        $readmemh("../rtl/fc_weights.mem", fc_w);
    end

    always @(posedge clk) begin
        if (wr_en)
            img_mem[wr_addr] <= wr_data;
    end

    reg signed [31:0] conv1_buf [0:7199];
    reg signed [31:0] pool1_buf [0:1799];
    reg signed [39:0] gap_sum   [0:7];

    localparam S_IDLE  = 3'd0,
               S_CONV1 = 3'd1,
               S_POOL1 = 3'd2,
               S_CONV2 = 3'd3,
               S_DENSE = 3'd4,
               S_DONE  = 3'd5;

    reg [2:0] state;
    reg [3:0] f, ch;
    reg [4:0] r, c;
    integer kr, kc, cls_i, fi;
    reg signed [31:0] cur_bias;
    reg signed [39:0] op_a, op_b;
    reg signed [39:0] acc, p0, p1, p2, p3, m01, m23, mmax;
    reg signed [39:0] best_logit, cur_logit;
    reg [1:0] best_cls;

    function signed [31:0] get_c1_bias;
        input [2:0] idx;
        begin
            case (idx)
                3'd0: get_c1_bias = C1_B_0;
                3'd1: get_c1_bias = C1_B_1;
                3'd2: get_c1_bias = C1_B_2;
                3'd3: get_c1_bias = C1_B_3;
                3'd4: get_c1_bias = C1_B_4;
                3'd5: get_c1_bias = C1_B_5;
                3'd6: get_c1_bias = C1_B_6;
                default: get_c1_bias = C1_B_7;
            endcase
        end
    endfunction

    function signed [31:0] get_c2_bias;
        input [2:0] idx;
        begin
            case (idx)
                3'd0: get_c2_bias = C2_B_0;
                3'd1: get_c2_bias = C2_B_1;
                3'd2: get_c2_bias = C2_B_2;
                3'd3: get_c2_bias = C2_B_3;
                3'd4: get_c2_bias = C2_B_4;
                3'd5: get_c2_bias = C2_B_5;
                3'd6: get_c2_bias = C2_B_6;
                default: get_c2_bias = C2_B_7;
            endcase
        end
    endfunction

    function signed [31:0] get_fc_bias;
        input [1:0] idx;
        begin
            case (idx)
                2'd0: get_fc_bias = FC_B_0;
                2'd1: get_fc_bias = FC_B_1;
                2'd2: get_fc_bias = FC_B_2;
                default: get_fc_bias = FC_B_3;
            endcase
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            pred_class <= 2'd0;
            f <= 4'd0; r <= 5'd0; c <= 5'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        f <= 4'd0; r <= 5'd0; c <= 5'd0;
                        state <= S_CONV1;
                    end
                end

                S_CONV1: begin
                    cur_bias = get_c1_bias(f[2:0]);
                    acc = {{8{cur_bias[31]}}, cur_bias};
                    for (ch = 0; ch < 3; ch = ch + 1) begin
                        for (kr = 0; kr < 3; kr = kr + 1) begin
                            for (kc = 0; kc < 3; kc = kc + 1) begin
                                op_a = {32'sd0, img_mem[(((r+kr)*32 + (c+kc))*3) + ch]};
                                op_b = {{32{c1_w[f*27 + ch*9 + kr*3 + kc][7]}}, c1_w[f*27 + ch*9 + kr*3 + kc]};
                                acc  = acc + (op_a * op_b);
                            end
                        end
                    end
                    conv1_buf[f*900 + r*30 + c] <= (acc > 0) ? (acc >>> 8) : 32'sd0;

                    if (c == 5'd29) begin
                        c <= 5'd0;
                        if (r == 5'd29) begin
                            r <= 5'd0;
                            if (f == 4'd7) begin
                                f <= 4'd0;
                                state <= S_POOL1;
                            end else
                                f <= f + 4'd1;
                        end else
                            r <= r + 5'd1;
                    end else
                        c <= c + 5'd1;
                end

                S_POOL1: begin
                    p0 = {{8{conv1_buf[f*900 + (r*2)*30 + (c*2)][31]}},     conv1_buf[f*900 + (r*2)*30 + (c*2)]};
                    p1 = {{8{conv1_buf[f*900 + (r*2)*30 + (c*2+1)][31]}},   conv1_buf[f*900 + (r*2)*30 + (c*2+1)]};
                    p2 = {{8{conv1_buf[f*900 + (r*2+1)*30 + (c*2)][31]}},   conv1_buf[f*900 + (r*2+1)*30 + (c*2)]};
                    p3 = {{8{conv1_buf[f*900 + (r*2+1)*30 + (c*2+1)][31]}}, conv1_buf[f*900 + (r*2+1)*30 + (c*2+1)]};
                    m01 = (p0 > p1) ? p0 : p1;
                    m23 = (p2 > p3) ? p2 : p3;
                    mmax = (m01 > m23) ? m01 : m23;
                    pool1_buf[f*225 + r*15 + c] <= mmax[31:0];

                    if (c == 5'd14) begin
                        c <= 5'd0;
                        if (r == 5'd14) begin
                            r <= 5'd0;
                            if (f == 4'd7) begin
                                f <= 4'd0;
                                gap_sum[0] <= 40'sd0; gap_sum[1] <= 40'sd0;
                                gap_sum[2] <= 40'sd0; gap_sum[3] <= 40'sd0;
                                gap_sum[4] <= 40'sd0; gap_sum[5] <= 40'sd0;
                                gap_sum[6] <= 40'sd0; gap_sum[7] <= 40'sd0;
                                state <= S_CONV2;
                            end else
                                f <= f + 4'd1;
                        end else
                            r <= r + 5'd1;
                    end else
                        c <= c + 5'd1;
                end

                S_CONV2: begin
                    cur_bias = get_c2_bias(f[2:0]);
                    acc = {{8{cur_bias[31]}}, cur_bias};
                    for (ch = 0; ch < 8; ch = ch + 1) begin
                        for (kr = 0; kr < 3; kr = kr + 1) begin
                            for (kc = 0; kc < 3; kc = kc + 1) begin
                                op_a = {{8{pool1_buf[ch*225 + (r+kr)*15 + (c+kc)][31]}},
                                        pool1_buf[ch*225 + (r+kr)*15 + (c+kc)]};
                                op_b = {{32{c2_w[f*72 + ch*9 + kr*3 + kc][7]}}, c2_w[f*72 + ch*9 + kr*3 + kc]};
                                acc  = acc + (op_a * op_b);
                            end
                        end
                    end
                    if (acc > 0)
                        gap_sum[f] <= gap_sum[f] + (acc >>> 8);

                    if (c == 5'd12) begin
                        c <= 5'd0;
                        if (r == 5'd12) begin
                            r <= 5'd0;
                            if (f == 4'd7)
                                state <= S_DENSE;
                            else
                                f <= f + 4'd1;
                        end else
                            r <= r + 5'd1;
                    end else
                        c <= c + 5'd1;
                end

                S_DENSE: begin
                    best_cls = 2'd0;
                    for (cls_i = 0; cls_i < 4; cls_i = cls_i + 1) begin
                        cur_bias  = get_fc_bias(cls_i[1:0]);
                        cur_logit = {{8{cur_bias[31]}}, cur_bias};
                        for (fi = 0; fi < 8; fi = fi + 1) begin
                            op_a = gap_sum[fi] >>> 10;
                            op_b = {{32{fc_w[cls_i*8 + fi][7]}}, fc_w[cls_i*8 + fi]};
                            cur_logit = cur_logit + (op_a * op_b);
                        end
                        if (cls_i == 0 || cur_logit > best_logit) begin
                            best_logit = cur_logit;
                            best_cls = cls_i[1:0];
                        end
                    end
                    pred_class <= best_cls;
                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start)
                        state <= S_IDLE;
                end
            endcase
        end
    end
endmodule