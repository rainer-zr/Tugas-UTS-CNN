`timescale 1ns / 1ps

module tb_cnn_coffee;

    parameter NUM_IMAGES  = 400;
    parameter IMG_BYTES   = 3072; // 32 x 32 x 3 (RGB)
    parameter CLK_PERIOD  = 20;   // 50 MHz

    reg         clk;
    reg         rst_n;
    reg         wr_en;
    reg  [11:0] wr_addr;
    reg  [7:0]  wr_data;
    reg         start;
    wire        done;
    wire [1:0]  pred_class;

    reg [7:0] all_pixels [0:(NUM_IMAGES * IMG_BYTES) - 1];
    reg [7:0] all_labels [0:NUM_IMAGES - 1];

    integer img_idx;
    integer px_idx;
    integer correct_cnt;
    integer cycle_cnt;
    integer log_file;
    real    accuracy;

    cnn_coffee_top dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .wr_en      (wr_en),
        .wr_addr    (wr_addr),
        .wr_data    (wr_data),
        .start      (start),
        .done       (done),
        .pred_class (pred_class)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    initial begin
        rst_n       = 1'b0;
        wr_en       = 1'b0;
        wr_addr     = 12'd0;
        wr_data     = 8'd0;
        start       = 1'b0;
        correct_cnt = 0;

        $readmemh("../../dataset/test_32x32_bmp/test_vectors.mem", all_pixels);
        $readmemh("../../dataset/test_32x32_bmp/test_labels.mem",  all_labels);

        log_file = $fopen("rtl_predictions.csv", "w");
        $fdisplay(log_file, "img_idx,true_label,rtl_pred,match_gt,latency_cycles");

        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);

        for (img_idx = 0; img_idx < NUM_IMAGES; img_idx = img_idx + 1) begin
            for (px_idx = 0; px_idx < IMG_BYTES; px_idx = px_idx + 1) begin
                @(posedge clk);
                wr_en   <= 1'b1;
                wr_addr <= px_idx[11:0];
                wr_data <= all_pixels[(img_idx * IMG_BYTES) + px_idx];
            end

            @(posedge clk);
            wr_en <= 1'b0;

            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            cycle_cnt = 0;
            while (done !== 1'b1 && cycle_cnt < 500000) begin
                @(posedge clk);
                cycle_cnt = cycle_cnt + 1;
            end

            if (cycle_cnt >= 500000) begin
                $display("ERROR: Timeout pada gambar ke-%0d", img_idx);
                $fclose(log_file);
                $finish;
            end

            if (pred_class === all_labels[img_idx][1:0]) begin
                correct_cnt = correct_cnt + 1;
                $fdisplay(log_file, "%0d,%0d,%0d,1,%0d",
                          img_idx, all_labels[img_idx][1:0], pred_class, cycle_cnt);
            end else begin
                $fdisplay(log_file, "%0d,%0d,%0d,0,%0d",
                          img_idx, all_labels[img_idx][1:0], pred_class, cycle_cnt);
            end

            @(posedge clk);
            if (done === 1'b1) begin
                rst_n <= 1'b0;
                @(posedge clk);
                rst_n <= 1'b1;
                @(posedge clk);
            end
        end

        accuracy = (correct_cnt * 100.0) / NUM_IMAGES;
        $display("========================================");
        $display("Total Gambar Uji (Verilog RGB) : %0d", NUM_IMAGES);
        $display("Prediksi Benar (RTL RGB)       : %0d/%0d", correct_cnt, NUM_IMAGES);
        $display("Akurasi Verilog RTL            : %0.2f%%", accuracy);
        $display("========================================");

        $fclose(log_file);
        $finish;
    end

endmodule
