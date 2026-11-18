`timescale 1ns/1ps

module tb_cnn_coffee;
    localparam NUM_IMAGES = 400;

    reg         clk;
    reg         rst_n;
    reg         wr_en;
    reg  [11:0] wr_addr;
    reg  [7:0]  wr_data;
    reg         start;
    wire        done;
    wire [1:0]  pred_class;

    reg [7:0] all_pixels [0:(NUM_IMAGES*3072)-1];
    reg [3:0] all_labels [0:NUM_IMAGES-1];

    integer img_idx, px_idx, correct_cnt;

    cnn_coffee_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .start(start),
        .done(done),
        .pred_class(pred_class)
    );

    always #10 clk = ~clk;

    initial begin
        $readmemh("../../dataset/test_32x32_bmp/test_vectors.mem", all_pixels);
        $readmemh("../../dataset/test_32x32_bmp/test_labels.mem", all_labels);

        clk = 0;
        rst_n = 0;
        wr_en = 0;
        start = 0;
        correct_cnt = 0;

        #50 rst_n = 1;

        for (img_idx = 0; img_idx < NUM_IMAGES; img_idx = img_idx + 1) begin
            for (px_idx = 0; px_idx < 3072; px_idx = px_idx + 1) begin
                @(posedge clk);
                wr_en   <= 1'b1;
                wr_addr <= px_idx[11:0];
                wr_data <= all_pixels[img_idx*3072 + px_idx];
            end
            @(posedge clk);
            wr_en <= 1'b0;

            @(posedge clk);
            start <= 1'b1;
            wait (done == 1'b1);
            @(posedge clk);
            start <= 1'b0;

            if (pred_class == all_labels[img_idx][1:0])
                correct_cnt = correct_cnt + 1;

            #40;
        end

        $display("========================================");
        $display("Total Gambar Uji (Verilog RGB) : %0d", NUM_IMAGES);
        $display("Prediksi Benar (RTL RGB)       : %0d/%0d", correct_cnt, NUM_IMAGES);
        $display("Akurasi Verilog RTL            : %0d.%02d%%",
                 (correct_cnt * 100) / NUM_IMAGES,
                 ((correct_cnt * 10000) / NUM_IMAGES) % 100);
        $display("========================================");
        $finish;
    end
endmodule