# Implementasi Perangkat Keras (Verilog RTL) — Akselerator 2D-CNN

Folder ini berisi desain **Register-Transfer Level (RTL)** untuk akselerator **2-Layer 2D-CNN terkuantisasi INT8** yang dirancang menggunakan bahasa **Verilog** untuk simulasi pada **ModelSim** serta sintesis pada **Intel Quartus Prime** (target FPGA **Cyclone-V DE10-Standard**).

---

## 1. Struktur Direktori `implementasi/`

### `rtl/` (Desain Sintesis & Memori Parameter)
* **`cnn_coffee_top.v`**[cite: 10] — Modul utama (*top-level*) akselerator CNN yang mengimplementasikan *pipeline* inferensi berbasis *Finite State Machine* (FSM) dan aritmetika *fixed-point* bertanda (`40-bit signed`).
* **`cnn_weights.vh`**[cite: 10] — Header Verilog berisi parameter bias 32-bit bertanda (`C1_B`, `C2_B`, `FC_B`) untuk setiap lapisan.
* **`c1_weights.mem`**[cite: 10] — Memori bobot konvolusi lapisan pertama (`8 × 3 × 3 × 3 = 216` parameter INT8).
* **`c2_weights.mem`**[cite: 10] — Memori bobot konvolusi lapisan kedua (`8 × 8 × 3 × 3 = 576` parameter INT8).
* **`fc_weights.mem`**[cite: 10] — Memori bobot *Dense/Fully-Connected layer* (`4 × 8 = 32` parameter INT8).

### `tb/` (Testbench Verifikasi)
* **`tb_cnn_coffee.v`**[cite: 9] — *Testbench* otomatis untuk memuat 400 vektor citra uji RGB 32×32 piksel (`test_vectors.mem`), menjalankan inferensi siklus demi siklus pada `cnn_coffee_top`, memverifikasi bentuk gelombang (*waveform*), dan menghitung akurasi klasifikasi terhadap `test_labels.mem` (**97.00% — 388/400 benar**).

---

## 2. Arsitektur Hardware & Antarmuka Modul (`cnn_coffee_top`)

| Port | Arah | Lebar | Deskripsi |
| :--- | :---: | :---: | :--- |
| `clk` | `input` | 1-bit | Sinyal clock sistem |
| `rst_n` | `input` | 1-bit | Reset aktif-rendah (*active-low asynchronous reset*) |
| `wr_en` | `input` | 1-bit | Sinyal *write-enable* untuk penulisan piksel input ke memori internal |
| `wr_addr` | `input` | 12-bit | Alamat memori piksel (`0` hingga `3071` untuk citra `32×32×3`) |
| `wr_data` | `input` | 8-bit | Data intensitas kanal warna RGB piksel (`0–255`) |
| `start` | `input` | 1-bit | Sinyal pemicu dimulainya proses inferensi FSM |
| `done` | `output` | 1-bit | Indikator proses inferensi selesai dan hasil prediksi siap dibaca |
| `pred_class` | `output` | 2-bit | Kelas prediksi sangrai kopi (`00`: Green, `01`: Light, `10`: Medium, `11`: Dark) |

### Tahapan *Finite State Machine* (FSM)
1. **`S_IDLE` (`3'd0`)**: Menunggu sinyal `start = 1`.
2. **`S_CONV1` (`3'd1`)**: Konvolusi 2D kernel `3×3` pada citra RGB `32×32×3` menjadi `30×30×8`, diikuti fungsi aktivasi `ReLU` dan kuantisasi *arithmetic shift* (`>>> 8`).
3. **`S_POOL1` (`3'd2`)**: *Max-Pooling* `2×2` untuk mereduksi dimensi *feature map* dari `30×30×8` menjadi `15×15×8`.
4. **`S_CONV2` (`3'd3`)**: Konvolusi 2D kernel `3×3` (`15×15×8` $\rightarrow$ `13×13×8`), aktivasi `ReLU`, dan akumulasi *Global Average Pooling* (GAP).
5. **`S_DENSE` (`3'd4`)**: Normalisasi GAP (`>>> 10`), perkalian matriks *Fully-Connected* (`8` fitur $\rightarrow$ `4` kelas), dan pembanding skor maksimum (*argmax*).
6. **`S_DONE` (`3'd5`)**: Mengeluarkan kelas prediksi akhir pada `pred_class` dan mengaktifkan sinyal `done = 1`.

---

## 3. Cara Kompilasi & Simulasi di ModelSim

Buka **ModelSim** dan jalankan perintah berikut pada jendela **Transcript**:

```tcl
cd C:/Rainer/TUGAS/FPGA/Tugas-UTS-CNN/implementasi/tb
vlib work
vlog +incdir+../rtl ../rtl/cnn_coffee_top.v tb_cnn_coffee.v
vsim work.tb_cnn_coffee
run -all
