# Coffee Roast Classification CNN Accelerator on SoC FPGA

## Identitas Mahasiswa
- **Nama:** Rainer Zaviere Raffi
- **NIM:** 101022300061
- **Judul Proyek:** Klasifikasi Tingkat Sangrai Biji Kopi Menggunakan Akselerator 2D-CNN berbasis SoC FPGA

---

## Deskripsi Proyek
Repository ini berisi rancangan dan implementasi sistem **2D Convolutional Neural Network (2D-CNN)** ringan terkuantisasi **INT8** untuk mengklasifikasikan tingkat sangrai (*roast level*) biji kopi ke dalam 4 kelas (**Green**, **Light**, **Medium**, dan **Dark**). Model dilatih dan diverifikasi menggunakan dataset citra biji kopi (*Coffee Bean Dataset*) beresolusi **32×32 piksel RGB**, kemudian diakselerasi pada perangkat keras (**Verilog RTL**) untuk target sistem **SoC NIOS-V pada FPGA Intel Cyclone-V (Terasic DE10-Standard)**.

---

## Struktur Repository
1. **`dataset/`**: Berisi dataset citra biji kopi (`train/` 1.200 citra dan `test/` 400 citra) yang diambil dari Kaggle (*Coffee Bean Dataset* — `gpiosenka/coffee-bean-dataset-resized-224-x-224`) serta hasil pra-pemrosesan 400 citra uji ke dalam format **RGB ukuran 32×32 piksel (`.bmp`)** dan vektor memori (`.mem`) di folder `dataset/test_32x32_bmp/`.
2. **`algoritma/`**: Berisi kode Python (`prepare_dataset.py`, `train_cnn_coffee.py`, `run_cnn_coffee.py`, `verify_bmp_python.py`) dan notebook Google Colab (`train_and_verify_cnn.ipynb`) untuk membaca citra `.bmp`, memvisualisasikan *feature map* lapisan konvolusi dan aktivasi ReLU, menyimpan bobot terkuantisasi **INT8** (`weights_int8.json`), serta memverifikasi akurasi algoritma secara *integer-only*.
3. **`implementasi/`**: Berisi kode sumber Verilog RTL (`rtl/cnn_coffee_top.v`, `cnn_weights.vh`, dan file memori bobot `c1_weights.mem`, `c2_weights.mem`, `fc_weights.mem`) untuk akselerator 2D-CNN terkuantisasi INT8 beserta testbench (`tb/tb_cnn_coffee.v`) yang dikompilasi dan disimulasikan menggunakan **ModelSim** dan **Quartus Prime** untuk target **Cyclone-V DE10-Standard**.
4. **`output/`**: Berisi laporan progres pengerjaan *step-by-step* (`output/README.md`), tabel perbandingan akurasi perangkat lunak (Python) vs perangkat keras (Verilog RTL), visualisasi pembacaan citra `.bmp` & *feature map* ReLU, serta tangkapan layar hasil simulasi Transcript dan Waveform ModelSim (`output/images/`).

---

## Alur Kerja & Metodologi Step-by-Step

### Step 1: Akuisisi & Pra-Pemrosesan Dataset (`dataset/` & `algoritma/prepare_dataset.py`)
* Mengambil dataset citra biji kopi dari Kaggle (`gpiosenka/coffee-bean-dataset-resized-224-x-224`) yang terdiri atas 4 kelas sangrai (`0: green`, `1: light`, `2: medium`, `3: dark`).
* Melakukan *resizing* pada seluruh 400 citra uji menjadi resolusi target **32×32 piksel RGB** dalam format **`.bmp`** (`dataset/test_32x32_bmp/`).
* Mengekspor nilai piksel RGB seluruh 400 citra uji ke dalam format heksadesimal **`test_vectors.mem`** (`400 × 3.072 = 1.228.800` byte) dan **`test_labels.mem`** agar dapat dibaca langsung oleh testbench Verilog menggunakan `$readmemh`.

### Step 2: Pelatihan Model & Kuantisasi Integer-Only (`algoritma/train_cnn_coffee.py`)
* Membangun dan melatih arsitektur 2-layer 2D-CNN ringan:
  `Input (32×32×3)` $\rightarrow$ `Conv2D 3×3 (8 filter) + ReLU` $\rightarrow$ `MaxPooling 2×2` $\rightarrow$ `Conv2D 3×3 (8 filter) + ReLU` $\rightarrow$ `GlobalAveragePooling` $\rightarrow$ `Dense (4 kelas)`.
* Melakukan *post-training quantization* (PTQ) untuk mengonversi seluruh bobot (`c1_w`, `c2_w`, `fc_w`) menjadi bilangan bulat bertanda **8-bit (`INT8`)** dan bias menjadi **32-bit (`INT32`)**, lalu menyimpannya ke `weights_int8.json` serta file memori hardware (`.mem` dan `.vh`).

### Step 3: Validasi Algoritma & Visualisasi Feature Map Python (`algoritma/run_cnn_coffee.py`)
* Membaca kembali file citra `.bmp` 32×32 piksel menggunakan Python dan menampilkannya untuk memastikan integritas pembacaan kanal warna RGB.
* Memvisualisasikan 8 kanal *feature map* (`30×30×8`) hasil ekstraksi fitur lapisan konvolusi pertama dan aktivasi `ReLU`.
* Menjalankan inferensi *integer-only* (tanpa *floating-point*, mengganti pembagian dengan operasi *bit-shift* `>> 8` dan `>> 10`) pada 400 citra uji `.bmp`, menghasilkan akurasi **96.50% (386/400 benar)** sebagai *Golden Model*.

### Step 4: Perancangan Akselerator Hardware Verilog RTL (`implementasi/rtl/cnn_coffee_top.v`)
* Menerjemahkan algoritma 2D-CNN terkuantisasi ke dalam deskripsi perangkat keras **Verilog RTL** berbasis *Finite State Machine* (FSM) 6 tahap (`S_IDLE` $\rightarrow$ `S_CONV1` $\rightarrow$ `S_POOL1` $\rightarrow$ `S_CONV2` $\rightarrow$ `S_DENSE` $\rightarrow$ `S_DONE`).
* Menggunakan jalur data akumulator bertanda **40-bit (`signed [39:0]`)** untuk mencegah *overflow* selama operasi *Multiply-Accumulate* (MAC) serta operasi *arithmetic right shift* (`>>>`) yang ekuivalen dengan model Python.

### Step 5: Simulasi Verifikasi & Uji Akurasi di ModelSim (`implementasi/tb/tb_cnn_coffee.v`)
* Mengompilasi desain RTL dan testbench di **ModelSim**, memuat bobot `.mem` dan 400 citra uji ke memori modul, serta mengamati gelombang sinyal (*waveform*) per siklus clock (`clk`, `wr_en`, `start`, transisi `state`, `done`, dan `pred_class`).
* Mengevaluasi hasil klasifikasi seluruh 400 citra uji di ModelSim dengan capaian akurasi **97.00% (388/400 benar)** — selaras dengan Golden Model Python (**96.50%**). Dokumentasi lengkap dan bukti visual tersedia pada folder [`output/`](output/README.md).

---

## Ringkasan Hasil Pengujian

| Lingkungan Pengujian | Metode Komputasi | Jumlah Citra Uji | Prediksi Benar | Akurasi |
| :--- | :---: | :---: | :---: | :---: |
| **Python Golden Model (`run_cnn_coffee.py`)** | *Integer-Only* (`INT8` / `INT64`) | 400 citra `.bmp` 32×32 RGB | 386 / 400 | **96.50%** |
| **Verilog RTL ModelSim (`cnn_coffee_top.v`)** | *Hardware FSM* (`INT8` / `INT40`) | 400 citra `.bmp` 32×32 RGB | 388 / 400 | **97.00%** |
