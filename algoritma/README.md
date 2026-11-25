# Algoritma & Golden Model Python — 2D-CNN Klasifikasi Tingkat Sangrai Kopi

Folder ini berisi skrip Python dan notebook Google Colab untuk pra-pemrosesan citra `.bmp`, pelatihan model, kuantisasi parameter ke **INT8**, visualisasi *feature map*, serta verifikasi inferensi *bit-exact* sebagai acuan (*Golden Model*) sebelum diimplementasikan ke Verilog RTL.

---

## 1. Daftar File di Folder `algoritma/`

* **`train_and_verify_cnn.ipynb`** — Notebook Google Colab lengkap yang memuat seluruh alur kerja *end-to-end* beserta output eksekusi: pra-pemrosesan dataset, pelatihan 2-layer 2D-CNN, kuantisasi bobot/bias ke integer bertanda (`INT8`/`INT32`), ekspor parameter ke Verilog, simulasi verifikasi RTL, serta grafik evaluasi.
* **`prepare_dataset.py`** — Skrip pra-pemrosesan citra asli dataset Kaggle (`224×224` PNG) menjadi **400 citra uji beresolusi 32×32 piksel RGB (`.bmp`)** sekaligus menyusun file vektor memori `test_vectors.mem` dan `test_labels.mem`.
* **`train_cnn_coffee.py`** — Skrip pelatihan arsitektur 2-layer 2D-CNN (`Conv2D 3×3` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool 2×2` $\rightarrow$ `Conv2D 3×3` $\rightarrow$ `ReLU` $\rightarrow$ `GAP` $\rightarrow$ `Dense`) dan kuantisasi parameter ke `weights_int8.json` serta file memori Verilog (`.mem` & `.vh`).
* **`run_cnn_coffee.py`** — Skrip inferensi *integer-only* untuk membaca seluruh 400 file `.bmp` 32×32 RGB, memvisualisasikan *feature map* hasil lapisan konvolusi pertama + aktivasi `ReLU` (`30×30×8`), dan menghitung akurasi uji (**97.00% — 388/400 benar, 100.00% *bit-exact* dengan Verilog RTL**).
* **`verify_bmp_python.py`** — Skrip verifikasi pembacaan format file `.bmp` dan tampilan grid sampel citra uji dari keempat kelas (*Green*, *Light*, *Medium*, *Dark*).
* **`weights_int8.json`** — File penyimpanan bobot dan bias terkuantisasi integer (`c1_w`, `c1_b`, `c2_w`, `c2_b`, `fc_w`, `fc_b`) yang identik dengan memori parameter pada modul Verilog `cnn_coffee_top.v`.

---

## 2. Spesifikasi Arsitektur & Kuantisasi Integer

| Lapisan (*Layer*) | Dimensi Input | Operasi & Parameter | Dimensi Output | Representasi Numerik |
| :--- | :---: | :--- | :---: | :---: |
| **Input** | `32 × 32 × 3` | Citra RGB `.bmp` | `32 × 32 × 3` | `UINT8` (`0` – `255`) |
| **Conv2D_1 + ReLU** | `32 × 32 × 3` | 8 filter `3×3`, *stride* 1, `ReLU`, *shift* `>> 8` | `30 × 30 × 8` | Bobot `INT8`, Akumulator `INT40` |
| **MaxPooling2D_1** | `30 × 30 × 8` | Pool `2×2`, *stride* 2 | `15 × 15 × 8` | `INT32` |
| **Conv2D_2 + ReLU** | `15 × 15 × 8` | 8 filter `3×3`, *stride* 1, `ReLU`, *shift* `>> 8` | `13 × 13 × 8` | Bobot `INT8`, Akumulator `INT40` |
| **Global Average Pool** | `13 × 13 × 8` | Akumulasi spasial & *shift* `>> 10` | `8` | `INT40` |
| **Dense (Classifier)** | `8` | Matriks `8 × 4` + Bias + `Argmax` | `4` kelas | Bobot `INT8`, Logit `INT40` |
