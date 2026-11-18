# Coffee Roast Classification CNN Accelerator on SoC FPGA

## Identitas Mahasiswa
* **Nama:** Rainer Zaviere Raffi
* **NIM:** 101022300061
* **Judul Proyek:** Klasifikasi Tingkat Sangrai Biji Kopi Menggunakan Akselerator 2D-CNN berbasis SoC FPGA

---

## Deskripsi Proyek
Repository ini berisi rancangan dan implementasi sistem **Convolutional Neural Network (2D-CNN)** ringan terkuantisasi **INT8** untuk mengklasifikasikan tingkat sangrai (*roast level*) biji kopi ke dalam 4 kelas (**Green**, **Light**, **Medium**, dan **Dark**). Model dilatih menggunakan dataset citra biji kopi (*Coffee Bean Dataset*) beresolusi **32×32 piksel RGB**, kemudian diuji dan diakselerasi melalui perangkat keras (**Verilog RTL**) untuk target sistem **SoC NIOS-V pada FPGA Cyclone-V (Terasic DE10-Standard)**.

---

## Struktur Repository
1. **`dataset/`**: Berisi dataset citra biji kopi (`train/` dan `test/`) yang diambil dari Kaggle (*Coffee Bean Dataset* — `gpiosenka/coffee-bean-dataset-resized-224-x-224`) serta hasil pra-pemrosesan citra uji ke dalam format **RGB ukuran 32×32 piksel (`.bmp`)** dan vektor memori (`.mem`) di folder `dataset/test_32x32_bmp/`.
2. **`algoritma/`**: Berisi kode Python (`prepare_dataset.py`, `train_cnn_coffee.py`, dan `verify_bmp_python.py`) untuk membaca dan mengonversi citra `.bmp`, melatih model 2D-CNN, melakukan kuantisasi bobot ke **INT8** (`weights_int8.json`), serta memverifikasi akurasi algoritma secara *bit-exact*.
3. **`implementasi/`**: Berisi kode sumber Verilog RTL (`rtl/cnn_coffee_top.v`, `cnn_weights.vh`, dan file memori bobot `.mem`) untuk akselerator 2D-CNN terkuantisasi INT8 beserta testbench (`tb/tb_cnn_coffee.v`) yang dikompilasi dan disimulasikan menggunakan **ModelSim** dan **Quartus Prime** untuk target **Cyclone-V DE10-Standard**.
4. **`output/`**: Berisi dokumentasi progres pengerjaan *step-by-step* (`output/README.md`), hasil visualisasi pembacaan citra `.bmp` 32×32, tabel perbandingan akurasi perangkat lunak (Python) vs perangkat keras (Verilog), serta tangkapan layar hasil simulasi ModelSim (`output/images/`).

---

## Alur Kerja & Metodologi
1. **Preprocessing Citra (Python):** Mengonversi seluruh dataset latih dan uji biji kopi dari Kaggle ke format **RGB berukuran 32×32 piksel (`.bmp`)** serta mengekspor matriks pikselnya menjadi file vektor memori (`test_vectors.mem` dan `test_labels.mem`).
2. **Pelatihan & Validasi Algoritma Bit-Exact (Python):** Melatih arsitektur 2-layer 2D-CNN (`Conv2D 3×3` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling 2×2` $\rightarrow$ `Conv2D 3×3` $\rightarrow$ `ReLU` $\rightarrow$ `GlobalAveragePooling` $\rightarrow$ `Dense`), mengkuantisasi seluruh parameter ke bilangan bulat **INT8**, dan menguji akurasinya pada 400 citra uji `.bmp`.
3. **Desain Akselerator Hardware (Verilog):** Mengimplementasikan logika perhitungan konvolusi 2D, aktivasi ReLU, *max pooling*, *global average pooling*, dan *fully connected layer* berbasis aritmetika integer (`signed`) ke dalam modul `cnn_coffee_top.v`.
4. **Verifikasi RTL & Integrasi SoC (ModelSim & Quartus):** Menguji modul `cnn_coffee_top.v` menggunakan `tb_cnn_coffee.v` di ModelSim untuk memverifikasi kesamaan akurasi 100% (*bit-exact*) terhadap Golden Model Python sebelum diintegrasikan ke sistem **SoC NIOS-V pada FPGA Cyclone-V (DE10-Standard)**.
