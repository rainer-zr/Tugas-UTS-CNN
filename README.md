# Akselerator Perangkat Keras 2D-CNN untuk Klasifikasi Tingkat Sangrai Biji Kopi pada FPGA DE10-Standard

## Identitas Mahasiswa
* **Nama:** Rainer Zaviere Raffi
* **NIM:** 101022300061
* **Program Studi:** S1 Teknik Elektro — Fakultas Teknik Elektro, Telkom University
* **Target Perangkat Keras:** Terasic DE10-Standard (Intel Cyclone V SX SoC — `5CSXFC6D6F31C6`)

---

## Deskripsi Proyek
Repository ini berisi perancangan dan implementasi akselerator perangkat keras **2D Convolutional Neural Network (2D-CNN)** terkuantisasi **INT8** untuk mengklasifikasikan tingkat sangrai (*roast level*) biji kopi ke dalam 4 kelas (**Green**, **Light**, **Medium**, dan **Dark**).

Seluruh arsitektur diverifikasi secara *bit-exact* menggunakan operasi aritmetika bilangan bulat (*integer-only*) antara **Golden Model Python** dan **Verilog RTL** pada simulator **ModelSim**, sehingga keluaran prediksi perangkat keras cocok 100% dengan simulasi algoritma perangkat lunak.

---

## Struktur Repository
```text
Tugas-UTS-CNN/
├── README.md                              # Dokumentasi utama proyek & struktur repository
├── dataset/
│   ├── README.md                          # Sumber dataset & spesifikasi pra-pemrosesan citra
│   ├── train/                             # Citra latih biji kopi (Green, Light, Medium, Dark)
│   ├── test/                              # Citra uji asli biji kopi
│   └── test_32x32_bmp/                    # Citra uji 32x32 piksel (.bmp) & vektor memori (.mem)
├── algoritma/
│   ├── prepare_dataset.py                 # Konversi citra ke format 32x32 .bmp & file .mem Verilog
│   ├── train_cnn_coffee.py                # Pelatihan model 2D-CNN & kuantisasi bobot ke INT8
│   ├── verify_bmp_python.py               # Golden model inferensi bit-exact & pengukuran akurasi
│   └── weights_int8.json                  # Parameter bobot dan bias terkuantisasi INT8
├── implementasi/
│   ├── rtl/
│   │   ├── cnn_coffee_top.v               # Modul top-level akselerator 2D-CNN Verilog RTL
│   │   ├── cnn_weights.vh                 # Parameter bias terkuantisasi untuk RTL
│   │   ├── c1_weights.mem                 # Memori bobot kernel Conv2D Layer 1 (INT8)
│   │   ├── c2_weights.mem                 # Memori bobot kernel Conv2D Layer 2 (INT8)
│   │   └── fc_weights.mem                 # Memori bobot Fully Connected Layer (INT8)
│   └── tb/
│       └── tb_cnn_coffee.v                # Testbench ModelSim untuk pengujian otomatis 400 citra
└── output/
    ├── README.md                          # Laporan progres bertahap & perbandingan akurasi
    └── images/                            # Dokumentasi visual hasil Python & simulasi ModelSim
