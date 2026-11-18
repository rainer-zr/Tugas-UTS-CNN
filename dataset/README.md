# Dataset: Coffee Bean Roast Level (32×32 RGB)

* **Sumber Asli:** Kaggle — *Coffee Bean Dataset (Resized 224×224)* (`gpiosenka/coffee-bean-dataset-resized-224-x-224`)[cite: 10].
* **Preprocessing Target:** Dikonversi menjadi citra **32×32 piksel RGB (3 kanal warna, 8-bit per kanal, rentang 0–255)** dalam format `.bmp` untuk pengujian algoritma Python dan simulasi Verilog RTL[cite: 10].
* **Label Kelas:**
  * `0` (`green`) — Biji kopi mentah (*Green Bean*)[cite: 10]
  * `1` (`light`) — Sangrai terang (*Light Roast*)[cite: 10]
  * `2` (`medium`) — Sangrai sedang (*Medium Roast*)[cite: 10]
  * `3` (`dark`) — Sangrai gelap (*Dark Roast*)

---

## Struktur Isi Folder `dataset/`

1. **`train/`**: 1.200 citra latih asli (`224×224` RGB `.png`) yang terbagi merata ke dalam 4 folder kelas (`Green`, `Light`, `Medium`, `Dark` — masing-masing 300 citra)[cite: 10].
2. **`test/`**: 400 citra uji asli (`224×224` RGB `.png`) yang terbagi merata ke dalam 4 folder kelas (`Green`, `Light`, `Medium`, `Dark` — masing-masing 100 citra)[cite: 10].
3. **`test_32x32_bmp/`**: 400 citra uji hasil *resizing* ke format **32×32 piksel RGB (`.bmp`)** (`green_000.bmp` s.d. `dark_099.bmp`) beserta dua file memori heksadesimal untuk testbench Verilog[cite: 10]:
   * **`test_vectors.mem`** — Vektor piksel seluruh 400 citra uji (`400 × 3.072 = 1.228.800` byte RGB berurutan baris-kolom-kanal) yang dibaca langsung oleh `$readmemh` pada `tb_cnn_coffee.v`.
   * **`test_labels.mem`** — 400 label kelas referensi (`0`–`3`) sebagai pembanding akurasi otomatis pada simulasi ModelSim.
4. **`Coffee Bean.csv`**: Metadata indeks file, pembagian *train/test*, dan label kelas dari dataset asli Kaggle[cite: 10].
