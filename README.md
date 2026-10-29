# Sistem Pengenalan & Konversi Aksara Bima (CRNN + Translation + Rendering)

Repositori ini memuat implementasi lengkap dari sistem terpadu Aksara Bima (Mbojo), yang terdiri dari dua sistem utama:
1. **Sistem A (Vision / OCR):** Mengubah citra foto aksara Bima menjadi teks Latin menggunakan model **CRNN (CNN + BiLSTM + CTC Loss)**.
2. **Sistem B (Linguistik & Ortografi):** Menerjemahkan kata Indonesia ke Bahasa Bima Latin, memenggal suku katanya ke 90 kelas kanonis, dan merender menjadi **citra Aksara Bima utuh**.

---

## Struktur Berkas Proyek

```
c:\xampp\htdocs\archive (1)\
├── Dataset_class/                 # 90 folder suku kata asli (25 citra/kelas, total 2.250 gambar)
├── 01_generate_synthetic_words.py # Generator citra kata sintetis untuk pelatihan CRNN
├── dataset.py                     # PyTorch Dataset & Collate Function untuk CTC
├── model.py                       # Arsitektur CRNN (7-layer CNN + 2-layer BiLSTM + Linear)
├── decoder.py                     # Greedy Decoder, Prefix Beam Search, & Levenshtein CER
├── train.py                       # Script pelatihan CRNN & evaluasi validasi CER
├── translator.py                  # Leksikon kamus kata Indonesia -> Bahasa Bima
├── syllabifier.py                 # Mesin pemenggal suku kata fonotaktik Bima (90 kelas)
├── renderer.py                    # Mesin perakit & perender glyph aksara ke citra utuh
├── app.py                         # Antarmuka web interaktif berbasis Streamlit
├── requirements.txt               # Daftar pustaka Python yang dibutuhkan
└── README.md                      # Dokumentasi teknis proyek
```

---

## Petunjuk Penggunaan Cepat

### 1. Instalasi Pustaka
```bash
pip install -r requirements.txt
```

### 2. Membangun Dataset Kata Sintetis
Karena dataset awal berupa suku kata tunggal terisolasi, jalankan script berikut untuk membuat 3.000 citra kata/frasa sintetis yang realistis:
```bash
python 01_generate_synthetic_words.py
```
*Output: Folder `data_crnn/` berisi gambar kata sintetis beserta `train.csv`, `val.csv`, dan `test.csv`.*

### 3. Melatih Model CRNN
Latih model CRNN untuk mengenali urutan aksara:
```bash
python train.py
```
*Output: `best_crnn.pt` (bobot model terbaik) dan `charset.json` (indeks karakter CTC).*

### 4. Menjalankan Aplikasi Web
Jalankan dashboard antarmuka visual:
```bash
streamlit run app.py
```
Buka browser pada alamat `http://localhost:8501`.

