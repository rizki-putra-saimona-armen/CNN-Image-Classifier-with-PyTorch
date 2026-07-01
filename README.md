# CNN Image Classifier with PyTorch

Proyek klasifikasi citra (image classification) menggunakan **Convolutional Neural Network (CNN)** yang dibangun dari nol dengan **PyTorch**, dibantu library [`jcopdl`](https://pypi.org/project/jcopdl/) untuk mempermudah proses training, callback, dan checkpointing.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Deskripsi

Notebook ini mendemonstrasikan alur lengkap membangun model **CNN** untuk klasifikasi citra biner (2 kelas), meliputi:

- Preprocessing & augmentasi data gambar
- Desain arsitektur CNN modular (feature extractor + classifier)
- Training loop dengan early stopping & checkpointing otomatis
- Visualisasi hasil prediksi

Cocok digunakan sebagai referensi belajar **Deep Learning berbasis PyTorch**, khususnya untuk kasus klasifikasi gambar sederhana.

---

##  Struktur Proyek

```
.
├── Part_3_-CNN_with_Pytorch.ipynb   # Notebook utama
├── data/
│   ├── train/                       # Dataset training (per-folder = per kelas)
│   └── test/                        # Dataset testing (per-folder = per kelas)
├── model/                           # Output checkpoint model (otomatis dibuat)
└── README.md
```

> Dataset menggunakan format `ImageFolder` dari `torchvision`, artinya setiap subfolder di dalam `data/train/` dan `data/test/` merepresentasikan satu kelas/label.

---

##  Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

2. Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib tqdm jupyter
pip install jcopdl luwiji
```

---

##  Arsitektur Model

Model CNN dibangun secara modular menggunakan `conv_block` dan `linear_block` dari `jcopdl`:

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor
        self.conv = nn.Sequential(
            conv_block(3,   8),
            conv_block(8,  16),
            conv_block(16, 32),
            conv_block(32, 64),
            nn.Flatten()
        )

        # Classifier
        self.fc = nn.Sequential(
            linear_block(1024, 256, dropout=0.1),
            linear_block(256, 2, activation="lsoftmax")
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
```

**Ringkasan arsitektur:**

| Layer            | Detail                              |
|-------------------|--------------------------------------|
| Conv Block 1–4    | Ekstraksi fitur bertingkat (3→8→16→32→64 channel) |
| Flatten           | Meratakan feature map menjadi vektor |
| Fully Connected 1 | 1024 → 256, dengan dropout 0.1       |
| Fully Connected 2 | 256 → 2, aktivasi log-softmax        |

---

## 🖼️ Preprocessing & Augmentasi Data

| Data Latih (Train)                     | Data Uji (Test)                |
|-----------------------------------------|----------------------------------|
| Random Rotation (±15°)                  | Resize ke 70px                  |
| Random Resized Crop (64px, scale 0.8–1.0)| Center Crop 64px               |
| Random Horizontal Flip                  | -                                |
| ToTensor                                | ToTensor                        |

---

##  Training

Model dilatih menggunakan:
- **Loss function:** `NLLLoss` (Negative Log Likelihood)
- **Optimizer:** `AdamW` (learning rate 0.001)
- **Batch size:** 64
- **Early stopping** otomatis berdasarkan `test_score` via `jcopdl.callback`

Jalankan seluruh sel notebook secara berurutan, atau langsung via terminal:

```bash
jupyter notebook Part_3_-CNN_with_Pytorch.ipynb
```

Selama training, Anda akan melihat:
- Progress bar per epoch (train & test)
- Plot cost & accuracy secara real-time
- Checkpoint model otomatis tersimpan di folder `model/`

---

##  Hasil & Evaluasi

Setelah training selesai, notebook akan menampilkan grid visualisasi prediksi model pada data uji, lengkap dengan label warna:

-  **Hijau** → prediksi benar
-  **Merah** → prediksi salah

---

##  Tech Stack

- **PyTorch** & **Torchvision** — Deep learning framework
- **jcopdl** — Callback, layer builder, dan config management
- **NumPy** & **Matplotlib** — Komputasi numerik & visualisasi
- **tqdm** — Progress bar training

---

##  Lisensi

Proyek ini menggunakan lisensi **MIT** — bebas digunakan, dimodifikasi, dan didistribusikan ulang untuk keperluan pembelajaran maupun pengembangan lebih lanjut.

---

##  Kontribusi

Kontribusi, saran, dan diskusi sangat terbuka! Silakan buat *issue* atau *pull request* jika ingin membantu mengembangkan proyek ini.

---

<p align="center">Made with  using PyTorch</p>
