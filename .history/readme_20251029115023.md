# 🛰️ Pix2Pix Satellite → Map Translation

A deep learning project that converts **satellite images into map views** using the **Pix2Pix Generative Adversarial Network (GAN)**.

---

## 🚀 Overview
This project demonstrates how conditional GANs (Pix2Pix) can translate aerial (satellite) imagery into map-style representations. Trained on paired images of satellite views and corresponding map renderings, the model learns a one-to-one mapping.

---

## 📚 Table of Contents
- Overview
- Project structure
- Model architecture
- Training
- Requirements
- Results
- Demo animation
- Tools & frameworks
- Future improvements
- License
- Acknowledgements

---

## 🧩 Project structure
```
pix2pix-satellite-to-map/
│
├── data/                 # Small sample dataset
│   ├── train/            # 50 training images
│   └── test/             # 10 testing images
│
├── src/                  # Model, data loader & trainer
│   ├── load_data.py
│   ├── pix2pix_model.py
│   └── train.py
│
├── outputs/              # Generated examples + demo GIF
│   ├── epoch_001.png
│   ├── epoch_002.png
│   ├── ...
│   └── demo.gif
│
├── saved_models/         # (optional) saved weights
│
├── requirements.txt      # dependencies
└── README.md
```

---

## 🧠 Model Architecture
- **Generator:** U‑Net architecture (encoder–decoder with skip connections)  
- **Discriminator:** PatchGAN (classifies 70×70 image patches)  
- **Loss Functions:**  
  - Adversarial loss  
  - L1 reconstruction loss (for pixel-level accuracy)  

---

## 🏋️‍♂️ Training

To train on your full dataset:
```bash
python src/train.py
```

For a quick demo (on the smaller subset included):
```bash
python src/train.py --epochs 1
```

---

## 🧾 Requirements
Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🌍 Results
Below are examples of satellite → generated map → ground truth comparisons. Example outputs are saved to the outputs/ directory (per-epoch images and demo GIF).

Satellite Input | Generated Map | Ground Truth
--- | --- | ---
epoch_001.png | epoch_001_gen.png | epoch_001_gt.png
epoch_002.png | epoch_002_gen.png | epoch_002_gt.png
... | ... | ...

---

## 🎞️ Demo Animation
Below is a GIF showing training progress (rendered from outputs/demo.gif):

![Training progress demo](outputs/demo.gif)

> Note: On GitHub the relative path renders automatically if outputs/demo.gif is committed. For local preview, ensure the GIF exists at outputs/demo.gif or use an absolute URL.

---

## 🧰 Tools & Frameworks
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow
- ImageIO (for GIF generation)

---

## 📈 Future Improvements
- Integrate real-world GIS datasets  
- Experiment with attention-based U‑Net  
- Add evaluation metrics (FID, PSNR)

---

## 📜 License
This project is open-sourced under the MIT License.

---

## 💡 Acknowledgements
Dataset originally inspired by the Pix2Pix “Maps” dataset (Isola et al., 2017).