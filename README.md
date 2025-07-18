<h1 align="center">Machine Learning Noise Cancellation System</h1>

<p align="center">
  <a href="https://www.um.edu.mt/courses/studyunit/ICT3908">
    <img src="https://img.shields.io/badge/University%20of%20Malta-ICT3908-blue?style=for-the-badge&logo=python&logoColor=white" alt="ICT3908 Course">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Built%20with-PyTorch-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  </a>
  <a href="https://datashare.ed.ac.uk/handle/10283/2791">
    <img src="https://img.shields.io/badge/Dataset-NoisySpeechDB-yellow?style=for-the-badge&logo=data" alt="Noisy Speech DB">
  </a>
</p>

<p align="center">
  Final Year Project submitted for the degree of <strong>B.Sc. (Hons.) in Computer Engineering</strong> at the <a href="https://www.um.edu.mt">University of Malta</a> under the supervision of <a href="https://www.um.edu.mt/profile/trevorspiteri">Dr. Trevor Spiteri</a>. This project was awarded an <strong>A Grade</strong>.
</p>

---

## 🔊 Project Overview

This project implements a modular, extensible **machine learning-based speech denoising system** trained on magnitude spectrograms. It compares five deep learning architectures for real-time and batch noise suppression, using objective speech quality metrics to evaluate their performance.

All experiments are reproducible through a configurable Python pipeline using `main.py`, with hyperparameters and paths controlled via `config.py`. Model logic, dataset loading, and utility functions are modularised in the `Utils/` directory.

---

## 🧠 Model Architectures Tested

| Model       | ↑SNR (dB) | ↓MSE     | ↑PESQ | ↑STOI | ↓LSD (dB) | Time (s) |
|-------------|--------------|-------------|---------|---------|------------|----------|
| Baseline    |  2.28        | 0.005152    | 1.8451  | 0.8928  | 0.9042     | 56       |
| CNN         |  4.64        | 0.001344    | 1.7410  | 0.8073  | 0.7956     | 71       |
| CED         | 13.19        | 0.000161    | 1.6780  | 0.8386  | 0.7655     | 65       |
| R-CED       | 14.53        | 0.000117    | 2.0542  | 0.8677  | 0.6480     | 74       |
| UNet        | 16.99        | 0.000069    | 2.1384  | 0.8940  | 0.7076     | 87       |
| Conv-TasNet | 18.06        | 0.000063    | 2.4329  | 0.9112  | 0.6741     | 139      |

> 🏆 **Conv-TasNet** achieved the best overall performance across all metrics.

---

## 🗂️ Repository Structure

```
.Project/
├── main.py              # Entry point for training/evaluation
├── config.py            # Central config for datasets, models, training params
├── Utils/
│   ├── dataset.py       # Spectrogram conversion + augmentation
│   ├── denoise.py       # Inference utilities
│   ├── train.py         # Model training/validation logic
│   └── models.py        # Model architectures (CNN, CED, R-CED, UNet, Conv-TasNet)
├── Models/              # Saved model weights
├── Output/              # Evaluation outputs & audio files
├── Cache/               # Intermediate spectrograms and logs
├── extras/              # Test scripts and experimental helpers
└── main.pdf             # 📄 Final Year Project Report (Grade A)
```

---

## 📥 Dataset

The system uses the **Noisy Speech Database** provided by the University of Edinburgh:
- 🔗 [https://datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)
- License: Creative Commons Attribution 4.0 International

The dataset contains 30 hours of speech mixed with various real-world noise profiles. Preprocessing converts raw audio to **magnitude spectrograms** for supervised training.

---

## ⚙️ Usage

All functionality is controlled via `main.py` and the `config.py` file:

```bash
python main.py
```

To change:
- Dataset paths
- Training parameters
- Model selection

Edit `config.py` accordingly.

> 📁 Denoising can be performed on a single `.wav` or batched dataset using the `denoise.py` module.

---

## 📈 Evaluation Metrics

The following objective speech metrics were computed:
- **SNR**: Signal-to-noise ratio
- **MSE**: Mean squared error
- **PESQ**: Perceptual evaluation of speech quality
- **STOI**: Short-time objective intelligibility
- **LSD**: Log-spectral distance

Both real-time inference and batch metric evaluation are supported.

---

## 🚫 Omitted Files

Certain SLURM `.sh` scripts and internal university paths (e.g. `/opt/users/gpel0001/`) were intentionally excluded from this repository for privacy and security. These scripts were used to run long-term jobs on the University of Malta's GPU cluster.

---

## 📘 Report

📄 The full final year project report is included as [`main.pdf`](main.pdf).  
It details the model architectures, evaluation methodology, and experimental results, and received an **A Grade** from the University of Malta.

---

## 👨🏻‍💻 Author

**Graham Pellegrini**  
B.Sc. (Hons.) Computer Engineering  
University of Malta  
GitHub: [@GrahamPellegrini](https://github.com/GrahamPellegrini)
