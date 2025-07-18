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

This project explores the application of machine learning (ML) techniques for speech enhancement, aiming to justify their superiority over classical methods in real-time denoising scenarios.

A modular and extensible pipeline was developed to support:
- Variable-length audio inputs
- Real-time inference
- Efficient batch evaluation
- Adjustable models, datasets, and processing strategies

Classical denoising baselines such as **Spectral Subtraction**, **Wiener Filtering**, and **MMSE-LSA** were implemented for comparison. These were benchmarked against five trained ML models: **CNN**, **CED**, **R-CED**, **UNet**, and **Conv-TasNet** — using magnitude spectrograms as the input representation.

Key contributions include:
- A reusable Python pipeline with full model configuration via `config.py`
- Exploration of dataset handling strategies (Static/Dynamic Bucketing, PTO)
- Memory optimization using FP16 precision and gradient accumulation for large models

> 🏆 **Conv-TasNet** outperformed all methods across objective and perceptual metrics.

---

## 🧠 Model Architectures Tested

| Model       | ↑SNR (dB) | ↓MSE     | ↑PESQ | ↑STOI | ↓LSD (dB) | Time (s) |
|-------------|-----------|----------|--------|--------|-----------|----------|
| Baseline    |  2.28     | 0.005152 | 1.8451 | 0.8928 | 0.9042    | 56       |
| CNN         |  4.64     | 0.001344 | 1.7410 | 0.8073 | 0.7956    | 71       |
| CED         | 13.19     | 0.000161 | 1.6780 | 0.8386 | 0.7655    | 65       |
| R-CED       | 14.53     | 0.000117 | 2.0542 | 0.8677 | 0.6480    | 74       |
| UNet        | 16.99     | 0.000069 | 2.1384 | 0.8940 | 0.7076    | 87       |
| Conv-TasNet | 18.06     | 0.000063 | 2.4329 | 0.9112 | 0.6741    | 139      |

> 🔍 ML models outperformed all classical denoising methods in both numerical and perceptual evaluation.

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

The system uses the **Noisy Speech Database** from the University of Edinburgh:
- 🔗 [https://datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)
- License: Creative Commons Attribution 4.0 International

> Audio preprocessing is done through **magnitude spectrograms**, which standardizes input for all model types.

---

## ⚙️ Usage

All functionality is controlled via `main.py` and `config.py`:

```bash
python main.py
```

To modify:
- Dataset or output locations
- Model selection or loss functions
- Batch sizes, precision, or training strategy

Edit `config.py` accordingly.

> Denoising can be performed on a single `.wav` file or entire batches. Output can be written to waveform or evaluated with metrics.

---

## 📈 Evaluation Metrics

- **SNR** – Signal-to-noise ratio
- **MSE** – Mean squared error
- **PESQ** – Perceptual evaluation of speech quality
- **STOI** – Short-time objective intelligibility
- **LSD** – Log-spectral distance

Each model was evaluated using a consistent real-time batch pipeline. All metrics reported in this README are averaged across the full test set.

---

## 🔬 Notable Engineering Findings

- ✅ **Dynamic Bucketing** was the most efficient dataset handling strategy, balancing memory and performance.
- ⚙️ **OOM Handling Techniques** like FP16, garbage collection, and gradient accumulation allowed training of large models like UNet and Conv-TasNet without degradation.
- 📊 The perceptual quality of **ML models** exceeded classical methods — especially in PESQ and STOI.
- 🚀 The modular pipeline allows future researchers to test custom datasets and model variants with minimal effort.

---

## 🚫 Omitted Files

Some internal SLURM `.sh` scripts, user emails, and university path references have been **excluded** from this public version.

If needed, portable `.sh` templates can be found in the `extras/` directory to reproduce training or notebook jobs on any SLURM-compatible cluster.

---

## 📘 Final Report

📄 The full dissertation is included here: [`main.pdf`](main.pdf)

It contains methodology, architecture diagrams, evaluation metrics, and ablation study results. This project was awarded an **A Grade**.

---

## 👨🏻‍💻 Author

**Graham Pellegrini**  
B.Sc. (Hons.) Computer Engineering  
University of Malta  
GitHub: [@GrahamPellegrini](https://github.com/GrahamPellegrini)
