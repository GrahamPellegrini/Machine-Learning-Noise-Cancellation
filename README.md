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
  Final Year Project submitted for the degree of <strong>B.Sc. (Hons.) in Computer Engineering</strong> at the <a href="https://www.um.edu.mt">University of Malta</a> under the supervision of <a href="https://www.um.edu.mt/profile/trevorspiteri">Dr. Trevor Spiteri</a>.
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
- Evaluation of three dataset handling strategies (Static Bucketing, Dynamic Bucketing, PTO)
- Memory optimization using FP16 precision, garbage collection, and gradient accumulation
- Comparative study of ML vs. classical denoising methods

> 📌 While ML models clearly outperformed classical methods numerically, **perceptual scores such as PESQ remained modest**, highlighting a potential frontier for future improvement.

---

## 🧠 Model & Method Performance

<table>
<thead>
<tr><th>Method</th><th>&uarr;SNR (dB)</th><th>&darr;MSE</th><th>&uarr;PESQ</th><th>&uarr;STOI</th><th>&darr;LSD (dB)</th><th>Time (s)</th></tr>
</thead>
<tbody>
<tr style="background-color:#f9f9f9"><td>Baseline</td><td>-2.28</td><td>0.005152</td><td>1.8451</td><td>0.8928</td><td>0.9042</td><td>56</td></tr>
<tr style="background-color:#eef7ff"><td>SS</td><td>3.09</td><td>0.001525</td><td>1.4535</td><td>0.8457</td><td>0.7671</td><td>61</td></tr>
<tr style="background-color:#eef7ff"><td>WF</td><td>0.46</td><td>0.002875</td><td>2.0639</td><td>0.8889</td><td>0.7535</td><td>73</td></tr>
<tr style="background-color:#eef7ff"><td>MMSE-LSA</td><td>-0.86</td><td>0.003726</td><td>2.0238</td><td>0.8943</td><td>0.7971</td><td>83</td></tr>
<tr style="background-color:#fff6ea"><td>CNN</td><td>4.64</td><td>0.001344</td><td>1.7410</td><td>0.8073</td><td>0.7956</td><td>71</td></tr>
<tr style="background-color:#fff6ea"><td>CED</td><td>13.19</td><td>0.000161</td><td>1.6780</td><td>0.8386</td><td>0.7655</td><td>65</td></tr>
<tr style="background-color:#fff6ea"><td>R-CED</td><td>14.53</td><td>0.000117</td><td>2.0542</td><td>0.8677</td><td>0.6480</td><td>74</td></tr>
<tr style="background-color:#fff6ea"><td>UNet</td><td>16.99</td><td>0.000069</td><td>2.1384</td><td>0.8940</td><td>0.7076</td><td>87</td></tr>
<tr style="background-color:#fff6ea"><td>Conv-TasNet</td><td>18.06</td><td>0.000063</td><td>2.4329</td><td>0.9112</td><td>0.6741</td><td>139</td></tr>
</tbody>
</table>

> 🔍 While Conv-TasNet achieved strong metrics across the board, PESQ scores still fall well short of the theoretical max (4.5), suggesting room for perceptual enhancement.

---

## 🗂️ Repository Structure

```bash
.Project/
├── main.py              # Entry point for training/evaluation
├── config.py            # Central config for datasets, models, training params
├── Utils/
│   ├── dataset.py       # Spectrogram conversion + augmentation
│   ├── denoise.py       # Inference utilities
│   ├── train.py         # Model training/validation logic
│   └── models.py        # Model architectures (CNN, CED, R-CED, UNet, Conv-TasNet)
├── Models/              # Saved model weights by experiment
│   ├── 25/
│   ├── dataset/
│   └── oom/
├── Output/              # Denoising outputs (wav, txt, png)
│   ├── 25/
│   ├── dataset/
│   ├── oom/
│   ├── png/
│   ├── txt/
│   └── wav/
├── Cache/               # Cached spectrograms and length logs
│   ├── dynamic/
│   ├── static/
│   └── pto/
├── ssh/                 # SLURM-compatible job scripts
│   ├── main.sh
│   ├── latex.sh
│   └── notebook.sh
├── Template/            # Report LaTeX source
│   ├── main.pdf         # Final Year Project Report
│   ├── main.tex
│   ├── build/
│   ├── content/
│   └── references.bib
└── .gitignore
```

---

## 📥 Dataset

The system uses the **Noisy Speech Database** from the University of Edinburgh:
- 🔗 [https://datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)
- License: Creative Commons Attribution 4.0 International

> Audio is converted to **magnitude spectrograms** for all training, validation, and inference steps.

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

> Inference can denoise a single `.wav` or full batch with metric evaluation.

---

## 📊 Evaluation Metrics

- **SNR** – Signal-to-noise ratio
- **MSE** – Mean squared error
- **PESQ** – Perceptual evaluation of speech quality 
- **STOI** – Short-time objective intelligibility
- **LSD** – Log-spectral distance

All metrics reflect **average batch performance** across full dataset splits.

---

## 🔬 Key Findings

- ✅ **Dynamic Bucketing** enabled optimal training speed for variable-length inputs.
- 💡 **OOM techniques** (FP16, GC, accumulation) enabled deep model training with no metric drop.
- 📉 Classical models (e.g., **WF**) performed decently on perceptual metrics, but fell far behind in numerical ones.
- 🧠 **Conv-TasNet** emerged best overall, but still left headroom in PESQ and generalisation.
- ⚒️ The pipeline supports:
  - Transformer and diffusion model integration
  - Real-time audio and beamforming extensions
  - Generalisation to unseen noise domains

---

## 📘 Final Report

📄 Read the full dissertation here: [`main.pdf`](main.pdf)

Includes methodology, system design, ablation studies, and model evaluation.

> 🎓 This project received an **A Grade** in the B.Sc. (Hons.) Computer Engineering programme.

---

## 👨‍💻 Author

**Graham Pellegrini**  
B.Sc. (Hons.) Computer Engineering  
University of Malta  
GitHub: [@GrahamPellegrini](https://github.com/GrahamPellegrini)
