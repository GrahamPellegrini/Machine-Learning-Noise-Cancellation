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
  Final Year Project submitted for the degree of <strong>B.Sc. (Hons.) in Computer Engineering</strong> at the <a href="https://www.um.edu.mt">UOM</a> under the supervision of <a href="https://www.um.edu.mt/profile/trevorspiteri">Dr. Trevor Spiteri</a>.
</p>

---

## 🔊 Project Overview

This project explores the application of machine learning (ML) techniques for speech enhancement, aiming to justify their superiority over classical methods in real-time denoising scenarios.

A modular and extensible Python pipeline was built to support:
- Variable-length waveform inputs using dynamic bucketing
- Real-time inference via spectrogram-based denoising
- Consistent metric-based evaluation for batch and single-file use
- Reproducible experiments via centralised configuration and logging

Three classical denoising methods — **Spectral Subtraction (SS)**, **Wiener Filtering (WF)**, and **MMSE-LSA** — were implemented to provide baselines.

These were compared against five ML models trained from scratch:
- **CNN**: A shallow baseline to verify the pipeline
- **CED / R-CED**: Encoder-decoder variants with and without residuals
- **UNet**: A deeper skip-connected model for better feature preservation
- **Conv-TasNet**: A time-domain network yielding top performance

All models were trained on **magnitude spectrograms**, with batch handling strategies (Static, Dynamic Bucketing, PTO) evaluated separately.

> The pipeline supports training from scratch, memory-efficient inference, and consistent evaluation. Laying the foundation for future experimentation with advanced architectures.

---

## 📊 Evaluation Metrics

- **SNR** – Signal-to-noise ratio
- **MSE** – Mean squared error
- **PESQ** – [Perceptual evaluation of speech quality](https://doi.org/10.1109/ICASSP.2001.941023)
- **STOI** – [Short-time objective intelligibility](https://doi.org/10.1109/TASL.2011.2114881)
- **LSD** – [Log-spectral distance](https://doi.org/10.1109/PACRIM.1993.407206)

The metrics were selected based on relevance in related literature and provide a balanced view of both numerical accuracy and perceptual quality in denoising performance.

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

---

## 🔬 Key Findings

- **Dynamic Bucketing** was selected as the preferred dataset handling method. It used **K-Means clustering** to assign samples into optimally sized buckets, improving training efficiency over **Static Bucketing** while avoiding the runtime penalties of **Padding-Truncation Output-Truncation (PTO)** during inference.
  
- **OOM mitigation techniques** — including **mixed-precision (FP16)**, **garbage collection**, and **gradient accumulation** — enabled training of deeper models on the university GPU cluster. Evaluation showed **no degradation** in model quality, and in some cases, **slight improvements** due to accumulated gradient stability.

- Among classical methods:
  - **Spectral Subtraction (SS)** achieved strong **numerical performance** (e.g., SNR), but performed poorly on perceptual metrics like PESQ and STOI.
  - **Wiener Filtering (WF)** and **MMSE-LSA** achieved better **perceptual quality**, but failed to match ML models in numerical fidelity.
  - Overall, no classical approach provided a **comprehensive improvement** across all evaluation dimensions.

- **Conv-TasNet** emerged as the **top-performing ML model**, achieving the best SNR, PESQ, and STOI. Originally developed for **speech separation**, its **temporal masking architecture** and **learned bottlenecks** translated effectively to the **spectrogram-based denoising** task used in this pipeline.

- The pipeline is built for **future extensibility** and supports:
  - Integration of **transformer** and **diffusion-based models**
  - Real-time inference with **beamforming** support
  - Exploration of **unseen noise generalisation** and more diverse datasets

---

## 🚧 Limitations & Future Work

Despite achieving strong numerical and perceptual performance, this project leaves several avenues for improvement:

### Limitations
- **Perceptual Ceiling**: While Conv-TasNet outperformed classical models, its **PESQ score (2.43)** remains well below the perceptual upper bound of 4.5.
- **Generalisation to Unseen Noise**: The models struggled with noise types not present during training. More diverse datasets are needed to improve real-world robustness.
- **Resource Constraints**: Due to limited GPU memory, batch sizes and model complexity were capped. Techniques like gradient accumulation and FP16 were essential but not ideal.

### Future Directions
- **Model Expansion**: Incorporating **transformer-based models** (e.g. ScaleFormer) or **diffusion-based architectures** could unlock further gains in intelligibility and naturalness.
- **Multi-Channel Input**: Extend the pipeline to support **beamforming** and microphone array input for spatial filtering in real-world deployments.
- **Self-Supervised Pretraining**: Introduce **SSL or reinforcement learning** strategies to improve generalisation with limited labelled data.
- **Real-Time Integration**: Adapt the inference system for **on-device deployment** in edge hardware like headphones or smartphones with Active Noise Cancellation (ANC) support.

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
│   ├── main.pdf         # 📄 Final Year Project Report
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

> Inference can denoise a single `.wav` or full batch denoice with metric evaluation as a `.txt`. 

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
