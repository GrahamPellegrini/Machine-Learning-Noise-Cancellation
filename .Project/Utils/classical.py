"""
Author: Graham Pellegrini
Date: 2025-03-13
Project: UOM Final Year Project (FYP)

This util module is used within the denoising pipeline for evaluating classical
approaches against deep learning models. It includes the methods for Spectral Subtraction and Wiener Filtering. Other classical methods such as LMS, RLS, etc. are not implemented in this module as they require a clean or reference signal which is not in the scope of this project. The methods are used as baselines for comparison with deep learning-based denoising models. Their results are to be compared with the deep learning models, to further justify the use of deep learning models for speech enhancement.
"""
import numpy as np

import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as AF

from scipy.signal import wiener

from pesq import pesq
from pystoi import stoi

# === Denoising Methods ===
def denoise_classical(noisy_waveform, method):
    # Define STFT and inverse STFT
    spectrogram = T.Spectrogram(n_fft=1024, hop_length=256, power=None)
    inverse_spectrogram = T.InverseSpectrogram(n_fft=1024, hop_length=256)

    # Compute STFT
    noisy_waveform = F.pad(noisy_waveform, (0, max(0, 1024 - noisy_waveform.shape[-1])))
    noisy_stft = spectrogram(noisy_waveform)

    if method == "spectral_sub":
        # Use torchaudio's spectral subtraction
        noise_est = torch.mean(torch.abs(noisy_stft), dim=0, keepdim=True)
        denoised_stft = AF.spectral_subtract(noisy_stft, noise_est, alpha=1.5)

    elif method == "wiener":
        # Convert tensor STFT to NumPy for Wiener filtering
        magnitude = torch.abs(noisy_stft).cpu().numpy()
        phase = torch.angle(noisy_stft).cpu().numpy()

        denoised_magnitude = np.apply_along_axis(wiener, axis=-1, arr=magnitude)
        denoised_stft = torch.polar(torch.tensor(denoised_magnitude, dtype=torch.float32), torch.tensor(phase, dtype=torch.float32))

    else:
        raise ValueError(f"Invalid method: {method}. Supported methods: 'spectral_sub', 'wiener'")

    # Convert back to waveform
    denoised_waveform = inverse_spectrogram(denoised_stft)

    return denoised_waveform


# === Test Loader Denoising ===
def classical(test_loader, sr, method, pto=False):
    # Initialize lists for metrics and output file
    snr_list, mse_list, pesq_list, stoi_list, lsd_list = [], [], [], [], []
    classical_pth = "Output/txt/" + method + "_metrics.txt"

    # Iterate through test loader and compute metrics
    with torch.no_grad():
        for batch in test_loader:
            if pto:
                noisy_real, noisy_imag, clean_real, clean_imag, orig_lengths = batch
            else:
                noisy_real, noisy_imag, clean_real, clean_imag = batch
                orig_lengths = None
            
            # Convert the real and imaginary part to complex
            noisy_waveform = torch.complex(noisy_real, noisy_imag).abs()
            clean_waveform = torch.complex(clean_real, clean_imag).abs()
            
            # Use classical denoising method
            denoised_waveform = denoise_classical(noisy_waveform, method)
            
            # Ensure same length before computing metrics
            min_length = min(denoised_waveform.shape[-1], clean_waveform.shape[-1])
            denoised_waveform = denoised_waveform[..., :min_length]
            clean_waveform = clean_waveform[..., :min_length]
            
            # Compute SNR
            snr = F.signal_to_noise_ratio(denoised_waveform, clean_waveform).item()
            snr_list.append(snr)
            
            # Compute MSE
            mse = F.mse_loss(denoised_waveform, clean_waveform).item()
            mse_list.append(mse)

            # Compute LSD
            clean_log = torch.log10(torch.abs(clean_waveform) + 1e-6)
            denoised_log = torch.log10(torch.abs(denoised_waveform) + 1e-6)
            lsd = torch.mean(torch.sqrt(torch.mean((clean_log - denoised_log) ** 2, dim=-1))).item()
            lsd_list.append(lsd)
            
           # Resample to 16kHz for PESQ and STOI
            if sr != 16000:
                denoised_waveform = torchaudio.transforms.Resample(sr, 16000)(denoised_waveform)
                clean_waveform = torchaudio.transforms.Resample(sr, 16000)(clean_waveform)

            # Ensure correct shape for PESQ & STOI
            clean_wav = clean_waveform.squeeze().cpu().numpy().flatten()
            denoised_wav = denoised_waveform.squeeze().cpu().numpy().flatten()

            # Compute PESQ
            pesq_score = pesq(16000, clean_wav, denoised_wav, 'wb')
            pesq_list.append(pesq_score)

            # Compute STOI
            stoi_score = stoi(clean_wav, denoised_wav, 16000, extended=False)
            stoi_list.append(stoi_score)
    
    # Write metrics to file
    with open(classical_pth, "w") as f:
        f.write(f"↑SNR: {np.mean(snr_list):.4f}\n")
        f.write(f"↓MSE: {np.mean(mse_list):.6f}\n")
        f.write(f"↑PESQ: {np.mean(pesq_list):.4f}\n")
        f.write(f"↑STOI: {np.mean(stoi_list):.4f}\n")
        f.write(f"↓LSD: {np.mean(lsd_list):.6f}\n")

# === Single File Denoising ===
def single_classical(noisy_signal, orig_sr, sr, method, output_pth):

    # Convert to mono if needed
    if noisy_signal.shape[0] > 1:
        noisy_signal = torch.mean(noisy_signal, dim=0, keepdim=True)

    # Resample if needed
    if orig_sr != sr:
        noisy_signal = torchaudio.transforms.Resample(orig_sr, sr)(noisy_signal)
    
    # Normalize the waveform
    noisy_signal = noisy_signal / torch.max(torch.abs(noisy_signal) + 1e-6)

    # Use classical denoising method
    denoised_waveform = denoise_classical(noisy_signal, method)

    # Prevent clipping
    denoised_waveform = torch.clamp(denoised_waveform, -1, 1)

    # Save denoised waveform
    output_pth = "Output/wav/" + method + "_denoised.wav"
    torchaudio.save(output_pth, denoised_waveform, sr)

