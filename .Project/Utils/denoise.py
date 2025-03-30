"""
Author: Graham Pellegrini
Date: 2025-03-13
Project: UOM Final Year Project (FYP)

The main denoise script for the project. This script is used to denoise audio files using the trained models or the classical methods. It follows the training process to evalute the models on the test set. The script can be used to denoise single audio files or the entire test set. The metrics are computed and saved to a file for evaluation. The script is used to compare the performance of the deep learning models against the classical methods. The script is also used to evaluate the models on the test set and save the metrics for comparison. This is the main script for the denoising pipeline.
"""

import os
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

import torchaudio

from pesq import pesq
from pystoi import stoi

# === Model Denoising ===
def denoise(device, model, model_pth, test_loader, sr, n_fft, hop_length, metric_pth, pto=False):

    # Load the model
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.to(device)
    model.eval()

    # Initialize metrics
    snr_list, mse_list, pesq_list, stoi_list, lsd_list = [], [], [], [], []

    # Define transforms
    inverse_spectrogram = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(device)

    # Itterate through test set
    with torch.no_grad():
        for batch in test_loader:
            if pto:
                noisy_real, noisy_imag, clean_real, clean_imag, orig_lengths = batch
            else:
                noisy_real, noisy_imag, clean_real, clean_imag = batch
                orig_lengths = None

            # Move to device
            noisy_real, noisy_imag = noisy_real.to(device), noisy_imag.to(device)
            clean_real, clean_imag = clean_real.to(device), clean_imag.to(device)

            # Run through model
            denoised_real, denoised_imag = model(noisy_real, noisy_imag)
            denoised_real, denoised_imag = denoised_real.squeeze(1), denoised_imag.squeeze(1)

            # Reconstruct complex spectrogram
            denoised_spec = torch.complex(denoised_real, denoised_imag)
            clean_spec = torch.complex(clean_real, clean_imag)

            # Convert back to waveform
            denoised_waveform = inverse_spectrogram(denoised_spec).cpu().squeeze()
            clean_waveform = inverse_spectrogram(clean_spec).cpu().squeeze()

            # PTO managing paddinf back to original length
            if pto:
                max_length = max(orig_lengths).item()  # Find max original length

                def pad_to_max(tensor, length):
                    if tensor.shape[-1] < length:
                        return F.pad(tensor, (0, length - tensor.shape[-1]))
                    # Trim excess if needed
                    return tensor[:length]  

                denoised_waveform = torch.stack([pad_to_max(denoised_waveform[i], max_length) for i in range(denoised_waveform.shape[0])])
                clean_waveform = torch.stack([pad_to_max(clean_waveform[i], max_length) for i in range(clean_waveform.shape[0])])

            # Compute SNR
            snr = F.signal_to_noise_ratio(denoised_waveform, clean_waveform).item()
            snr_list.append(snr)
            
            # Compute MSE
            mse = F.mse_loss(denoised_waveform, clean_waveform).item()
            mse_list.append(mse)

            # Compute LSD
            clean_log = torch.log10(torch.abs(clean_spec) + 1e-6)
            denoised_log = torch.log10(torch.abs(denoised_spec) + 1e-6)
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

    # Save computed metrics
    with open(metric_pth, "w") as f:
        f.write(f"↑SNR: {np.mean(snr_list):.4f}\n")
        f.write(f"↓MSE: {np.mean(mse_list):.6f}\n")
        f.write(f"↑PESQ: {np.mean(pesq_list):.4f}\n")
        f.write(f"↑STOI: {np.mean(stoi_list):.4f}\n")
        f.write(f"↓LSD: {np.mean(lsd_list):.4f}\n")

    print(f"✔ Metrics saved to {metric_pth}")

# === Model Single Denoising ===
def single_denoise(device, model, model_pth, noisy_pth, output_pth, sr, nfft, hop_length):

    # Ensure the model file exists
    if not os.path.exists(model_pth):
        raise FileNotFoundError(f"Model file not found: {model_pth}")
    
    if not os.path.exists(noisy_pth):
        raise FileNotFoundError(f"Noisy audio file not found: {noisy_pth}")

    # Load model weights
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.to(device)
    model.eval()

    # Define spectrogram transforms
    spectrogram = T.Spectrogram(n_fft=nfft, hop_length=hop_length, power=None).to(device)
    inverse_spectrogram = T.InverseSpectrogram(n_fft=nfft, hop_length=hop_length).to(device)

    # Load noisy audio
    noisy_signal, orig_sr = torchaudio.load(noisy_pth)

    # Convert stereo to mono
    if noisy_signal.shape[0] > 1:
        noisy_signal = noisy_signal.mean(dim=0, keepdim=True)

    # Resample if necessary
    if orig_sr != sr:
        noisy_signal = T.Resample(orig_sr, sr)(noisy_signal)

    # Compute spectrogram (Complex-valued)
    noisy_spec = spectrogram(noisy_signal.to(device))

    # Normalize the spectrogram
    noisy_spec = (noisy_spec - noisy_spec.mean()) / (noisy_spec.std() + 1e-6)

    # Separate real and imaginary components
    noisy_real, noisy_imag = noisy_spec.real, noisy_spec.imag

    # Process through model
    with torch.no_grad():
        denoised_real, denoised_imag = model(noisy_real.unsqueeze(1), noisy_imag.unsqueeze(1))  # Add channel dim

    # Remove extra channel dim
    denoised_real, denoised_imag = denoised_real.squeeze(1), denoised_imag.squeeze(1)

    # Reconstruct complex spectrogram
    denoised_spec = torch.complex(denoised_real, denoised_imag)

    # Convert back to waveform
    denoised_waveform = inverse_spectrogram(denoised_spec)

    # Prevent potential clipping issues
    denoised_waveform = torch.clamp(denoised_waveform, min=-1.0, max=1.0)

    # Save the denoised output
    torchaudio.save(output_pth, denoised_waveform.cpu(), sr)
    print(f"✅ Denoised audio saved to: {output_pth}") 