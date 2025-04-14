import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

import scipy.signal as SG

from torchmetrics.audio import SignalNoiseRatio
from torchmetrics import MeanSquaredError
from pesq import pesq
from pystoi import stoi


def compute_metrics(denoised, clean, sr):
    # Ensure contiguous tensors
    denoised = denoised.contiguous()
    clean = clean.contiguous()

    device = denoised.device

    # Compute SNR and MSE (torchmetrics on correct device)
    snr = SignalNoiseRatio().to(device)(denoised, clean).item()
    mse = MeanSquaredError().to(device)(denoised, clean).item()

    # Compute LSD
    clean_log = torch.log10(torch.abs(clean) + 1e-6)
    denoised_log = torch.log10(torch.abs(denoised) + 1e-6)
    lsd = torch.mean(torch.sqrt(torch.mean((clean_log - denoised_log) ** 2, dim=-1))).item()

    # Resample for PESQ and STOI if needed
    if sr != 16000:
        resample = T.Resample(orig_freq=sr, new_freq=16000).to(device)
        denoised = resample(denoised)
        clean = resample(clean)

    # Convert to NumPy
    clean_np = clean.squeeze().cpu().numpy().flatten()
    denoised_np = denoised.squeeze().cpu().numpy().flatten()

    # Clip to expected [-1, 1] range
    clean_np = np.clip(clean_np, -1.0, 1.0)
    denoised_np = np.clip(denoised_np, -1.0, 1.0)

    # Validate before PESQ/STOI to avoid NaN/Inf issues
    if (
        clean_np.shape[0] < 160 or denoised_np.shape[0] < 160 or
        np.any(np.isnan(clean_np)) or np.any(np.isnan(denoised_np)) or
        np.any(np.isinf(clean_np)) or np.any(np.isinf(denoised_np)) or
        np.allclose(clean_np, 0, atol=1e-5) or np.allclose(denoised_np, 0, atol=1e-5)
    ):
        pesq_score = 0.0
        stoi_score = 0.0
    else:
        pesq_score = pesq(16000, clean_np, denoised_np, 'wb')
        stoi_score = stoi(clean_np, denoised_np, 16000, extended=False)

    return snr, mse, lsd, pesq_score, stoi_score

def classical(noisy_waveform, method):
    """
    Apply classical denoising methods to a batch of waveforms.
    Args:
        noisy_waveform (Tensor): Shape (batch, 1, time)
        method (str): One of ['spectral_sub', 'wiener', 'quad_filter']
    Returns:
        Tensor: Denoised waveforms, shape (batch, time)
    """
    if noisy_waveform.dim() != 3 or noisy_waveform.shape[1] != 1:
        raise ValueError(f"[!] Expected input shape (batch, 1, time), got {noisy_waveform.shape}")

    device = noisy_waveform.device
    n_fft = 1024
    hop_length = 256
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None).to(device)
    inverse_spectrogram = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(device)

    denoised_batch = []

    for i in range(noisy_waveform.shape[0]):
        x = noisy_waveform[i, 0]  # (time,)
        orig_len = x.shape[-1]
        x = F.pad(x, (0, max(0, n_fft - orig_len)))  # ensure enough for one FFT frame

        stft = spectrogram(x)  # (freq, time), complex
        magnitude, phase = torch.abs(stft), torch.angle(stft)

        if method == "spectral_sub":
            noise_est = magnitude.mean(dim=1, keepdim=True)
            den_mag = torch.clamp(magnitude - noise_est, min=0.0)
            den_stft = torch.polar(den_mag, phase)

        elif method == "wiener":
            noise_psd = torch.min(magnitude**2, dim=1, keepdim=True).values
            signal_psd = torch.clamp(magnitude**2 - noise_psd, min=1e-6)
            gain = signal_psd / (signal_psd + noise_psd)

            alpha = 0.85
            smoothed_gain = gain.clone()
            for t in range(1, gain.shape[1]):
                smoothed_gain[:, t] = alpha * smoothed_gain[:, t - 1] + (1 - alpha) * gain[:, t]

            den_mag = smoothed_gain * magnitude
            den_stft = torch.polar(den_mag, phase)
        
        elif method == "mmse_lsa":
            beta = 0.98  # smoothing for a priori SNR
            alpha = 0.85 # smoothing for noise PSD

            # Estimate noise PSD from the first 6 frames (assumed silence)
            noise_est = magnitude[:, :6].mean(dim=1, keepdim=True) ** 2
            noise_psd = noise_est.repeat(1, magnitude.shape[1])

            gamma = torch.clamp((magnitude**2) / (noise_psd + 1e-6), min=1e-6, max=1000)
            xi = torch.zeros_like(gamma)
            gain = torch.ones_like(gamma)  # <- added to avoid UnboundLocalError

            xi[:, 0] = 1.0  # initial a priori SNR

            for t in range(1, gamma.shape[1]):
                xi[:, t] = beta * (gain[:, t - 1] ** 2) * gamma[:, t - 1] + (1 - beta) * torch.clamp(gamma[:, t] - 1, min=0.0)

            nu = xi * gamma / (1 + xi)
            # Use approximation for exponential integral
            exp_int = torch.exp(-nu) * torch.log1p(nu + 1e-6)

            gain = (xi / (1 + xi)) * torch.exp(0.5 * exp_int)
            gain = torch.clamp(gain, min=1e-5, max=1.0)

            den_mag = gain * magnitude
            den_stft = torch.polar(den_mag, phase)

            # Note mention we tried to implement the preceeding wavelet transorm like in the paper but the increased complexity did not yield any improvement

        else:
            raise ValueError(f"[!] Unknown classical method: {method}")

        den_waveform = inverse_spectrogram(den_stft)
        den_waveform = torch.clamp(den_waveform.squeeze(0), -1.0, 1.0)
        denoised_batch.append(den_waveform[:orig_len])

    return torch.stack(denoised_batch)  # (batch, time)



def batch_denoise(device, model, model_pth, classical_method, test_loader, sr, n_fft, hop_length, metric_pth, pto=False):
    snr_list, mse_list, lsd_list, pesq_list, stoi_list = [], [], [], [], []

    # === Classical Denoising ===
    if classical_method is not None:
        for batch in test_loader:
            if pto:
                noisy_real, noisy_imag, clean_real, clean_imag, orig_lengths = batch
            else:
                noisy_real, noisy_imag, clean_real, clean_imag = batch

            # Move tensors to device (optional but recommended for GPU consistency)
            noisy_real, noisy_imag = noisy_real.to(device), noisy_imag.to(device)
            clean_real, clean_imag = clean_real.to(device), clean_imag.to(device)

            # Instantiate inverse STFT on correct device
            inverse_spec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(device)

            # Convert spectrogram back to waveform
            spec = torch.complex(noisy_real.squeeze(1), noisy_imag.squeeze(1))
            clean_spec = torch.complex(clean_real.squeeze(1), clean_imag.squeeze(1))
            noisy_waveform = inverse_spec(spec)
            clean_waveform = inverse_spec(clean_spec)

            # Normalize to avoid extreme values
            noisy_waveform = noisy_waveform / (torch.max(torch.abs(noisy_waveform)) + 1e-6)

            # Denoise using classical method
            denoised_waveform = classical(noisy_waveform.unsqueeze(1), classical_method)

            # Truncate or pad to match lengths
            min_len = min(denoised_waveform.shape[-1], clean_waveform.shape[-1])
            denoised_waveform = denoised_waveform[..., :min_len]
            clean_waveform = clean_waveform[..., :min_len]

            # Compute metrics
            snr, mse, lsd, pesq_s, stoi_s = compute_metrics(denoised_waveform, clean_waveform, sr)
            snr_list.append(snr); mse_list.append(mse); lsd_list.append(lsd)
            pesq_list.append(pesq_s); stoi_list.append(stoi_s)

    # === Model-based Denoising ===
    else:
        model.load_state_dict(torch.load(model_pth, map_location=device))
        model.to(device).eval()
        inverse_spec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(device)

        with torch.no_grad():
            for batch in test_loader:
                if pto:
                    noisy_real, noisy_imag, clean_real, clean_imag, orig_lengths = batch
                else:
                    noisy_real, noisy_imag, clean_real, clean_imag = batch

                # Move to device
                noisy_real, noisy_imag = noisy_real.to(device), noisy_imag.to(device)
                clean_real, clean_imag = clean_real.to(device), clean_imag.to(device)

                # Denoise
                den_real, den_imag = model(noisy_real, noisy_imag)
                den_spec = torch.complex(den_real.squeeze(1), den_imag.squeeze(1))
                clean_spec = torch.complex(clean_real.squeeze(1), clean_imag.squeeze(1))

                den_waveform = inverse_spec(den_spec).cpu()
                clean_waveform = inverse_spec(clean_spec).cpu()

                if pto:
                    max_len = max(orig_lengths).item()
                    pad_fn = lambda x: F.pad(x, (0, max_len - x.shape[-1])) if x.shape[-1] < max_len else x[..., :max_len]
                    den_waveform = torch.stack([pad_fn(w) for w in den_waveform])
                    clean_waveform = torch.stack([pad_fn(w) for w in clean_waveform])

                # Metrics
                snr, mse, lsd, pesq_s, stoi_s = compute_metrics(den_waveform, clean_waveform, sr)
                snr_list.append(snr); mse_list.append(mse); lsd_list.append(lsd)
                pesq_list.append(pesq_s); stoi_list.append(stoi_s)

    # Save results
    with open(metric_pth, "w") as f:
        f.write(f"↑SNR: {np.mean(snr_list):.4f}\n")
        f.write(f"↓MSE: {np.mean(mse_list):.6f}\n")
        f.write(f"↓LSD: {np.mean(lsd_list):.6f}\n")
        f.write(f"↑PESQ: {np.mean(pesq_list):.4f}\n")
        f.write(f"↑STOI: {np.mean(stoi_list):.4f}\n")

    print(f"✔ Metrics saved to {metric_pth}")

# === Shared Single File Denoising ===
def single_denoise(device, model, model_pth, noisy_pth, output_pth, sr, nfft, hop_length, classical_method=None):
    if classical_method is not None:
        # === Classical Single Denoise ===
        if not os.path.exists(noisy_pth): raise FileNotFoundError(f"Missing audio: {noisy_pth}")

        waveform, orig_sr = torchaudio.load(noisy_pth)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            waveform = T.Resample(orig_sr, sr)(waveform)
            
        # Normalize to avoid extreme values
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)

        den_waveform = classical(waveform, classical_method)
        waveform = den_waveform.cpu()
    else:
        # === Model-based Single Denoise ===
        if not os.path.exists(model_pth): raise FileNotFoundError(f"Missing model: {model_pth}")
        if not os.path.exists(noisy_pth): raise FileNotFoundError(f"Missing audio: {noisy_pth}")

        model.load_state_dict(torch.load(model_pth, map_location=device))
        model.to(device).eval()

        spec_fn = T.Spectrogram(n_fft=nfft, hop_length=hop_length, power=None).to(device)
        inv_fn = T.InverseSpectrogram(n_fft=nfft, hop_length=hop_length).to(device)

        waveform, orig_sr = torchaudio.load(noisy_pth)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            waveform = T.Resample(orig_sr, sr)(waveform)

        spec = spec_fn(waveform.to(device))
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        real, imag = spec.real, spec.imag

        with torch.no_grad():
            den_real, den_imag = model(real.unsqueeze(1), imag.unsqueeze(1))
        den_spec = torch.complex(den_real.squeeze(1), den_imag.squeeze(1))
        waveform = inv_fn(den_spec).cpu()

    waveform = torch.clamp(waveform, -1.0, 1.0)
    torchaudio.save(output_pth, waveform, sr)
    print(f"✅ Denoised audio saved to: {output_pth}")
