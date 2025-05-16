import time
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from torch.nn import functional as F
from Utils.models import ConvTasNet

# === Configuration ===
SAMPLE_RATE = 48000
N_FFT = 1024
HOP_LENGTH = 256
CHUNK_DURATION = 1.0  # in seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MODEL_PATH = "Models/ConvTasNet_dynamic.pth"
OUTPUT_PATH = "Output/wav/realtime_denoised.wav"

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvTasNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Define Transforms ===
spectrogram = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=None).to(device)
inverse_spec = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to(device)

# === Audio Buffer ===
processed_audio = []
inference_times = []

# === Audio Callback Function ===
def callback(indata, frames, time_info, status):
    global processed_audio, inference_times

    if status:
        print(f"⚠️ Stream warning: {status}")

    # Convert numpy to tensor
    audio_tensor = torch.tensor(indata.T, dtype=torch.float32).to(device)
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # Convert to mono

    # Preprocessing
    spec = spectrogram(audio_tensor)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    real, imag = spec.real, spec.imag

    # Inference
    start = time.perf_counter()
    with torch.no_grad():
        denoised_real, denoised_imag = model(real.unsqueeze(0), imag.unsqueeze(0))
    end = time.perf_counter()

    inference_times.append(end - start)

    # Reconstruct audio
    denoised_spec = torch.complex(denoised_real.squeeze(0), denoised_imag.squeeze(0))
    waveform = inverse_spec(denoised_spec).cpu().squeeze().numpy()
    processed_audio.append(waveform)

# === Start Real-Time Stream ===
print(f"🎙️ Starting real-time denoising at {SAMPLE_RATE}Hz...")
with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
    input("Press Enter to stop recording...\n")

# === Save Denoised Output ===
final_audio = np.concatenate(processed_audio)
torchaudio.save(OUTPUT_PATH, torch.tensor(final_audio).unsqueeze(0), SAMPLE_RATE)
print(f"✅ Denoised audio saved to: {OUTPUT_PATH}")

# === Report Inference Time ===
avg_time = sum(inference_times) / len(inference_times)
print(f"⚡ Average Inference Time per Chunk: {avg_time:.4f} seconds")
print(f"📏 Real-Time Ratio: {(CHUNK_DURATION / avg_time):.2f}x")
