"""
Author: Graham Pellegrini
Date: 2025-03-13
Project: UOM Final Year Project (FYP)

This util module is used within both piplines for training and denoising. It includes the 3 different datasets used in the project that explore 3 different methods of handeling the variable length audio files in the dataset. The issue of fixed padding is mitigated differently in each and the different datasets can be chosen and compared to see which method is most effective for the task at hand. The datasets are used to train the models and evaluate their performance on the test set. The datasets are also used to evaluate the performance of the classical methods for comparison with the deep learning models. The respective BucketSampler and the pto_collate function are also defined within this file.
"""

import os
import pickle
import random
import numpy as np

import matplotlib.pyplot as plt

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

from torch.utils.data import Dataset, Sampler
from sklearn.cluster import KMeans


# === PTO/OT Dataset ===
class PTODataset(Dataset):
    def __init__(self, dataset_dir, set_type, rate, n_fft, hop_length):
        # Initialize parameters
        self.rate = rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cache_file = os.path.join("cache/pto", f"{set_type}_lengths.pkl")

        # Get all clean and noisy files
        clean_dir = os.path.join(dataset_dir, f"clean_{set_type}_wav")
        noisy_dir = os.path.join(dataset_dir, f"noisy_{set_type}_wav")
        # Sort the files by name and get the full path
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)], key=lambda x: os.path.basename(x))
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)], key=lambda x: os.path.basename(x))

        # Define the STFT transform to get real and imaginary parts
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        # Original lengths of the audio files
        self.lengths = self.compute_lengths()

    # Compute lengths of each audio file and cache them for the output cropping back to original length
    def compute_lengths(self):
        # If cache file exists, load it
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        # Otherwise, compute and cache
        lengths = [torchaudio.load(file)[0].shape[1] for file in self.clean_files]
        with open(self.cache_file, "wb") as f:
            pickle.dump(lengths, f)
        return lengths
    
    # Return the number of files in the dataset
    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load the audio files and the original length at idx
        noisy_waveform, n_sr = torchaudio.load(self.noisy_files[idx])
        clean_waveform, c_sr = torchaudio.load(self.clean_files[idx])
        orig_length = int(self.lengths[idx])

        # Convert stereo to mono
        noisy_waveform = noisy_waveform.mean(dim=0, keepdim=True) if noisy_waveform.shape[0] > 1 else noisy_waveform
        clean_waveform = clean_waveform.mean(dim=0, keepdim=True) if clean_waveform.shape[0] > 1 else clean_waveform

        # Resample
        if n_sr != self.rate:
            noisy_waveform = T.Resample(n_sr, self.rate)(noisy_waveform)
        if c_sr != self.rate:
            clean_waveform = T.Resample(c_sr, self.rate)(clean_waveform)

        # Compute STFT
        noisy_spec = self.spectrogram(noisy_waveform)
        clean_spec = self.spectrogram(clean_waveform)

        # Normalize
        noisy_spec = (noisy_spec - noisy_spec.mean()) / (noisy_spec.std() + 1e-6)
        clean_spec = (clean_spec - clean_spec.mean()) / (clean_spec.std() + 1e-6)

        # Return real and imaginary parts of the spectrograms, along with the original length
        return noisy_spec.real, noisy_spec.imag, clean_spec.real, clean_spec.imag, orig_length


# === Dynamic Bucketing Dataset ===
class DynamicBuckets(Dataset):
    def __init__(self, dataset_dir, set_type, rate, n_fft, hop_length, num_buckets=5):
        # Initialize parameters
        self.rate = rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cache_file = os.path.join("cache/dynamic/", f"{set_type}_buckets.pkl")

        # Get all clean and noisy files
        clean_dir = os.path.join(dataset_dir, f"clean_{set_type}_wav")
        noisy_dir = os.path.join(dataset_dir, f"noisy_{set_type}_wav")
        # Sort the files by name and get the full path
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)], key=lambda x: os.path.basename(x))
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)], key=lambda x: os.path.basename(x))

        # Define the STFT transform to get real and imaginary parts
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)

        # Load or compute bucket assignments
        self.bucket_sizes, self.bucket_indices = self.bucket_handler(num_buckets)

    # Function to compute the optimal bucket sizes
    def compute_bucket_sizes(self, num_buckets):
        print("Analyzing dataset for optimal bucket sizes...")
        lengths = []
        for file in self.clean_files:
            waveform, _ = torchaudio.load(file)
            lengths.append(waveform.shape[1])

        lengths = np.array(lengths).reshape(-1, 1)  # Reshape for clustering

        # Apply K-Means clustering to determine bucket centers
        kmeans = KMeans(n_clusters=num_buckets, random_state=42, n_init=10)
        kmeans.fit(lengths)
        bucket_sizes = np.sort(kmeans.cluster_centers_.flatten()).astype(int)
        print(f"Dynamic bucket sizes: {bucket_sizes}")

        return bucket_sizes

    # Function to handle the bucketing of the audio files with the dynamic bucket sizes
    def bucket_handler(self, num_buckets):
        # If cache file exists, load it
        if os.path.exists(self.cache_file):
            print(f"Loading cached bucket assignments from {self.cache_file}...")
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)

        # Otherwise, compute and cache
        bucket_sizes = self.compute_bucket_sizes(num_buckets)
        bucket_indices = []

        print(f"Assigning files to {len(bucket_sizes)} buckets...")
        for file in self.clean_files:
            waveform, _ = torchaudio.load(file)
            length = waveform.shape[1]
            # Find the closest bucket size
            bucket_idx = np.argmin(np.abs(bucket_sizes - length))
            bucket_indices.append(bucket_idx)

        # Save to cache
        with open(self.cache_file, "wb") as f:
            pickle.dump((bucket_sizes, bucket_indices), f)

        return bucket_sizes, bucket_indices
    
    # Padding method for the dynamic bucketing
    def collate(self, waveform, target_length):
        if waveform.shape[1] < target_length:
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:, :target_length]
        return waveform

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load noisy & clean audio
        noisy_waveform, n_sr = torchaudio.load(self.noisy_files[idx])
        clean_waveform, c_sr = torchaudio.load(self.clean_files[idx])

        # Convert stereo to mono
        noisy_waveform = noisy_waveform.mean(dim=0, keepdim=True) if noisy_waveform.shape[0] > 1 else noisy_waveform
        clean_waveform = clean_waveform.mean(dim=0, keepdim=True) if clean_waveform.shape[0] > 1 else clean_waveform

        # Resample if needed
        if n_sr != self.rate:
            noisy_waveform = T.Resample(n_sr, self.rate)(noisy_waveform)
        if c_sr != self.rate:
            clean_waveform = T.Resample(c_sr, self.rate)(clean_waveform)

        # Get bucket size and pad/truncate
        bucket_idx = self.bucket_indices[idx]
        target_length = self.bucket_sizes[bucket_idx]
        noisy_waveform = self.collate(noisy_waveform, target_length)
        clean_waveform = self.collate(clean_waveform, target_length)

        # Compute STFT (Complex Spectrogram)
        noisy_spec = self.spectrogram(noisy_waveform)
        clean_spec = self.spectrogram(clean_waveform)

        # Normalize spectrograms
        noisy_spec = (noisy_spec - noisy_spec.mean()) / (noisy_spec.std() + 1e-6)
        clean_spec = (clean_spec - clean_spec.mean()) / (clean_spec.std() + 1e-6)

        # Separate into real & imaginary components
        return noisy_spec.real, noisy_spec.imag, clean_spec.real, clean_spec.imag

# === Static Bucketing Dataset ===
class StaticBuckets(Dataset):
    def __init__(self, dataset_dir, set_type, rate, n_fft, hop_length, bucket_sizes):
        # Initialize parameters
        self.rate = rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bucket_sizes = bucket_sizes
        self.cache_file = os.path.join("cache/static/", f"{set_type}_buckets.pkl")

        # Get all clean and noisy files
        clean_dir = os.path.join(dataset_dir, f"clean_{set_type}_wav")
        noisy_dir = os.path.join(dataset_dir, f"noisy_{set_type}_wav")
        # Sort the files by name and get the full path
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)], key=lambda x: os.path.basename(x))
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)], key=lambda x: os.path.basename(x))

        # Define the STFT transform to get real and imaginary parts
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)

        # Compute bucket indices
        self.bucket_indices = self.bucket_handler()

    # Function to handle the bucketing of the audio files with the static bucket sizes
    def bucket_handler(self):
        # If cache file exists, load it
        if os.path.exists(self.cache_file):
            print(f"Loading cached bucket assignments from {self.cache_file}...")
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        # Otherwise, compute and cache
        else:
            print(f"Computing bucket assignments and caching to {self.cache_file}...")
            bucket_indices = []
            for file in self.clean_files:
                waveform, _ = torchaudio.load(file)
                length = waveform.shape[1]
                bucket_idx = min([i for i, b in enumerate(self.bucket_sizes) if length <= b], default=len(self.bucket_sizes) - 1)
                bucket_indices.append(bucket_idx)

            # Save to cache
            with open(self.cache_file, "wb") as f:
                pickle.dump(bucket_indices, f)

            return bucket_indices

    # Padding method for the static bucketing
    def collate(self, waveform, target_length):
        # Pad or truncate waveform to target length
        if waveform.shape[1] < target_length:
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:, :target_length]
        return waveform

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load noisy & clean audio
        noisy_waveform, n_sr = torchaudio.load(self.noisy_files[idx])
        clean_waveform, c_sr = torchaudio.load(self.clean_files[idx])

        # Convert stereo to mono
        noisy_waveform = noisy_waveform.mean(dim=0, keepdim=True) if noisy_waveform.shape[0] > 1 else noisy_waveform
        clean_waveform = clean_waveform.mean(dim=0, keepdim=True) if clean_waveform.shape[0] > 1 else clean_waveform

        # Resample if needed
        if n_sr != self.rate:
            noisy_waveform = T.Resample(n_sr, self.rate)(noisy_waveform)
        if c_sr != self.rate:
            clean_waveform = T.Resample(c_sr, self.rate)(clean_waveform)

        # Get bucket size and pad/truncate
        bucket_idx = self.bucket_indices[idx]
        target_length = self.bucket_sizes[bucket_idx]
        noisy_waveform = self.collate(noisy_waveform, target_length)
        clean_waveform = self.collate(clean_waveform, target_length)

        # Compute STFT (Complex Spectrogram)
        noisy_spec = self.spectrogram(noisy_waveform)
        clean_spec = self.spectrogram(clean_waveform)

        # Normalize the spectrograms
        noisy_spec = (noisy_spec - noisy_spec.mean()) / (noisy_spec.std() +1e-6)
        clean_spec = (clean_spec - clean_spec.mean()) / (clean_spec.std() +1e-6)

        # Separate into real & imaginary components
        noisy_real, noisy_imag = noisy_spec.real, noisy_spec.imag
        clean_real, clean_imag = clean_spec.real, clean_spec.imag
        
        return noisy_real, noisy_imag, clean_real, clean_imag
    
# === PTO/OT Collate Function ===
def pto_collate(batch):

    tn_real, tn_imag, tc_real, tc_imag, orig_lengths = zip(*batch)

    # 🚀 Step 1: Ensure a Fixed `F` Dimension
    target_F = max(x.shape[1] for x in tn_real)  # Find max `F`
    tn_real = [F.interpolate(x.unsqueeze(0), size=(target_F, x.shape[2]), mode="bilinear", align_corners=False).squeeze(0) for x in tn_real]
    tn_imag = [F.interpolate(x.unsqueeze(0), size=(target_F, x.shape[2]), mode="bilinear", align_corners=False).squeeze(0) for x in tn_imag]
    tc_real = [F.interpolate(x.unsqueeze(0), size=(target_F, x.shape[2]), mode="bilinear", align_corners=False).squeeze(0) for x in tc_real]
    tc_imag = [F.interpolate(x.unsqueeze(0), size=(target_F, x.shape[2]), mode="bilinear", align_corners=False).squeeze(0) for x in tc_imag]

    # 🚀 Step 2: Find Maximum `T` and Pad to It (Handles All Lengths Properly)
    max_T = max(x.shape[2] for x in tn_real)  # Find max `T`
    tn_real = [F.pad(x, (0, max_T - x.shape[2])) for x in tn_real]
    tn_imag = [F.pad(x, (0, max_T - x.shape[2])) for x in tn_imag]
    tc_real = [F.pad(x, (0, max_T - x.shape[2])) for x in tc_real]
    tc_imag = [F.pad(x, (0, max_T - x.shape[2])) for x in tc_imag]

    # Convert to stacked tensors
    tn_real = torch.stack(tn_real)
    tn_imag = torch.stack(tn_imag)
    tc_real = torch.stack(tc_real)
    tc_imag = torch.stack(tc_imag)

    # Convert `orig_lengths` to a tensor
    orig_lengths = torch.tensor(orig_lengths, dtype=torch.int64)

    return tn_real, tn_imag, tc_real, tc_imag, orig_lengths

# === Dynamic/Static Bucketing Collate Function ===
class BucketSampler(Sampler):
    def __init__(self, bucket_indices, batch_size):
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.buckets = {}
        
        # Group indices by bucket
        for idx, bucket in enumerate(bucket_indices):
            if bucket not in self.buckets:
                self.buckets[bucket] = []
            self.buckets[bucket].append(idx)
        
        self.bucket_batches = []
        for bucket, indices in self.buckets.items():
            random.shuffle(indices)  # Shuffle within bucket
            self.bucket_batches.extend([indices[i:i+batch_size] for i in range(0, len(indices), batch_size)])
        
        # Shuffle the buckets once there is no order anymore
        random.shuffle(self.bucket_batches)

    def __iter__(self):
        for batch in self.bucket_batches:
            yield batch

    def __len__(self):
        return len(self.bucket_batches)
    

# === Visualization Function ===
def visualize_dataset_padding(dataset, method, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract original waveform lengths
    if method in ["dynamic", "static"]:  # Dynamic/Static Bucketing
        original_lengths = [torchaudio.load(f)[0].shape[1] for f in dataset.clean_files]
    elif method == "pto":  # Padding-Truncation Output-Truncation
        original_lengths = dataset.lengths
    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'dynamic', 'static', 'pto'.")

    # Get sampling rate (default to 16kHz if unknown)
    sr = dataset.rate if hasattr(dataset, 'rate') else 16000  

    # Histogram of Original Lengths
    axes[0].hist(original_lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Waveform Length (samples)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{method.capitalize()} - Original Waveform Length Distribution")
    axes[0].grid(True)

    # If dataset uses Bucketing (Static/Dynamic)
    if method in ["dynamic", "static"] and hasattr(dataset, 'bucket_sizes'):
        bucket_sizes = dataset.bucket_sizes
        bucket_counts = [dataset.bucket_indices.count(i) for i in range(len(bucket_sizes))]

        # Plot Bar Chart of Buckets
        axes[1].bar(bucket_sizes, bucket_counts, width=sr // 2, color='orange', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Bucket Size (samples)")
        axes[1].set_ylabel("Number of Samples")
        axes[1].set_title(f"{method.capitalize()} - Bucketed Distribution")
        axes[1].set_xticks(bucket_sizes)
        axes[1].set_xticklabels([f"{b//sr}s" for b in bucket_sizes])  # Convert to seconds
        axes[1].grid(True)

    # If dataset uses PTO (Padding-Truncation Output-Truncation)
    elif method == "pto" and hasattr(dataset, 'lengths'):
        max_padded_length = max(original_lengths)
        axes[1].bar(["Original", "Padded"], 
                    [sum(original_lengths), len(original_lengths) * max_padded_length],
                    color=['blue', 'orange'], alpha=0.7)
        axes[1].set_ylabel("Total Length (samples)")
        axes[1].set_title(f"{method.capitalize()} - Effect of Padding (PTO)")
        axes[1].grid(True)

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved at: {save_path}")
    else:
        plt.show()
