import os
import time
import matplotlib.pyplot as plt

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from Utils.train import train_eval
from Utils.denoise import batch_denoise, single_denoise
from Utils.models import CNN, CED, RCED, UNet, ConvTasNet
from Utils.dataset import DynamicBuckets, StaticBuckets, PTODataset, pto_collate, BucketSampler, visualize_dataset_padding



if __name__ == "__main__":

    # Check if GPU availability and set the device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    # Check if the device is set correctly
    print(f"Device set to: {device}")

    # Set the audio backend to soundfile (Uncomment if needed)
    # torchaudio.set_audio_backend("soundfile")

    # === Load Dataset Parameters ===
    dataset_dir = config.DATASET_DIR
    sr= config.SAMPLE_RATE
    n_fft = config.N_FFT
    hop_length = config.HOP_LENGTH
    batch_size = config.BATCH_SIZE
    accum_steps = config.ACCUMULATION_STEPS
    num_workers = config.NUM_WORKERS

    if config.MODE == "train":

        # === Load Padding Method Parameters ===
        pad_method = config.PAD_METHOD
        num_bucket = config.NUM_BUCKET
        bucket_sizes = [sr, sr * 2, sr * 3, sr * 4, sr * 5]

        # === Load Model Parameters ===
        model_name = config.MODEL
        epochs = config.EPOCHS
        lr = config.LEARNING_RATE
        model_pth = config.MODEL_PTH

        # === Load Dataset ===

        # Start the timing for dataset loading
        start_time = time.time()
        if pad_method == "dynamic":
            # Initialize DynamicBuckets
            train_dataset = DynamicBuckets(dataset_dir, "trainset_56spk", sr, n_fft, hop_length, num_bucket)
            val_dataset = DynamicBuckets(dataset_dir, "trainset_28spk", sr, n_fft, hop_length, num_bucket)

            # Initialize BucketSamplers
            train_sampler = BucketSampler(train_dataset.bucket_indices, batch_size=batch_size)
            val_sampler = BucketSampler(val_dataset.bucket_indices, batch_size=batch_size)

            # Initialize DataLoaders
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers)
        
        elif pad_method == "static":
            # Initialize StaticBuckets
            train_dataset = StaticBuckets(dataset_dir, "trainset_56spk", sr, n_fft, hop_length, bucket_sizes)
            val_dataset = StaticBuckets(dataset_dir, "trainset_28spk", sr, n_fft, hop_length, bucket_sizes)

            # Initialize BucketSamplers
            train_sampler = BucketSampler(train_dataset.bucket_indices, batch_size=batch_size)
            val_sampler = BucketSampler(val_dataset.bucket_indices, batch_size=batch_size)

            # Initialize DataLoaders
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers)

        elif pad_method == "pto":
            # Initialize PTODataset
            train_dataset = PTODataset(dataset_dir, "trainset_56spk", sr, n_fft, hop_length)
            val_dataset = PTODataset(dataset_dir, "trainset_28spk", sr, n_fft, hop_length)

            # Initialize DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pto_collate)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pto_collate)

        else:
            raise ValueError(f"Invalid PAD_METHOD: {pad_method}\n Choose from ['static', 'dynamic', 'pto']")
        # End the timing for dataset loading
        end_time = time.time()

        # Calculate the time taken for dataset loading
        loading_time = end_time - start_time

        # Dataset loaded completion message
        print(f"--- Dataset {dataset_dir} loaded successfully ---")

        # Print the time taken for dataset loading
        print(f"@ Time {dataset_dir} Loading: {loading_time:.2f} seconds")
        
        # Visualize Dataset Padding
        if config.VISUALIZE:
            visualize_pth = "Output/png/" + pad_method + "_padding.png"
            visualize_dataset_padding(train_dataset, pad_method, visualize_pth)

        # === Load Model ===
        if model_name == "CNN":
            model = CNN()
        elif model_name == "CED":
            model = CED()
        elif model_name == "RCED":
            model = RCED()
        elif model_name == "UNet":
            model = UNet()
        elif model_name == "ConvTasNet":   
            model = ConvTasNet()
        else:
            raise ValueError(f"Invalid MODEL: {model_name}\n Choose from ['CNN', 'UNet', 'ConvTasNet']")
        
        # Loaded model completion message
        print(f"--- Model `{model_name}` loaded successfully ---")
        
        # Define the optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # === Train & Evaluate Model ===
        print("Training & Evaluating Model...")
        train_eval(
            device,
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epochs,
            accum_steps,
            model_pth,
            pto=(pad_method == "pto"),
            scheduler=config.SCHEDULER
        )


    elif config.MODE == "denoise":
        # === Load padding method and test dataset ===
        pad_method = config.PAD_METHOD
        metric_pth = config.METRICS_PTH

        if pad_method == "dynamic":
            test_dataset = DynamicBuckets(dataset_dir, "testset", sr, n_fft, hop_length, config.NUM_BUCKET)
            test_sampler = BucketSampler(test_dataset.bucket_indices, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=num_workers)

        elif pad_method == "static":
            test_dataset = StaticBuckets(dataset_dir, "testset", sr, n_fft, hop_length, [sr, sr*2, sr*3, sr*4, sr*5])
            test_sampler = BucketSampler(test_dataset.bucket_indices, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=num_workers)

        elif pad_method == "pto":
            test_dataset = PTODataset(dataset_dir, "testset", sr, n_fft, hop_length)
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=pto_collate)

        else:
            raise ValueError(f"Invalid PAD_METHOD: {pad_method}")

        print(f"--- Test Dataset {dataset_dir} loaded successfully ---")

        # === Determine denoising path (classical or model-based) ===
        if config.CLASSICAL:
            model = None
            model_pth = None
        else:
            model_name = config.MODEL
            model_pth = config.MODEL_PTH

            if model_name == "CNN":
                model = CNN()
            elif model_name == "CED":
                model = CED()
            elif model_name == "RCED":
                model = RCED()
            elif model_name == "UNet":
                model = UNet()
            elif model_name == "ConvTasNet":  
                model = ConvTasNet()
                #model = ConvTasNet(enc_dim=128, feature_dim=48, kernel_size=(5,5),num_layers=3,num_stacks=1,)
                
            else:
                raise ValueError(f"Invalid MODEL: {model_name}")

            print(f"--- Model `{model_name}` loaded successfully ---")

        # === Run appropriate denoising mode ===
        if config.SINGLE:
            single_denoise(
                device,
                model,
                model_pth,
                config.NOISY_PTH,
                config.OUTPUT_PTH,
                sr,
                n_fft,
                hop_length,
                classical_method=config.CLASSICAL_METHOD if config.CLASSICAL else None
            )
        else:
            batch_denoise(
                device,
                model,
                model_pth,
                config.CLASSICAL_METHOD if config.CLASSICAL else None,
                test_loader,
                sr,
                n_fft,
                hop_length,
                metric_pth,
                pto=(pad_method == "pto")
            )
    else:
        raise ValueError(f"Invalid MODE: {config.MODE}\n Choose from ['train', 'denoise']")