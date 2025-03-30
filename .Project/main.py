import os
import matplotlib.pyplot as plt
import optuna

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from Utils.train import train_eval
from Utils.optuna import objective
from Utils.denoise import denoise, single_denoise
from Utils.classical import classical, single_classical
from Utils.models import CNN, CNN_CED, RCED, UNet, ConvTasNet
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
    torchaudio.set_audio_backend("soundfile")

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
        print(f"Using `{pad_method}` dataset padding method.")
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
        
        # Visualize Dataset Padding
        if config.VISUALIZE:
            visualize_pth = "Output/png/" + pad_method + "_padding.png"
            visualize_dataset_padding(train_dataset, pad_method, visualize_pth)

        # === Load Model ===
        print(f"Initializing `{model_name}` model.")
        if model_name == "CNN":
            model = CNN()
        elif model_name == "CNN_CED":
            model = CNN_CED()
        elif model_name == "RCED":
            model = RCED()
        elif model_name == "UNet":
            model = UNet()
        elif model_name == "ConvTasNet":
            model = ConvTasNet()
        else:
            raise ValueError(f"Invalid MODEL: {model_name}\n Choose from ['CNN', 'UNet', 'ConvTasNet']")
        
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
        # Load Metrics Path and Padding Method
        metric_pth = config.METRICS_PTH
        pad_method = config.PAD_METHOD

        # Load Test Dataset
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
        
        # Classical Denoising
        if config.CLASSICAL:
            classical_method = config.CLASSICAL_METHOD

            # Single classical method
            if config.SINGLE:
                noisy_pth = config.NOISY_PTH
                output_pth ="Output/wav/" + noisy_pth.split("/")[-1].replace(".wav", f"_{classical_method}.wav")

                noisy_signal, orig_sr = torchaudio.load(noisy_pth)

                single_classical(noisy_signal, orig_sr, sr , classical_method, output_pth)

            else:
                classical(test_loader, sr, classical_method, pto=(pad_method == "pto"))

        # Model Denoising     
        else:
            # Load Model
            model_name = config.MODEL
            model_pth = config.MODEL_PTH

            # === Load Model ===
            print(f"Initializing `{model_name}` model.")
            if model_name == "CNN":
                model = CNN()
            elif model_name == "CNN_CED":
                model = CNN_CED()
            elif model_name == "RCED":
                model = RCED()
            elif model_name == "UNet":
                model = UNet()
            elif model_name == "ConvTasNet":
                model = ConvTasNet()
            else:
                raise ValueError(f"Invalid MODEL: {model_name}\n Choose from ['CNN', 'UNet', 'ConvTasNet', 'OptimizedConvTasNet']")
            
            
            # Single Model Denoising
            if config.SINGLE:
                noisy_pth = config.NOISY_PTH
                output_pth = config.OUTPUT_PTH

                single_denoise(device, model, model_pth, noisy_pth, output_pth, sr, n_fft, hop_length)

            else:
                # Model Denoising
                denoise(device, model, model_pth, test_loader, sr, n_fft, hop_length, metric_pth, pto=(pad_method == "pto"))

    elif config.MODE == "hyperparam":
        # Set the objective to minimize the validation loss
        study = optuna.create_study(direction="minimize") 
        # Run the optimization for 20 trials
        study.optimize(objective, n_trials=20) 

        # Print the best hyperparameters
        print("Best Hyperparameters:")
        print(study.best_trial.params)

        # Save the best hyperparameters to metric path
        path = config.METRICS_PTH.replace("metrics.txt", "hyperparams.txt")
        with open(path, "w") as f:
            f.write(str(study.best_trial.params))
            