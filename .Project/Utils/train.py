"""
Author: Graham Pellegrini
Date: 2025-03-13
Project: UOM Final Year Project (FYP)

The main training script for the Speech Enhancement Models.
This script loads the dataset, model, and training parameters from the config.py file.
It then trains the model using the specified dataset padding method and model architecture.
The trained model is saved to the specified path in the config.py file.
The training and validation loss trends are plotted and saved to the `Output/` directory.
"""

import os
# Set the environment variable to allow for expandable CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import gc


# === Training & Evaluation Function ===
def train_eval(device, model, train_loader, val_loader, optimizer, criterion, epochs, accumulation_steps, save_pth, pto=False, scheduler=False):

    # Move model to device
    model.to(device)

    # Learning Rate Scheduler (Optional)
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Define a scaler for mixed precision training
    scaler = GradScaler()

    # Training and Validation variables
    best_val_loss = float('inf')  
    train_losses, val_losses = [], []  
    val_snrs = []  

    # Training Loop over epochs
    for epoch in range(epochs):

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.train()
        total_train_loss = 0.0

        # Training Loop 
        for batch_idx, batch in enumerate(train_loader):
        
            if pto:
                tn_real, tn_imag, tc_real, tc_imag, orig_lengths = batch  
            else:
                tn_real, tn_imag, tc_real, tc_imag = batch  

            noisy_real, noisy_imag = tn_real.to(device), tn_imag.to(device)
            clean_real, clean_imag = tc_real.to(device), tc_imag.to(device)

            optimizer.zero_grad()
            
            # Forward pass with mixed precision (FP16 rather than FP32)
            with autocast():
                outputs_real, outputs_imag = model(noisy_real, noisy_imag)

                # Handle PTO Dataset
                if pto:
                    truncated_real = []
                    truncated_imag = []
                    clean_real_trunc = []
                    clean_imag_trunc = []

                    for i in range(outputs_real.shape[0]):
                        orig_len = orig_lengths[i].item()
                        truncated_real.append(outputs_real[i, :, :, :orig_len].clone())
                        truncated_imag.append(outputs_imag[i, :, :, :orig_len].clone())
                        clean_real_trunc.append(clean_real[i, :, :, :orig_len].clone())
                        clean_imag_trunc.append(clean_imag[i, :, :, :orig_len].clone())

                    truncated_real = torch.stack(truncated_real)
                    truncated_imag = torch.stack(truncated_imag)
                    clean_real = torch.stack(clean_real_trunc)
                    clean_imag = torch.stack(clean_imag_trunc)
                else:
                    truncated_real, truncated_imag = outputs_real, outputs_imag

                # Compute loss
                loss = criterion(truncated_real, clean_real) + criterion(truncated_imag, clean_imag)

            loss = loss / accumulation_steps  # Scale loss
            scaler.scale(loss).backward()

            # Perform optimizer step every N mini-batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


            total_train_loss += loss.item()

        print(f"🔍 Max GPU Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"🔍 Max GPU Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation Loop
        model.eval()
        total_val_loss = 0.0
        snr_sum, snr_count = 0.0, 0  

        with torch.no_grad():
            for batch in val_loader:
                if pto:
                    vn_real, vn_imag, vc_real, vc_imag, orig_lengths = batch
                else:
                    vn_real, vn_imag, vc_real, vc_imag = batch

                val_real, val_imag = vn_real.to(device), vn_imag.to(device)
                val_clean_real, val_clean_imag = vc_real.to(device), vc_imag.to(device)

                val_outputs_real, val_outputs_imag = model(val_real, val_imag)

                # Handle PTO Dataset
                if pto:
                    val_truncated_real = []
                    val_truncated_imag = []
                    val_clean_real_trunc = []
                    val_clean_imag_trunc = []

                    for i in range(val_outputs_real.shape[0]):
                        orig_len = orig_lengths[i].item()
                        val_truncated_real.append(val_outputs_real[i, :, :, :orig_len].clone())
                        val_truncated_imag.append(val_outputs_imag[i, :, :, :orig_len].clone())
                        val_clean_real_trunc.append(val_clean_real[i, :, :, :orig_len].clone())
                        val_clean_imag_trunc.append(val_clean_imag[i, :, :, :orig_len].clone())

                    val_truncated_real = torch.stack(val_truncated_real)
                    val_truncated_imag = torch.stack(val_truncated_imag)
                    val_clean_real = torch.stack(val_clean_real_trunc)
                    val_clean_imag = torch.stack(val_clean_imag_trunc)
                else:
                    val_truncated_real, val_truncated_imag = val_outputs_real, val_outputs_imag

                val_loss = criterion(val_truncated_real, val_clean_real) + criterion(val_truncated_imag, val_clean_imag)
                total_val_loss += val_loss.item()

                # Compute SNR
                snr_real = (val_clean_real.norm() / (val_truncated_real - val_clean_real).norm()).item()
                snr_imag = (val_clean_imag.norm() / (val_truncated_imag - val_clean_imag).norm()).item()
                avg_snr = (snr_real + snr_imag) / 2
                snr_sum += avg_snr
                snr_count += 1

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_snr = snr_sum / snr_count 
        val_losses.append(avg_val_loss)
        val_snrs.append(avg_val_snr)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val SNR: {avg_val_snr:.2f} dB")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_pth)
            print(f"✔ Model saved to {save_pth} (Best Val Loss: {best_val_loss:.4f})")

        # Update Learning Rate if Scheduler is Enabled
        if scheduler:
            lr_scheduler.step()

    print("🎉 Training complete!")

    # 🚀 Save Loss & Accuracy Trends
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, marker='o', color='b', label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, marker='s', color='r', label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), val_snrs, marker='^', color='g', label="Val SNR (dB)")
    plt.xlabel("Epochs")
    plt.ylabel("SNR (dB)")
    plt.title("Validation SNR Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.savefig("Output/png/"+save_pth.split("/")[-1].replace(".pth", "_plot.png"))
    plt.show()

    print("Plots saved to `Output/png` directory.")