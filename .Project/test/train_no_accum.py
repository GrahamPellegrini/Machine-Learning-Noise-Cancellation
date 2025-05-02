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

import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import gc
import time



# === Training & Evaluation Function ===
def train_eval(device, model, train_loader, val_loader, optimizer, criterion, epochs, save_pth, pto=False, scheduler=False):
    # Start training timer
    start_time = time.time()

    # Move model to device
    model.to(device)

    # Optional scheduler
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Tracking variables
    best_val_loss = float('inf')
    train_losses, val_losses, val_snrs = [], [], []
    pto_truncation_time = 0.0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            # Unpack batch
            if pto:
                tn_real, tn_imag, tc_real, tc_imag, orig_lengths = batch
            else:
                tn_real, tn_imag, tc_real, tc_imag = batch

            # Move to device
            noisy_real, noisy_imag = tn_real.to(device), tn_imag.to(device)
            clean_real, clean_imag = tc_real.to(device), tc_imag.to(device)

            optimizer.zero_grad()

            # Forward
            outputs_real, outputs_imag = model(noisy_real, noisy_imag)

            # Handle PTO truncation
            if pto:
                start = time.time()
                truncated_real, truncated_imag = [], []
                clean_real_trunc, clean_imag_trunc = [], []

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

                pto_truncation_time += time.time() - start
            else:
                truncated_real, truncated_imag = outputs_real, outputs_imag

            # Loss and backward
            loss = criterion(truncated_real, clean_real) + criterion(truncated_imag, clean_imag)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        total_val_loss, snr_sum, snr_count = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                if pto:
                    vn_real, vn_imag, vc_real, vc_imag, orig_lengths = batch
                else:
                    vn_real, vn_imag, vc_real, vc_imag = batch

                val_real, val_imag = vn_real.to(device), vn_imag.to(device)
                val_clean_real, val_clean_imag = vc_real.to(device), vc_imag.to(device)

                val_outputs_real, val_outputs_imag = model(val_real, val_imag)

                if pto:
                    start = time.time()
                    val_truncated_real, val_truncated_imag = [], []
                    val_clean_real_trunc, val_clean_imag_trunc = [], []

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

                    pto_truncation_time += time.time() - start
                else:
                    val_truncated_real, val_truncated_imag = val_outputs_real, val_outputs_imag

                val_loss = criterion(val_truncated_real, val_clean_real) + criterion(val_truncated_imag, val_clean_imag)
                total_val_loss += val_loss.item()

                snr_real = (val_clean_real.norm() / (val_truncated_real - val_clean_real).norm()).item()
                snr_imag = (val_clean_imag.norm() / (val_truncated_imag - val_clean_imag).norm()).item()
                snr_sum += (snr_real + snr_imag) / 2
                snr_count += 1

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_snr = snr_sum / snr_count
        val_losses.append(avg_val_loss)
        val_snrs.append(avg_val_snr)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val SNR: {avg_val_snr:.2f} dB")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_pth)
            print(f"✔ Model saved to {save_pth} (Best Val Loss: {best_val_loss:.4f})")

        # Step LR scheduler
        if scheduler:
            lr_scheduler.step()

    # Final time printouts
    print("--- Training Complete ---")
    if pto:
        print(f"@ Time PTO Truncation: {pto_truncation_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"@ Total Training Time: {total_time:.2f} seconds")
    print(f"@ Total Training Time: {total_time/60:.2f} minutes")
    print(f"@ Total Training Time: {total_time/3600:.2f} hours")

    # Plot Loss & SNR
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, epochs+1), val_losses, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), val_snrs, label="Val SNR (dB)", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("SNR (dB)")
    plt.title("Validation SNR")
    plt.grid(True)
    plt.legend()

    # Save plot
    plt.savefig("Output/png/" + save_pth.split("/")[-1].replace(".pth", "_clean_plot.png"))
    plt.show()
