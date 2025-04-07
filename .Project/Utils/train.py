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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"



import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import gc
import time



# === Training & Evaluation Function ===
def train_eval(device, model, train_loader, val_loader, optimizer, criterion, epochs, accumulation_steps, save_pth, pto=False, scheduler=False):

    # Start the timer for the training process
    train_time = time.time()

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

    # Initialize PTO truncation time
    pto_truncation_time = 0.0

    # Loop through epochs
    for epoch in range(epochs):

        # Garbage collection and CUDA memory management
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Set model to training mode
        model.train()
        total_train_loss = 0.0

        # -- Training Loop ---
        for batch_idx, batch in enumerate(train_loader):
            
            # Load batch data
            if pto:
                tn_real, tn_imag, tc_real, tc_imag, orig_lengths = batch  
            else:
                tn_real, tn_imag, tc_real, tc_imag = batch  

            # Move data to device
            noisy_real, noisy_imag = tn_real.to(device), tn_imag.to(device)
            clean_real, clean_imag = tc_real.to(device), tc_imag.to(device)

            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision (FP16 rather than FP32)
            with autocast():
                outputs_real, outputs_imag = model(noisy_real, noisy_imag)

                # Handle PTO Dataset Truncation
                if pto:
                    # Start the timer for PTO train truncation
                    start = time.time()
                    
                    # Truncate the outputs and clean signals to the original lengths
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

                    # Compute and append the truncation time
                    end = time.time()
                    train_time = end - start
                    pto_truncation_time += train_time

                else:
                    truncated_real, truncated_imag = outputs_real, outputs_imag

            # Compute the true loss without scalling for plotting
            # ✅ Ensure inputs to loss are in float32 (AMP compatibility)
            raw_loss = criterion(truncated_real.float(), clean_real.float()) + criterion(truncated_imag.float(), clean_imag.float())

            # Append the true loss to the total training loss
            total_train_loss += raw_loss.item()

            # Divide loss before backprop to implement gradient accumulation
            loss = raw_loss / accumulation_steps
            # Apply gradient scaling for mixed precision
            scaler.scale(loss).backward()

            # Perform optimizer step every N mini-batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Compute average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Set model to evaluation mode
        model.eval()
        total_val_loss = 0.0
        snr_sum, snr_count = 0.0, 0  

        # -- Validation Loop ---
        with torch.no_grad():
            for batch in val_loader:

                # Load batch data
                if pto:
                    vn_real, vn_imag, vc_real, vc_imag, orig_lengths = batch
                else:
                    vn_real, vn_imag, vc_real, vc_imag = batch

                # Move data to device
                val_real, val_imag = vn_real.to(device), vn_imag.to(device)
                val_clean_real, val_clean_imag = vc_real.to(device), vc_imag.to(device)

                val_outputs_real, val_outputs_imag = model(val_real, val_imag)

                # Handle PTO Dataset Truncation
                if pto:
                    # Start the timer for PTO val truncation
                    start = time.time()

                    # Truncate the outputs and clean signals to the original lengths
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

                    # Compute and append the truncation time
                    end = time.time()
                    val_time = end - start
                    pto_truncation_time += val_time
                else:
                    val_truncated_real, val_truncated_imag = val_outputs_real, val_outputs_imag

                # Compute validation loss
                val_loss = criterion(val_truncated_real, val_clean_real) + criterion(val_truncated_imag, val_clean_imag)
                total_val_loss += val_loss.item()

                # Compute SNR for validation
                snr_real = (val_clean_real.norm() / (val_truncated_real - val_clean_real).norm()).item()
                snr_imag = (val_clean_imag.norm() / (val_truncated_imag - val_clean_imag).norm()).item()
                avg_snr = (snr_real + snr_imag) / 2
                snr_sum += avg_snr
                snr_count += 1

        # Compute average validation loss and SNR for the epoch
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_snr = snr_sum / snr_count 
        val_losses.append(avg_val_loss)
        val_snrs.append(avg_val_snr)

        # Print training and validation statistics
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val SNR: {avg_val_snr:.2f} dB")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_pth)
            print(f"✔ Model saved to {save_pth} (Best Val Loss: {best_val_loss:.4f})")

        # Update Learning Rate if Scheduler is Enabled
        if scheduler:
            lr_scheduler.step()

    print("--- Training Complete ---")

    # Print the total time taken for PTO truncation
    if pto:
        print(f"@ Time PTO Truncation: {pto_truncation_time:.2f} seconds")
    
    # Print the total training time
    total_time = time.time() - train_time
    print(f"@ Total Training Time: {total_time:.2f} seconds")
    print(f"@ Total Training Time: {total_time/60:.2f} minutes")
    print(f"@ Total Training Time: {total_time/3600:.2f} hours")

    # Save Loss & Accuracy Trends
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