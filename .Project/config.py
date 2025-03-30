"""
Author: Graham Pellegrini
Date: 2025-03-13
Project: UOM Final Year Project (FYP)

The configuration file for the project. This file contains all the parameters for the project. The unified location allows for easy access and modification of the parameters. The parameters are used throughout the project in the various scripts and modules.
"""

# Application Mode
MODE = "train"  # OPTIONS: "train", "denoise", "hyperparam"

# Dataset Parameters
DATASET_DIR = "../ED-Noisy-Speech-Datashare"  # Path to the dataset
SAMPLE_RATE = 48000  # Dataset sample rate
N_FFT = 1024  # Number of FFT points
HOP_LENGTH = 256  # Hop length for the STFT
BATCH_SIZE = 2  # Batch size for the model
ACCUMULATION_STEPS = 2  # Gradient accumulation steps
NUM_WORKERS = 4  # Number of workers for the DataLoader


# Padding Method
PAD_METHOD = "dynamic" # OPTIONS: "dynamic", "static",  "pto"
VISUALIZE = True  # Visualize the padding method
NUM_BUCKET = 5  # Number of dynamic buckets (dynamic only)


# Model Parameters
MODEL = "UNet"  # OPTIONS: "CNN", "CNN_CED", "RCED", "UNet", "ConvTasNet"
EPOCHS = 25  # Number of epochs to train the model
LEARNING_RATE = 1e-3  # Learning rate for the model
SCHEDULER = True  # Use a learning rate scheduler
MODEL_PTH = "Models/" + MODEL + "_" + PAD_METHOD + ".pth"  # Path to save the model


# Denoise Parameters
SINGLE = False  # Denoise a single audio file
METRICS_PTH = "Output/txt/" + MODEL + "_" + PAD_METHOD + "_metrics.txt"  # Path to save the metrics
NOISY_PTH = DATASET_DIR + "/noisy_testset_wav/p232_014.wav"  # Path to the noisy audio file
CLEAN_PTH = DATASET_DIR + "/clean_testset_wav/p232_014.wav"  # Path to the clean audio file
OUTPUT_PTH = "Output/wav/" + MODEL + "_" + PAD_METHOD + "_p232_014.wav"  # Path to save the denoised audio file


# Classical Denoising Parameters
CLASSICAL = False  # Use classical methods for denoising
CLASSICAL_METHOD = "wiener"  # OPTIONS: "spectral_sub", "wiener"