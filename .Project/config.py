"""
Author: Graham Pellegrini
Date: 2025-03-13
Project: UOM Final Year Project (FYP)

The configuration file for the project. This file contains all the parameters for the project. The unified location allows for easy access and modification of the parameters. The parameters are used throughout the project in the various scripts and modules.
"""

# Application Mode
MODE = "denoise"  # OPTIONS: "train" or "denoise"

# Dataset Parameters
DATASET_DIR = "../ED-Noisy-Speech-Datashare"  # Path to the dataset
SAMPLE_RATE = 48000  # Dataset sample rate
N_FFT = 1024  # Number of FFT points
HOP_LENGTH = 256  # Hop length for the STFT
BATCH_SIZE = 2 # Batch size for the model
ACCUMULATION_STEPS = 2 # Gradient accumulation steps
NUM_WORKERS = 4  # Number of workers for the DataLoader


# Padding Methods
PAD_METHOD = "dynamic" # OPTIONS: "dynamic", "static",  "pto"
VISUALIZE = False  # Visualize the padding method
NUM_BUCKET = 5  # Number of dynamic buckets (dynamic only)


# Model Parameters
MODEL = "UNet"  # OPTIONS: "CNN", "CED", "RCED", "UNet", "ConvTasNet"
EPOCHS = 25  # Number of epochs to train the model
LEARNING_RATE = 1e-3  # Learning rate for the model
SCHEDULER = True  # Use a learning rate scheduler
MODEL_PTH = "Models/25/" + MODEL + "_" + PAD_METHOD + ".pth"  # Path to save the model

# Classical Denoising Parameters
CLASSICAL = True  # Use classical methods for denoising
CLASSICAL_METHOD = "mmse_lsa"  # Options: 'baseline', 'spectral_sub', 'wiener', 'mmse_lsa'


# Denoise Parameters
SINGLE = True  # Denoise a single audio file
METRICS_PTH = (
    "Output/txt/" + CLASSICAL_METHOD + "_metrics.txt"
    if CLASSICAL else
    "Output/txt/" + MODEL + "_" + PAD_METHOD + "_metrics.txt"
)

SPEAKER_ID = "p232_334"  # Speaker ID for the audio file
NOISY_PTH = DATASET_DIR + "/noisy_testset_wav/" +  SPEAKER_ID + ".wav"  # Path to the noisy audio file
CLEAN_PTH = DATASET_DIR + "/clean_testset_wav/" + SPEAKER_ID + ".wav"  # Path to the clean audio file
OUTPUT_PTH = (
    "Output/wav/" + CLASSICAL_METHOD + "_" + SPEAKER_ID + ".wav"
    if CLASSICAL else
    "Output/wav/" + MODEL + "_" + PAD_METHOD + "_" + SPEAKER_ID + ".wav"
)

