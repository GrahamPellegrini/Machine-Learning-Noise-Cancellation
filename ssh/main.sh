#!/bin/bash

# === SLURM Directives ===
#SBATCH --job-name=denoising-job        # Job name
#SBATCH --output=logs/output.log        # Standard output log
#SBATCH --error=logs/error.log          # Standard error log

# Define virtual environment activation path
VENV_PATH="your_virtualenv_path/bin/activate"

# Activate virtual environment if it exists
if [ -f "$VENV_PATH" ]; then
    echo "Virtual environment found, activating..."
    source "$VENV_PATH"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Change to the project directory
cd /path/to/your/.Project || exit

# Optional PyTorch memory setting
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:16

# Run the Python script
python3 main.py

echo "Job finished successfully."
