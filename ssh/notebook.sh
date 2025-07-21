#!/bin/bash

# === SLURM Directives ===
#SBATCH --job-name=notebook-job              # Job name
#SBATCH --output=logs/notebook_output.log    # Stdout log
#SBATCH --error=logs/notebook_error.log      # Stderr log

# Optional email notifications
##SBATCH --mail-user=your-email@example.com
##SBATCH --mail-type=ALL

# Define virtual environment path (optional)
VENV_PATH="/path/to/venv/bin/activate"

# Activate virtual environment
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Change to notebook directory
cd /path/to/your/project || exit

# Optional: PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:16

# Execute notebook in-place
python3 -m jupyter nbconvert --to notebook --execute --inplace wav_viz_bad.ipynb

echo "Notebook execution completed successfully."
