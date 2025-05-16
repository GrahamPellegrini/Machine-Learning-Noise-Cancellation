#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
#SBATCH --partition=teaching_gpu
#SBATCH --exclude=ict-d0-[01-03,13]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=7-00:00:00   # 7 days (168 hours)
##SBATCH --reservation=cce3015


# job parameters
#SBATCH --output=/opt/users/gpel0001/nnc-fyp/ssh/out/Notebook1.out
#SBATCH --error=/opt/users/gpel0001/nnc-fyp/ssh/err/Notebook1.err
#SBATCH --account=undergrad
#SBATCH --job-name=Notebook1

# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

# Define virtual environment activation path
VENV_PATH="/opt/users/gpel0001/nnc-fyp/nnc-venv/bin/activate"

# Activate virtual environment if it exists
if [ -f "$VENV_PATH" ]; then
    echo "Virtual environment found, activating"
    source "$VENV_PATH"
else
    echo "Virtual environment not found!"
    exit 1
fi

# Cd into the directory where the script is located
cd /opt/users/gpel0001/nnc-fyp/.Project

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:16

# Run notebook inplace 
python3 -m jupyter nbconvert --to notebook --execute --inplace wav_viz_bad.ipynb

# Indicate that the job has finished
echo "Job finished successfully"