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
#SBATCH --output=/opt/users/gpel0001/nnc-fyp/ssh/out/CED_STC.out
#SBATCH --error=/opt/users/gpel0001/nnc-fyp/ssh/err/CED_STC.err
#SBATCH --account=undergrad
#SBATCH --job-name=CED_STC

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run python script
python3 main.py

# Indicate that the job has finished
echo "Job finished successfully"