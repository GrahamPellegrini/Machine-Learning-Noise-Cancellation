#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
##SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=7-00:00:00   # 7 days (168 hours)
##SBATCH --reservation=cce3015


# job parameters
#SBATCH --output=/opt/users/gpel0001/nnc-fyp/ssh/out/sudo.out
#SBATCH --error=/opt/users/gpel0001/nnc-fyp/ssh/err/sudo.err
#SBATCH --account=undergrad
#SBATCH --job-name=UNet

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

sudo apt update
sudo apt install git-filter-repo

pip install git-filter-repo


git filter-repo --path .Project/Models/UNet_dynamic.pth --invert-paths

echo ".Project/Models/UNet_dynamic.pth" >> .gitignore
git add .gitignore
git commit -m "Ignore model file going forward"
git push origin main --force
