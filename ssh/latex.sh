#!/bin/bash
# ALWAYS specify CPU and RAM resources needed as well as walltime
##SBATCH --partition=teaching_gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-24:00:00
##SBATCH --reservation=cce3015


# job parameters
#SBATCH --output=/opt/users/gpel0001/nnc-fyp/ssh/out/latex.out
#SBATCH --error=/opt/users/gpel0001/nnc-fyp/ssh/err/latex.err
#SBATCH --account=undergrad
#SBATCH --job-name=Latex

# email user with progress
#SBATCH --mail-user=graham.pellegrini.22@um.edu.mt
#SBATCH --mail-type=all

cd /opt/users/gpel0001/nnc-fyp/Template

# Make clean
make clean

# Make
make

