#!/bin/bash

# === SLURM Directives ===
#SBATCH --job-name=latex-build         # Job name
#SBATCH --output=logs/latex_output.log # Standard output log
#SBATCH --error=logs/latex_error.log   # Standard error log

# Optional email notifications
##SBATCH --mail-user=your-email@example.com
##SBATCH --mail-type=ALL

# Change to your LaTeX project directory
cd /path/to/your/latex/project || exit

# Clean previous builds
make clean

# Compile LaTeX document
make

echo "LaTeX build completed."
