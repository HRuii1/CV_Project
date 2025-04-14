#!/bin/bash
#SBATCH --job-name=preprocess_msvd         # Name of your job
#SBATCH -A uva_cs4501_cv                     # Allocation group (you might still use this not cs4501)
#SBATCH --partition=gpu                    # Use GPU partition
#SBATCH --output=logs/preprocess_out.txt   # Save stdout here
#SBATCH --error=logs/preprocess_err.txt    # Save stderr here
#SBATCH --time=04:00:00                    # Adjust time
#SBATCH --mem=64G                          # More memory for CLIP + video decoding
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                       # One GPU

# Activate environment (adjust paths to yours)
module load python/3.11.4
module load cuda
source ~/venvs/msvd311/bin/activate

# Run the script
python load_mvsd_and_preprocessing_hf.py
