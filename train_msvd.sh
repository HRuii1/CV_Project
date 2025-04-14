#!/bin/bash
#SBATCH --job-name=train_msvd               # Change job name
#SBATCH -A uva_cs4501_cv                    # Allocation group
#SBATCH --partition=gpu                     # GPU partition
#SBATCH --output=logs/train_out.txt         # Stdout log
#SBATCH --error=logs/train_err.txt          # Stderr log
#SBATCH --time=08:00:00                     # Adjust time based on training needs
#SBATCH --mem=64G                           # Depends on model and batch size
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                        # Number of GPUs

# Load modules and activate your env
module load python/3.11.5
module load cuda
source ~/venvs/msvd311/bin/activate

# Move to your project directory if needed
cd ~/video_captioning/CV_Project

# Run the training script
python train.py
