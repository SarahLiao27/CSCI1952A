#!/bin/bash

# Request a GPU partition node and access to n GPUs
#SBATCH -p gpu --gres=gpu:1

# Number of CPU cores to request
#SBATCH -n 4

# Walltime
#SBATCH -t 04:00:00

# Job name
#SBATCH -J eureka

# Load a CUDA module
module load cuda

# Run program
# python eureka.py env=shadow_hand sample=4 iteration=2 model=gpt-4-0314