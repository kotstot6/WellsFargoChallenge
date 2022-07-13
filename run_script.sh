#!/bin/bash

##SBATCH -N 1
#SBATCH -p publicgpu
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -q wildfire                     # Run job under wildfire QOS queue
#SBATCH -t 01-00:00                     # wall time (D-HH:MM)
#SBATCH -o outputs/output/%j                  # STDOUT (%j = JobId)
#SBATCH -e outputs/error/%j                  # STDERR (%j = JobId)
#SBATCH --mail-type=ALL                 # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu     # send-to address

module load anaconda/py3
source activate pytorch_env
cd $1
python3 $2 $3
conda deactivate
