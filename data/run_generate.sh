#!/usr/bin/env bash

#SBATCH -J dataset_generation
#SBATCH -o ./dataset_generation_id%j.out
#SBATCH --partition=cpu_guest
#SBATCH --qos=cpu_guest
##SBATCH --partition=gpu
##SBATCH --qos=gpu
##SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-10:00:00
#SBATCH --mem=256G

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate env-gnn-decoder

## Run training script
python generate_dataset.py

conda deactivate

