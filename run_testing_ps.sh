#!/usr/bin/env bash

#SBATCH -J d5-d7-d9_testing_ps
#SBATCH -o ./d5-d7-d9_testing_ps_id%j.out
#SBATCH --partition=gpu_guest
#SBATCH --qos=gpu_guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100_vbd:1
#SBATCH --time=0-23:00:00
#SBATCH --mem=96G

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate env-gnn-decoder

## Run training script
python testing_ps.py

conda deactivate

