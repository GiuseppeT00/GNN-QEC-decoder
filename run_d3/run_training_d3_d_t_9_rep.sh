#!/usr/bin/env bash
##SBATCH -t 7-00:00:00 			# time limit days-hours:minutes:seconds
##SBATCH -J d3_d_t_3
##SBATCH -o ./job_outputs/d3_d_t_3_id%j.out
##SBATCH --cpus-per-task=1
##SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

#SBATCH -J d3_d_t_9_rep
#SBATCH -o ./job_outputs/d3/d3_d_t_9_rep_id%j.out
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --time=0-23:59:59
#SBATCH --mem=92G

# (1) RAM=128 , funziona ;

# Set and export parameters
export CODE_SIZE=3
export REPETITIONS=9
## Training settings
export NUM_ITERATIONS=100
export BATCH_SIZE=1000 #1000 ; (1) 1000 ;
export LEARNING_RATE=0.0001
export MANUAL_SEED=12345
## Benchmark
export BENCHMARK=1
## Buffer settings
export BUFFER_SIZE=1000 #1000 o 100 ; (1) 10 ;
export REPLACEMENTS_PER_ITERATION=250 #250 o 25 ; (1) 2 ;
## test_size is len(error_rate) * batch_size * test_size
export TEST_SIZE=10 #10000 ; (1) 10 ;
## Graph settings
export NUM_NODE_FEATURES=5
export EDGE_WEIGHT_POWER=2
export M_NEAREST_NODES=6
export USE_CUDA=1
export USE_VALIDATION=1

## IO settings
export JOB_NAME=$SLURM_JOB_NAME
## Load old model:
#export RESUMED_TRAINING_FILE_NAME='d5_d_t_9_rep_epoch269'

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate env-gnn-decoder

## Run training script
python buffer_training.py

conda deactivate

