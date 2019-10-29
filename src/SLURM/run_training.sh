#!/bin/bash
#SBATCH --job-name=cnn_binary_class_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=35G

unset OMP_NUM_THREADS

module load singularity/3.0.0

MODEL_FILE=$1

date
hostname

echo singularity exec --nv -B /pine -B /proj ${SIMG_PATH} python3 train.py --model ${MODEL_FILE}
singularity exec --nv -B /pine -B /proj ${SIMG_PATH} python3 train.py --model ${MODEL_FILE}
