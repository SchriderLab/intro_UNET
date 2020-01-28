#!/bin/bash
#SBATCH --job-name=cnn_binary_class_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G

unset OMP_NUM_THREADS

module load singularity/3.0.0

DATA=$1
IX=$2
OFILE=$3

echo singularity exec --nv -B /pine -B /proj ${SIMG_PATH} python3 src/models/get_evaluation_graphs.py --data ${DATA} --ix ${IX} --ofile ${OFILE}
singularity exec --nv -B /pine -B /proj ${SIMG_PATH} python3 src/models/get_evaluation_graphs.py --data ${DATA} --ix ${IX} --ofile ${OFILE}