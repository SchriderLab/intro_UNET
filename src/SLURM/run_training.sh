#!/bin/bash
#SBATCH --job-name=cnn_binary_class_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=35G

unset OMP_NUM_THREADS

module load singularity/3.0.0

MODEL_FILE=$1
DATA=$2
ODIR=$3
TAG=$4
BATCH_SIZE=$5
INDICES=$6
CONFIG=$7
N_GPUS=$8

mkdir -p ${ODIR}

date
hostname

echo singularity exec --nv -B /pine -B /proj ${SIMG_PATH} python3 src/models/train.py --model ${MODEL_FILE} --data ${DATA} --tag ${TAG} --gen_size ${BATCH_SIZE} --indices ${INDICES} --odir ${ODIR} --train_config ${CONFIG} --n_gpus ${N_GPUS}
singularity exec --nv -B /pine -B /proj ${SIMG_PATH} python3 src/models/train.py --model ${MODEL_FILE} --data ${DATA} --tag ${TAG} --gen_size ${BATCH_SIZE} --indices ${INDICES} --odir ${ODIR} --train_config ${CONFIG} --n_gpus ${N_GPUS}
