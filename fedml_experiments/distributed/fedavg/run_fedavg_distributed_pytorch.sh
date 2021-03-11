#!/usr/bin/env bash

CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=8
MODEL='CNN_cifar'
DISTRIBUTION='one-class'
ROUND=500
EPOCH=1
BATCH_SIZE=100
LR='0.1'
INFERENCE_BITS=0
LR_DECAY_STEP_SIZE=2000
DATASET='cifar10'
DATA_DIR="/home/yf22/dataset"

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_server_num ${SERVER_NUM} \
  --gpu_num_per_server ${GPU_NUM_PER_SERVER} \
  --model ${MODEL} \
  --dataset ${DATASET} \
  --data_dir ${DATA_DIR} \
  --partition_method ${DISTRIBUTION}  \
  --client_num_in_total ${CLIENT_NUM} \
  --client_num_per_round ${WORKER_NUM} \
  --comm_round ${ROUND} \
  --epochs ${EPOCH} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --inference_bits ${INFERENCE_BITS} \
  --lr_decay_step_size ${LR_DECAY_STEP_SIZE} \
  #--use_fake_data
