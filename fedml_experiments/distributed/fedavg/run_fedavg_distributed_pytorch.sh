#!/usr/bin/env bash

CLIENT_NUM=8
WORKER_NUM=8
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
MODEL='resnet38'
DISTRIBUTION='hetero'
ROUND=250
EPOCH=40
BATCH_SIZE=64
LR='0.001'
DATASET='cifar100'
DATA_DIR="/home/yz87/FedML/fedml_experiments/distributed/fedavg"

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR
