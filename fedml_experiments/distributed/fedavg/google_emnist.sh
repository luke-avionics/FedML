#!/usr/bin/env bash

CLIENT_NUM=3400
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=8
MODEL='cnn'
DISTRIBUTION='hetero'
ROUND=2000
EPOCH=10
BATCH_SIZE=20
LR='0.1'
INFERENCE_BITS=0
CYCLIC_NUM_BITS_SCHEDULE='None'
LR_DECAY_STEP_SIZE=2000
DATASET='femnist'
DATA_DIR="../../../data/FederatedEMNIST"

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
  --lr $LR \
  --inference_bits $INFERENCE_BITS \
  --cyclic_num_bits_schedule $CYCLIC_NUM_BITS_SCHEDULE \
  --lr_decay_step_size $LR_DECAY_STEP_SIZE
