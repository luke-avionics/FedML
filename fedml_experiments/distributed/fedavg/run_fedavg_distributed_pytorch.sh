#!/usr/bin/env bash

CLIENT_NUM=4
WORKER_NUM=4
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
MODEL='resnet38'
DISTRIBUTION='homo'
ROUND=160
EPOCH=1
BATCH_SIZE=32
LR='0.1'
INFERENCE_BITS=8
schedule=(4 8)
CYCLIC_NUM_BITS_SCHEDULE=${schedule[@]}
LR_DECAY_STEP_SIZE=2000
DATASET='cifar100'
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
  --cyclic_num_bits_schedule ${CYCLIC_NUM_BITS_SCHEDULE} \
  --lr_decay_step_size ${LR_DECAY_STEP_SIZE}
