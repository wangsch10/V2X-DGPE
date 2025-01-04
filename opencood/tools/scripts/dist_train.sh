#!/usr/bin/env bash

set -x
NGPUS=$1
YAML=$2
FUSION=$3
#DIR=$4



while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m torch.distributed.launch  --master_port=$PORT --nproc_per_node=${NGPUS} --use_env opencood/tools/train_ddp.py -y ${YAML} --fusion_method ${FUSION} #--model_dir ${DIR} 
#python -m torch.distributed.launch --master_port=$PORT  --nproc_per_node=${NGPUS}  train.py --launcher pytorch ${PY_ARGS}
#python -m torch.distributed.launch  --master_port=$PORT --nproc_per_node=${NGPUS} --use_env opencood/tools/train_ddp.py -y ${YAML} --model_dir ${DIR} --fusion_method ${FUSION} 

