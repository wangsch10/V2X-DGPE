#!/usr/bin/env bash

set -x
YAML=$1
modeldir=$2
FUSION=$3

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

#CUDA_VISIBLE_DEVICES=1,2,3 
python  opencood/tools/train_w_kd.py -y ${YAML} --model_dir ${modeldir} --fusion_method ${FUSION}


