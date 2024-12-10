#!/bin/bash

# model path and name
MODEL="/data2/zujingliu/workspace/LLM/llama/meta-llama/Llama-2-7b-hf"
MODEL_NAME="llama-2-7b-hf"

# hyper params
BS=4
SEED=0
TRAIN=1000
DEV=100
EVAL=1000
STEPS=20000
EVAL_STEPS=500
NEPOCHS=5

# mode and task
MODE="lora"
TASK="WinoGrande"

# tag
TAG=mezo-$MODE-$STEPS-$BS-$SEED

# hyper params for zo
LEARNING_RATES=(1e-7)
ZO_EPS=(1e-4)
WEIGHT_DECAYS=(0)

for LR in "${LEARNING_RATES[@]}"; do
  for EPS in "${ZO_EPS[@]}"; do
    for WD in "${WEIGHT_DECAYS[@]}"; do
      echo "Running with learning_rate=$LR, zo_eps=$EPS and weight_decay=$WD"
      CUDA_VISIBLE_DEVICES=3 python run.py \
        --lora \
        --model_name $MODEL \
        --task_name $TASK \
        --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
        --max_steps $STEPS \
        --num_train_epochs $NEPOCHS --no_reparam \
        --trainer zo_sgd \
        --learning_rate $LR --zo_eps $EPS --weight_decay $WD --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
        --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
        --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
        --train_as_classification=False --perturbation_mode=two_side
    done
  done
done