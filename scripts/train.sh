#!/bin/bash

python ../dfd/runners/train.py \
--data ../../../data/datasets_local/deepfake_600_v2/ \
--model efficientnet_deepfake_v4 -b 3 --sched step --epochs 200 --decay-epochs 2 --decay-rate .92 \
--opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.35 --drop-path 0.25 \
--basic_lr 0.0000005 --num-classes 2 \
--input-size-v2 12,600,600 \
--amp --output ../../../data/output/ --model-version v5 \
--reprob 0.2 --remax 0.05 --pin-mem --color-jitter 0.1 \
--share_file /home/shiyu/deepfake/sharefile_dir/train_sharefile \
--master_share_file /data/shiyu/deepfake/sharefile_dir/train_sharefile \
--json_file ./train_server_config.json \
--validation-batch-size-multiplier 2 \
--validation_frac 0.2 --train_frac 0.8 \
--eval-metric loss \
--class_names fake,real \
--flicker 0.05 --rotate_range 5 --blur_prob 0.05 \
--bn-momentum 0.001 \
--mixup 0.1 \
--label_balance \
#--initial-checkpoint ../model/checkpoint-13.pth.tar