#!/bin/bash

train () {
  gpu=$1
  topo=$2

  CUDA_VISIBLE_DEVICES=$gpu \
  python train_botnet.py \
  --devid 0 \
  --data_dir ./data/botnet \
  --data_name "$topo" \
  --batch_size 2 \
  --enc_sizes 32 32 32 32 32 32 32 32 32 32 32 32 \
  --act relu \
  --residual_hop 1 \
  --deg_norm rw \
  --final proj \
  --epochs 50 \
  --lr 0.005 \
  --early_stop 1 \
  --save_dir ./saved_models \
  --save_name "$topo"_model_lay12_rh1_rw_ep50.pt
}

train 0 chord
#train 1 debru
#train 2 kadem
#train 3 leet
#train 0 c2
#train 1 p2p
