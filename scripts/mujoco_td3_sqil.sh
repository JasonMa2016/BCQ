#!/usr/bin/env bash

ENV="Hopper-v2"

for seed in 5 10 15 20
do
    python3 train_td3_sqil.py --env_name $ENV --num_trajs 5 --seed $seed
    python3 train_td3_sqil_original.py --env_name $ENV --num_trajs 5 --seed $seed
done