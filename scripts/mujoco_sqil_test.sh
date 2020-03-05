#!/usr/bin/env bash

ENV="Walker2d-v2"

for seed in 0
do
    python3 train_ddpg_sqil.py --env_name $ENV --num_trajs 5 --seed $seed --good --new
    python3 train_ddpg_sqil.py --env_name $ENV --num_trajs 5 --seed $seed --good
done