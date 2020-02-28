#!/usr/bin/env bash

ENV="Hopper-v2"

for seed in 1 2 3 4
do
    python3 train_ddpg_sqil.py --env_name $ENV --num_trajs 5 --seed $seed --new
    python3 train_ddpg_sqil.py --env_name $ENV --num_trajs 5 --seed $seed

#    python3 train_ddpg_sqil_original.py --env_name $ENV --num_trajs 5 --seed $seed
done
