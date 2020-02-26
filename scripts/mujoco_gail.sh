#!/usr/bin/env bash

ENV="Hopper-v2"

for seed in 0 1 2 3 4
do
    python3 train_gail.py --env_name $ENV --num_trajs 5 --seed $seed
done
