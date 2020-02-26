#!/usr/bin/env bash

ENV="Hopper-v2"

for num_traj in 1 5 10
do
    for seed in 0 1 2 3 4
    do
        python3 train_gail.py --env_name $ENV --num_trajs $num_traj --seed $seed
            python3 train_gail.py --env_name $ENV --num_trajs $num_traj --seed $seed --good
    done
done
