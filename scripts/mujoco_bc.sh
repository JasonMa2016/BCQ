#!/usr/bin/env bash

TRAJ=5

for env in Hopper-v2 Walker2d-v2
do
    for seed in 0 1 2
    do
        python3 train_bc.py --env_name $env --num_trajs $TRAJ --seed $seed --good
        python3 train_bc.py --env_name $env --num_trajs $TRAJ --seed $seed
    done
done


