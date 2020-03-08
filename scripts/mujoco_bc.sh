#!/usr/bin/env bash

TRAJ=1

for env in Hopper-v2 Walker2d-v2
do
    for seed in 0
    do
        python3 train_bc.py --env_name $env --num_trajs $TRAJ --seed $seed --type "good"
        python3 train_bc.py --env_name $env --num_trajs $TRAJ --seed $seed --type "mixed"
        python3 train_bc.py --env_name $env --num_trajs $TRAJ --seed $seed --type "imperfect"
    done
done


