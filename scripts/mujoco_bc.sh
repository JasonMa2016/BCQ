#!/usr/bin/env bash

TRAJ=5
ENV='Walker2d-v2'

for seed in 0 1 2 3 4
    do
        python3 train_bc.py --env_name $ENV --num_trajs $TRAJ --seed $seed --type "good"
        python3 train_bc.py --env_name $ENV --num_trajs $TRAJ --seed $seed --type "mixed"
    #    python3 train_bc.py --env_name $ENV --num_trajs $TRAJ --seed $seed --type "imperfect"
    done


